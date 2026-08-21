"""
Inference wrapper — the ONLY module the UI (app.py) imports for verdicts.

Exposes:
    predict(file_path) -> dict     # branches on extension: image vs video
    predict_image(path) -> dict
    predict_video(path, fps_sample=2.0) -> dict

Uses:
    faces.py       for face detection + video sampling (Track 2 teammate)
    model_best.pt  for the trained classifier weights (my training run)

Label convention:
    output score in [0, 1] = probability the input is FAKE
    verdict = "FAKE" if score > 0.5 else "REAL"
"""
from __future__ import annotations
import io, os, time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
from torchvision import transforms, models

try:
    from faces import detect_and_crop, video_face_crops
    try:
        from faces import detect_with_boxes, video_face_crops_boxes
        _HAS_BOX_API = True
    except ImportError:
        _HAS_BOX_API = False
except Exception as e:
    raise ImportError(
        "predict.py requires faces.py from Track 2 to be in the same folder"
    ) from e


# ==================== Simple IoU tracker + EMA ====================

def _iou(a, b) -> float:
    """IoU of two boxes in xywh format."""
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1 = max(ax, bx); y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw); y2 = min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1: return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1.0)


class IoUTracker:
    """
    Frame-by-frame IoU tracker with a fixed disappear budget.
    Assigns persistent integer face_ids to overlapping bboxes across frames.
    """
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 8):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.next_id = 0
        # id -> {"bbox": xywh, "last_seen": frame_idx, "missed": int}
        self.tracks: dict[int, dict] = {}

    def update(self, frame_idx: int,
               detections: list[tuple[int, int, int, int]]
               ) -> list[int]:
        """Assign a face_id to each detection. Returns list of face_ids parallel to detections."""
        # 1. Match detections to existing tracks by best-IoU
        assignments: list[Optional[int]] = [None] * len(detections)
        used_track_ids: set[int] = set()
        # Greedy match: for each detection, find best-matching unused track
        for det_i, det in enumerate(detections):
            best_id, best_iou = None, self.iou_threshold
            for tid, track in self.tracks.items():
                if tid in used_track_ids: continue
                score = _iou(det, track["bbox"])
                if score > best_iou:
                    best_iou = score; best_id = tid
            if best_id is not None:
                assignments[det_i] = best_id
                used_track_ids.add(best_id)
        # 2. New tracks for unmatched detections
        for det_i, det in enumerate(detections):
            if assignments[det_i] is None:
                new_id = self.next_id; self.next_id += 1
                self.tracks[new_id] = {"bbox": det, "last_seen": frame_idx, "missed": 0}
                assignments[det_i] = new_id
            else:
                tid = assignments[det_i]
                self.tracks[tid]["bbox"] = det
                self.tracks[tid]["last_seen"] = frame_idx
                self.tracks[tid]["missed"] = 0
        # 3. Age & prune tracks that were not matched this frame
        for tid, track in list(self.tracks.items()):
            if track["last_seen"] != frame_idx:
                track["missed"] += 1
                if track["missed"] > self.max_missed:
                    del self.tracks[tid]
        return assignments  # type: ignore


class EMASmoother:
    """Exponential moving average per key. alpha computed from window N."""
    def __init__(self, window: int = 15):
        self.alpha = 2.0 / (window + 1)
        self.state: dict[int, float] = {}
    def update(self, key: int, value: float) -> float:
        prev = self.state.get(key, value)  # first observation is not smoothed
        smoothed = self.alpha * value + (1 - self.alpha) * prev
        self.state[key] = smoothed
        return smoothed
    def peek(self, key: int) -> Optional[float]:
        return self.state.get(key)

# ---- config ----
# Ensemble: v1 (Hemg-only) + v2 (Hemg + SD + aug). Take max score.
# Rationale: v1 is more suspicious, v2 handles OOD real photos.
# Also analyze BOTH tight face + wider context (face-swap artifacts live at boundary)
_PROJECT_ROOT  = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH   = os.environ.get("MODEL_WEIGHTS", os.path.join(_PROJECT_ROOT, "models", "model_v3_best.pt"))
WEIGHTS_PATH_2 = os.environ.get("MODEL_WEIGHTS_2", os.path.join(_PROJECT_ROOT, "models", "model_v2_best.pt"))
IMG_SIZE       = 224
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME     = "Ensemble: EfficientNet-B0 v3+v2 (CIFAKE+SD+aug) + Ateeqq-ViT"
MODEL_VERSION  = "5.0-tri-ensemble-cifake"
ATEEQQ_PATH    = os.path.join(_PROJECT_ROOT, "models", "ateeqq_aivshuman")
ATEEQQ_WEIGHT  = 0.0        # AI-generated-image detector excluded from deepfake verdict
LOCAL_WEIGHT   = 1.0        # our own models keep full weight
FPS_SAMPLE     = 4.0    # more frames per second (was 2) — better video coverage
TOP_K_FRAMES   = 5
MIN_FRAMES     = 8      # if video is short, sample at least this many frames
FAKE_THRESHOLD = 0.35   # calibrated: catches SD/StyleGAN without flooding real photos
WIDE_MARGIN = 0.9
MIN_FACE_AREA_FRAC = 0.005
MIN_DETECTION_CONF = 0.25
MIN_RELIABLE_BLUR = 50.0
MIN_TRACK_OBSERVATIONS = 3
MIN_RELIABLE_BLUR = 50.0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ---- lazy ensemble of models ----
_MODELS: list[nn.Module] = []
_MODEL_META: dict = {}
_ATEEQQ_PROC = None
_ATEEQQ_MODEL = None
_ATEEQQ_AI_IDX = 0


def _build_arch() -> nn.Module:
    m = models.efficientnet_b0(weights=None)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(0.3, inplace=True), nn.Linear(in_f, 1))
    return m


def _load_one(path: str) -> Optional[nn.Module]:
    if not os.path.exists(path): return None
    ckpt = torch.load(path, map_location=DEVICE)
    m = _build_arch()
    m.load_state_dict(ckpt["model"])
    m.to(DEVICE).eval()
    return m, ckpt


def _load_models():
    global _MODELS, _MODEL_META
    if _MODELS: return _MODELS
    loaded = []
    metas = []
    for p in (WEIGHTS_PATH, WEIGHTS_PATH_2):
        r = _load_one(p)
        if r is not None:
            m, ckpt = r
            loaded.append(m)
            metas.append({"path": p, "val_auc": ckpt.get("val_auc"),
                          "version": ckpt.get("version", "?")})
    if not loaded:
        raise FileNotFoundError(f"No model weights found ({WEIGHTS_PATH}, {WEIGHTS_PATH_2})")
    _MODELS = loaded
    _MODEL_META = {"models": metas, "n_models": len(loaded),
                   "threshold": FAKE_THRESHOLD}
    return _MODELS


# back-compat single-model accessor
def _load_model() -> nn.Module:
    return _load_models()[0]


# --------------- core inference ---------------

def _bgr_to_tensor(bgr: np.ndarray) -> torch.Tensor:
    """224x224 BGR uint8 -> 1x3x224x224 float, ImageNet-normalized."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return _preprocess(pil)


def _load_ateeqq():
    """Load Ateeqq's HuggingFace AI-vs-Human ViT model."""
    global _ATEEQQ_PROC, _ATEEQQ_MODEL, _ATEEQQ_AI_IDX
    if _ATEEQQ_MODEL is not None:
        return _ATEEQQ_PROC, _ATEEQQ_MODEL, _ATEEQQ_AI_IDX
    if not os.path.exists(ATEEQQ_PATH):
        return None, None, 0
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        _ATEEQQ_PROC  = AutoImageProcessor.from_pretrained(ATEEQQ_PATH)
        _ATEEQQ_MODEL = AutoModelForImageClassification.from_pretrained(ATEEQQ_PATH).to(DEVICE).eval()
        for i, l in _ATEEQQ_MODEL.config.id2label.items():
            if any(k in l.lower() for k in ("ai", "fake", "gen", "syn")):
                _ATEEQQ_AI_IDX = i; break
    except Exception as e:
        print(f"[predict] Ateeqq load failed, continuing without: {e}")
        _ATEEQQ_MODEL = None
    return _ATEEQQ_PROC, _ATEEQQ_MODEL, _ATEEQQ_AI_IDX


@torch.no_grad()
def _score_batch(crops_bgr: list[np.ndarray]) -> np.ndarray:
    """
    Tri-model ensemble: v1 + v2 EfficientNets (MAX between them) then
    WEIGHTED MEAN with Ateeqq ViT.
        combined = (LOCAL_WEIGHT * max(v1,v2) + ATEEQQ_WEIGHT * ateeqq) /
                   (LOCAL_WEIGHT + ATEEQQ_WEIGHT)
    Cancels individual model biases while keeping deepfake sensitivity.
    """
    if not crops_bgr:
        return np.array([], dtype=np.float32)
    ms = _load_models()
    batch = torch.stack([_bgr_to_tensor(c) for c in crops_bgr]).to(DEVICE)

    # Our two local models
    local_scores = []
    for m in ms:
        with torch.amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
            logits = m(batch).squeeze(1)
        local_scores.append(torch.sigmoid(logits).float().cpu().numpy())
    local_max = np.stack(local_scores, axis=0).max(axis=0)   # worst-of-local

    # Ateeqq ViT (optional; skips gracefully if not present)
    if ATEEQQ_WEIGHT <= 0:
        return local_max
    proc, ateeqq, ai_idx = _load_ateeqq()
    if ateeqq is None:
        return local_max
    try:
        pil_imgs = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr]
        x = proc(images=pil_imgs, return_tensors="pt").to(DEVICE)
        with torch.amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
            logits = ateeqq(**x).logits
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        ateeqq_scores = probs[:, ai_idx]
    except Exception:
        return local_max

    combined = (LOCAL_WEIGHT * local_max + ATEEQQ_WEIGHT * ateeqq_scores) / (LOCAL_WEIGHT + ATEEQQ_WEIGHT)
    return combined


def _verdict_from_score(score: float, band: float = 0.05) -> tuple[str, float]:
    """
    Return (verdict, confidence).
    Uses FAKE_THRESHOLD (default 0.35) — more sensitive than 0.5 to catch
    face-swap deepfakes the models are under-confident on.
    """
    thr = FAKE_THRESHOLD
    if abs(score - thr) < band:
        return "INCONCLUSIVE", 0.0
    if score > thr:
        return "FAKE", float(min(1.0, (score - thr) / max(1 - thr, 1e-6)))
    return "REAL", float(min(1.0, (thr - score) / max(thr, 1e-6)))


_GRADCAM = None

def _get_gradcam():
    """Lazy Grad-CAM instance targeting last conv block of Model 1."""
    global _GRADCAM
    if _GRADCAM is not None: return _GRADCAM
    try:
        from pytorch_grad_cam import GradCAM
        ms = _load_models()
        m = ms[0]
        # For EfficientNet-B0: features[-1] is the final conv layer (Conv-BN-SiLU stack)
        target_layers = [m.features[-1]]
        _GRADCAM = GradCAM(model=m, target_layers=target_layers)
    except Exception as e:
        print(f"[predict] Grad-CAM disabled: {e}")
        _GRADCAM = False
    return _GRADCAM


def _real_gradcam_heatmap(bgr: np.ndarray) -> np.ndarray:
    """Real Grad-CAM overlay on the image, showing where the model looked."""
    cam = _get_gradcam()
    if cam is False:
        return _fallback_heatmap(bgr)
    try:
        from pytorch_grad_cam.utils.image import show_cam_on_image
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_shape = (IMG_SIZE, IMG_SIZE)
        rgb_resized = cv2.resize(rgb, pil_shape).astype(np.float32) / 255.0
        input_tensor = _bgr_to_tensor(cv2.resize(bgr, pil_shape)).unsqueeze(0).to(DEVICE)
        # Target: increase FAKE probability (single-logit BCE model → target 0)
        with torch.enable_grad():
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
        visualization = show_cam_on_image(rgb_resized, grayscale_cam,
                                          use_rgb=True, image_weight=0.55)
        # Convert RGB back to BGR to match the rest of the pipeline
        overlay_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        # Free the extra gradient memory
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        # Resize back to the original crop size
        return cv2.resize(overlay_bgr, (bgr.shape[1], bgr.shape[0]))
    except Exception as e:
        print(f"[predict] Grad-CAM inference failed, fallback: {e}")
        return _fallback_heatmap(bgr)


def _fallback_heatmap(bgr: np.ndarray) -> np.ndarray:
    """Non-Grad-CAM fallback if the library fails (edges + jet colormap)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=5)
    edges = cv2.GaussianBlur(np.abs(edges), (0, 0), sigmaX=8)
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, 0.55, heat, 0.45, 0)


def _fake_heatmap(bgr: np.ndarray) -> np.ndarray:
    """Backwards-compat name — now runs REAL Grad-CAM."""
    return _real_gradcam_heatmap(bgr)


# --------------- public API ---------------

def predict_image(path: str) -> dict:
    """
    QUALITY-WEIGHTED per-face voting:
      - Detect ALL faces WITH bboxes + YOLO confidence
      - Filter out tiny background faces (< 3% of frame area)
      - Score each face at tight + wide crops (worst-of-two per face)
      - Weight each face's contribution by (area × detection_confidence)
      - Final verdict = weighted MEAN of face scores  (not naive MAX)
      - PRIMARY face for display = largest area (subject of the photo)
    A blurry background face no longer flips the verdict of a large clear face.
    """
    t0 = time.time()
    bgr = cv2.imread(path)
    if bgr is None:
        raise ValueError(f"could not read image: {path}")
    H, W = bgr.shape[:2]
    frame_area = H * W

    # Get faces WITH boxes + confidence for quality weighting
    if _HAS_BOX_API:
        detections = detect_with_boxes(bgr, out_size=IMG_SIZE, margin=0.3)  # [(bbox,conf,crop)]
    else:
        legacy = detect_and_crop(bgr, out_size=IMG_SIZE, margin=0.3)
        detections = [((0, 0, W, H), 0.9, c) for c in legacy]

    # Filter: keep faces at least 3% of frame area
    MIN_AREA_FRAC = 0.03
    kept = []
    for bbox, conf, crop in detections:
        _, _, w, h = bbox
        area_frac = (w * h) / max(frame_area, 1)
        if area_frac >= MIN_FACE_AREA_FRAC and conf >= MIN_DETECTION_CONF:
            kept.append((bbox, conf, crop, area_frac))


    # Wide crops for boundary artifact analysis (same faces, wider margin)
    tight_crops = [k[2] for k in kept]
    wide_crops  = []
    if kept:
        for bbox, _, _, _ in kept:
            wide_crops.append(_crop_wide(bgr, bbox, WIDE_MARGIN, IMG_SIZE))
    full_crop = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))

    all_crops = tight_crops + wide_crops + [full_crop]
    scores = _score_batch(all_crops) if all_crops else np.array([0.5])
    n_f = len(tight_crops)

    per_face_scores, weights, blur_scores = [], [], []
    tight_scores, wide_scores, scale_labels = [], [], []
    for i, (bbox, det_conf, crop_i, area_frac) in enumerate(kept):
        s_tight = float(scores[i])
        s_wide  = float(scores[n_f + i]) if i < n_f else s_tight
        # Require agreement between the tight face crop and wider context. The
        # old max() rule turned crop-specific artifacts into extreme false positives.
        face_score = float(np.sqrt(max(s_tight, 0.0) * max(s_wide, 0.0)))
        tight_scores.append(s_tight); wide_scores.append(s_wide)
        if s_tight > FAKE_THRESHOLD and s_wide > FAKE_THRESHOLD:
            scale_labels.append("FAKE")
        elif s_tight < FAKE_THRESHOLD and s_wide < FAKE_THRESHOLD:
            scale_labels.append("REAL")
        else:
            scale_labels.append("INCONCLUSIVE")
        per_face_scores.append(face_score)
        # NEW: blur factor — blurry face gets down-weighted (model can't judge it)
        blur_w = _blur_weight(crop_i)
        blur_scores.append(_blur_score(crop_i))
        # Weight = area × detection confidence × sharpness
        weights.append(area_frac * det_conf * blur_w)
    s_full = float(scores[-1]) if len(scores) > 2 * n_f else 0.5

    # Forensic group rule: one reliable suspicious face must not be hidden by
    # a median/majority of authentic faces.
    reliable_indices = [
        i for i, b in enumerate(blur_scores)
        if b >= MIN_RELIABLE_BLUR
        and kept[i][1] >= MIN_DETECTION_CONF
        and kept[i][3] >= MIN_FACE_AREA_FRAC
    ]
    if per_face_scores:
        candidates = reliable_indices or list(range(len(per_face_scores)))
        weighted_face_score = float(max(per_face_scores[i] for i in candidates))
        # Primary face = LARGEST detected face (main subject)
        primary_idx = int(np.argmax([k[3] for k in kept]))
        primary_crop = kept[primary_idx][2]
    else:
        weighted_face_score = s_full
        primary_idx = 0
        primary_crop = full_crop

    # Face-level evidence drives the verdict; full-frame score is metadata only.
    score = weighted_face_score

    # ═══ Copy-paste face tampering check — ONLY for exactly 2 faces (Mona Lisa) ═══
    dup_sim = 0.0; dup_i = dup_j = -1
    if len(kept) == 2:
        dup_sim, dup_i, dup_j = _detect_duplicate_faces([k[2] for k in kept])


    # Quality gating: do not mutate the probability to manufacture uncertainty.
    avg_brightness = float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean())
    has_reliable_face = bool(reliable_indices) and avg_brightness >= 45
    verdict, conf = _verdict_from_score(score)
    reliable_scale_labels = [scale_labels[i] for i in reliable_indices]
    if not detections:
        verdict, conf = "NO_FACE", 0.0
    elif not has_reliable_face:
        verdict, conf = "INCONCLUSIVE", 0.0
    elif "FAKE" in reliable_scale_labels:
        verdict, conf = _verdict_from_score(score)
    elif reliable_scale_labels and all(v == "REAL" for v in reliable_scale_labels):
        verdict, conf = "REAL", max(conf, 0.5)
    else:
        verdict, conf = "INCONCLUSIVE", 0.0
    # Build all-face list for UI (grid display)
    all_faces_data = []
    for i, (bbox, det_conf, crop, area_frac) in enumerate(kept):
        reliable = i in reliable_indices
        face_verdict = scale_labels[i]
        face_conf = (_verdict_from_score(per_face_scores[i])[1]
                     if face_verdict != "INCONCLUSIVE" else 0.0)
        if not reliable:
            face_verdict, face_conf = "INCONCLUSIVE", 0.0
        all_faces_data.append({
            "index": i,
            "score": float(per_face_scores[i]) if i < len(per_face_scores) else 0.0,
            "weight": float(weights[i]) if i < len(weights) else 0.0,
            "blur": float(blur_scores[i]) if i < len(blur_scores) else 0.0,
            "area_frac": float(area_frac),
            "det_conf": float(det_conf),
            "bbox": bbox,
            "crop_bgr": crop,
            "heatmap_bgr": _fake_heatmap(crop),
            "is_primary": i == primary_idx,
            "reliable": reliable,
            "verdict": face_verdict,
            "confidence": face_conf,
        })

    _load_models()
    return {
        "kind": "image",
        "score": score,
        "verdict": verdict,
        "confidence": conf,
        "face_crop_bgr": primary_crop,
        "heatmap_bgr": _fake_heatmap(primary_crop),
        "all_faces": all_faces_data,               # NEW: full list for UI grid
        "elapsed_ms": (time.time() - t0) * 1000,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "meta": {**_MODEL_META,
                 "n_faces_detected": len(detections),
                 "n_faces_kept": len(kept),
                 "per_face_scores": [round(s, 4) for s in per_face_scores],
                 "tight_face_scores": [round(s, 4) for s in tight_scores],
                 "wide_face_scores": [round(s, 4) for s in wide_scores],
                 "scale_labels": scale_labels,
                 "per_face_weights": [round(w, 4) for w in weights],
                 "weighted_face_score": round(weighted_face_score, 4),
                 "full_frame_score": round(s_full, 4),
                 "primary_face_idx": primary_idx,
                 "duplicate_face_similarity": round(dup_sim, 3),
                 "duplicate_face_pair": [dup_i, dup_j],
                 "per_face_blur_scores": [round(b, 1) for b in blur_scores]},
        "quality": {
            "reliable": has_reliable_face,
            "brightness": round(avg_brightness, 1),
            "reason": None if has_reliable_face else (
                "no face detected" if not detections else
                "detected face is too small, blurry, dark, or low-confidence"
            ),
        },
    }


def _blur_score(bgr: np.ndarray) -> float:
    """
    Laplacian variance = sharpness. Higher = sharper.
    Real photos: usually > 100. Motion-blurred / out-of-focus: < 50.
    Model can't reliably judge blurred faces — we down-weight them.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _blur_weight(bgr: np.ndarray) -> float:
    """
    Convert sharpness → weight multiplier in [0.15, 1.0].
    Very blurry face contributes only 15%; sharp face full weight.
    """
    v = _blur_score(bgr)
    # sigmoid-ish: <30 = 0.15, 60 = 0.5, >120 = 1.0
    if v < 30: return 0.15
    if v < 60: return 0.35
    if v < 100: return 0.7
    return 1.0


def _dhash_face(bgr: np.ndarray, hash_size: int = 16) -> np.ndarray:
    """Difference hash of a face crop, as a bit array."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size))
    return (resized[:, 1:] > resized[:, :-1]).flatten()


def _detect_duplicate_faces(crops: list[np.ndarray]) -> tuple[float, int, int]:
    """
    Check if any two face crops in an image are near-identical (copy-paste
    tampering, e.g. cloned face in the Mona Lisa case).
    Returns (max_similarity, i, j) where i, j are the matching face indices.
    Similarity > 0.9 = almost certainly a copy-paste.
    """
    if len(crops) < 2:
        return 0.0, -1, -1
    hashes = [_dhash_face(c) for c in crops]
    max_sim = 0.0; best = (-1, -1)
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            sim = float((hashes[i] == hashes[j]).mean())
            if sim > max_sim:
                max_sim = sim; best = (i, j)
    return max_sim, best[0], best[1]


def _crop_wide(img: np.ndarray, xywh, margin: float, out_size: int) -> np.ndarray:
    """Same crop math as faces.py but inline (avoids importing internal helper)."""
    x, y, w, h = [int(v) for v in xywh]
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * (1 + 2 * margin)
    x1 = int(max(0, cx - side / 2)); y1 = int(max(0, cy - side / 2))
    x2 = int(min(img.shape[1], cx + side / 2)); y2 = int(min(img.shape[0], cy + side / 2))
    if x2 <= x1 or y2 <= y1:
        return cv2.resize(img, (out_size, out_size))
    return cv2.resize(img[y1:y2, x1:x2], (out_size, out_size))


def predict_video(path: str, fps_sample: float = FPS_SAMPLE) -> dict:
    """
    Multi-face + TRACKED video pipeline (v3):
      - Sample video at fps_sample (default 6 fps ~ every 5th frame @ 30fps)
      - For every sampled frame, detect ALL faces with bboxes (YOLO)
      - IoU-track faces across frames → assign persistent face_id
      - Score every face crop through ensemble
      - Smooth each face_id's score with EMA over a 15-frame window
      - Per-frame score = MAX(smoothed scores across faces in that frame)
      - Video score     = mean(top-25% per-frame scores)  (skews to suspicious)
    Returns extra `per_face_timeline` for UI: score curves per tracked face.
    """
    from collections import defaultdict
    t0 = time.time()

    # -------- Sample + detect (with bboxes if faces.py provides them) --------
    used_fps = fps_sample
    if _HAS_BOX_API:
        pairs = video_face_crops_boxes(path, fps_sample=fps_sample,
                                       out_size=IMG_SIZE, margin=0.3)
        if len({p[0] for p in pairs}) < MIN_FRAMES:
            for try_fps in (8.0, 15.0, 30.0):
                if try_fps <= used_fps: continue
                pairs = video_face_crops_boxes(path, fps_sample=try_fps,
                                               out_size=IMG_SIZE, margin=0.3)
                used_fps = try_fps
                if len({p[0] for p in pairs}) >= MIN_FRAMES: break
    else:
        # Fallback: legacy signature without bboxes → assign dummy bbox 0,0,0,0
        legacy = video_face_crops(path, fps_sample=fps_sample,
                                  out_size=IMG_SIZE, margin=0.3)
        pairs = [(idx, (0, 0, 0, 0), 1.0, crop) for idx, crop in legacy]
    fps_sample = used_fps
    if not pairs:
        raise ValueError(f"no frames extracted from {path}")

    # -------- Score every face crop through ensemble (batched) --------
    crops = [p[3] for p in pairs]
    CHUNK = 32   # smaller chunk for Grad-CAM VRAM budget
    scored = []
    for i in range(0, len(crops), CHUNK):
        scored.append(_score_batch(crops[i:i + CHUNK]))
    raw_scores = np.concatenate(scored) if scored else np.array([], dtype=np.float32)

    # -------- Group by frame, run tracker + EMA in temporal order --------
    # ALSO: filter tiny background faces (< 3% of frame area) — same fix as images
    MIN_AREA_FRAC = 0.03
    frames_dict: dict[int, list[tuple[tuple, float, float, float, np.ndarray]]] = defaultdict(list)
    # tuple: (bbox, det_conf, area_frac, score, crop)
    for (frame_idx, bbox, det_conf, crop), s in zip(pairs, raw_scores):
        _, _, w, h = bbox
        # Frame dims from crop.shape isn't right (crop is 224). Use bbox pixels vs whole video frame guess: fall back to conservative area check via bbox pixels only
        area_pixels = w * h
        # Keep detections above ~10k pixels absolute (roughly 100×100 face) OR pass; we don't have frame dims here
        # We'll do relative filter later inside per-frame loop
        frames_dict[frame_idx].append((bbox, float(det_conf), area_pixels, float(s), crop))
    frame_indices = sorted(frames_dict.keys())

    tracker = IoUTracker(iou_threshold=0.3, max_missed=8)
    ema     = EMASmoother(window=15)
    per_face_timeline: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    reliable_track_scores: dict[int, list[float]] = defaultdict(list)
    per_frame_score: dict[int, float] = {}
    per_frame_best_crop: dict[int, np.ndarray] = {}
    per_frame_best_bbox: dict[int, tuple] = {}
    per_frame_n_faces: dict[int, int] = {}

    for frame_idx in frame_indices:
        faces_in_frame = frames_dict[frame_idx]
        # Frame-relative area filter: keep faces at least 30% of the biggest face in this frame
        max_area = max((f[2] for f in faces_in_frame), default=1)
        kept = [f for f in faces_in_frame if f[2] >= max(0.3 * max_area, 5000)]
        if not kept: kept = faces_in_frame  # fallback

        bboxes = [f[0] for f in kept]
        face_ids = tracker.update(frame_idx, bboxes)

        # Weighted aggregate for this frame
        weighted_num = 0.0; weighted_den = 0.0
        reliable_frame_scores = []
        best_crop = None; best_bbox = None; largest_area = -1
        for (bbox, det_conf, area, raw_s, crop), fid in zip(kept, face_ids):
            smoothed = ema.update(fid, raw_s)
            per_face_timeline[fid].append((frame_idx, float(smoothed), float(raw_s)))
            if det_conf >= MIN_DETECTION_CONF and area >= 5000 and _blur_score(crop) >= MIN_RELIABLE_BLUR:
                reliable_frame_scores.append(float(smoothed))
                reliable_track_scores[fid].append(float(smoothed))
            weight = area * det_conf
            weighted_num += smoothed * weight
            weighted_den += weight
            if area > largest_area:
                largest_area = area; best_crop = crop; best_bbox = bbox
        frame_score = max(reliable_frame_scores) if reliable_frame_scores else 0.5
        per_frame_score[frame_idx]     = float(frame_score)
        per_frame_best_crop[frame_idx] = best_crop if best_crop is not None else crops[0]
        per_frame_best_bbox[frame_idx] = best_bbox if best_bbox is not None else (0, 0, 0, 0)
        per_frame_n_faces[frame_idx]   = len(kept)

    per_frame     = [per_frame_score[i] for i in frame_indices]
    per_frame_arr = np.array(per_frame, dtype=np.float32)

    # Aggregate each person independently, then preserve the most suspicious
    # reliable track. This prevents other people from averaging it away.
    track_summaries = {}
    for fid, values in reliable_track_scores.items():
        if len(values) >= MIN_TRACK_OBSERVATIONS:
            track_summaries[fid] = float(np.percentile(values, 75))
    track_results = []
    for fid, score_i in sorted(track_summaries.items()):
        verdict_i, confidence_i = _verdict_from_score(score_i)
        track_results.append({"face_id": int(fid), "score": score_i,
                              "verdict": verdict_i,
                              "confidence": confidence_i,
                              "observations": len(reliable_track_scores[fid])})
    if track_summaries:
        agg = max(track_summaries.values())
        verdict, conf = _verdict_from_score(agg)
    else:
        agg = 0.5
        has_face_detection = any(float(p[2]) >= MIN_DETECTION_CONF for p in pairs)
        verdict, conf = (("INCONCLUSIVE", 0.0) if has_face_detection
                         else ("NO_FACE", 0.0))

    # -------- Top-K most suspicious frames --------
    order = np.argsort(-per_frame_arr)[:TOP_K_FRAMES]
    top_frames = []
    for i in order:
        fi = int(frame_indices[int(i)])
        crop = per_frame_best_crop[fi]
        top_frames.append({
            "frame_index": fi,
            "score": float(per_frame[int(i)]),
            "face_bgr": crop,
            "heatmap_bgr": _fake_heatmap(crop),  # real Grad-CAM
            "bbox": per_frame_best_bbox[fi],
            "n_faces_in_frame": per_frame_n_faces[fi],
        })

    _load_models()
    return {
        "kind": "video",
        "score": agg,
        "verdict": verdict,
        "confidence": conf,
        "per_frame": per_frame,
        "frame_indices": frame_indices,
        "fps_sampled": fps_sample,
        "top_frames": top_frames,
        "tracks": track_results,
        "elapsed_ms": (time.time() - t0) * 1000,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "meta": {**_MODEL_META,
                 "n_frames": len(per_frame),
                 "n_faces_total": int(len(raw_scores)),
                 "n_tracked_ids": len(per_face_timeline),
                 "track_summaries": {int(k): round(v, 4)
                                     for k, v in track_summaries.items()},
                 "per_face_timeline": {
                     fid: [(fi, round(s, 4), round(r, 4)) for fi, s, r in tl]
                     for fid, tl in per_face_timeline.items()
                 },
                 "avg_faces_per_frame": round(
                     float(np.mean(list(per_frame_n_faces.values()))), 2),
                 "max_face_score": round(float(raw_scores.max()) if len(raw_scores) else 0.0, 4)},
    }


def predict(path: str) -> dict:
    """Route to image or video by extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
        return predict_video(path)
    return predict_image(path)


# --------------- CLI demo ---------------
if __name__ == "__main__":
    import sys, json as _json
    if len(sys.argv) < 2:
        print("usage: python predict.py <file>")
        sys.exit(1)
    r = predict(sys.argv[1])
    # strip numpy arrays for pretty print
    printable = {k: v for k, v in r.items()
                 if not isinstance(v, (np.ndarray,))}
    printable.pop("top_frames", None)
    printable.pop("face_crop_bgr", None)
    printable.pop("heatmap_bgr", None)
    print(_json.dumps(printable, indent=2, default=str))
    print(f"\nverdict: {r['verdict']}  score: {r['score']:.4f}  "
          f"confidence: {r['confidence']:.2f}  elapsed: {r['elapsed_ms']:.0f}ms")
