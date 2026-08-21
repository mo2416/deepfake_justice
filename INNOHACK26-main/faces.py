"""
Face detection using YOLOv8n-face (primary) with YuNet fallback.

Public API (contract unchanged):
    detect_and_crop(img_bgr, out_size=256, margin=0.28) -> list[np.ndarray]
    sample_video_frames(video_path, fps_sample=6.0) -> list[tuple[int, np.ndarray]]
    video_face_crops(video_path, ...) -> list[tuple[int, np.ndarray]]
    draw_face_box(img_bgr, box, color, thickness) -> np.ndarray

NEW additions:
    detect_with_boxes(img_bgr, out_size=256, margin=0.28)
        -> list[tuple[bbox_xywh, conf, crop_bgr]]
    video_face_crops_boxes(video_path, ...)
        -> list[tuple[frame_idx, bbox_xywh, conf, crop_bgr]]
"""
from __future__ import annotations
import os
import sys
from typing import Optional
import cv2
import numpy as np

# -------- YOLO primary --------
_YOLO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "models", "yolo", "yolov8n-face.pt")
_YOLO = None            # ultralytics.YOLO
_YOLO_LOAD_ERR = None

def _load_yolo():
    global _YOLO, _YOLO_LOAD_ERR
    if _YOLO is not None or _YOLO_LOAD_ERR is not None:
        return _YOLO
    try:
        from ultralytics import YOLO
        _YOLO = YOLO(_YOLO_PATH)
        # warmup on dummy
        _YOLO.predict(np.zeros((320, 320, 3), dtype=np.uint8), verbose=False, device="cpu")
    except Exception as e:
        _YOLO_LOAD_ERR = str(e)
        _YOLO = None
    return _YOLO


# -------- YuNet fallback --------
_YUNET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "face_detection_yunet_2023mar.onnx")
_YUNET_CACHE: dict[tuple[int, int], cv2.FaceDetectorYN] = {}

def _get_yunet(w: int, h: int):
    key = (w, h)
    if key not in _YUNET_CACHE and os.path.exists(_YUNET_PATH):
        _YUNET_CACHE[key] = cv2.FaceDetectorYN.create(
            model=_YUNET_PATH, config="", input_size=(w, h),
            score_threshold=0.5, nms_threshold=0.3, top_k=5000,
        )
    return _YUNET_CACHE.get(key)


# -------- shared crop helper --------
def _crop_with_margin(img: np.ndarray, xywh, margin: float, out_size: int) -> np.ndarray:
    x, y, w, h = [int(v) for v in xywh]
    cx = x + w / 2.0; cy = y + h / 2.0
    side = max(w, h) * (1.0 + 2.0 * margin)
    x1 = int(round(cx - side / 2.0)); y1 = int(round(cy - side / 2.0))
    x2 = int(round(cx + side / 2.0)); y2 = int(round(cy + side / 2.0))
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img.shape[1], x2); y2 = min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return cv2.resize(img, (out_size, out_size))
    return cv2.resize(img[y1:y2, x1:x2], (out_size, out_size))


# -------- Detection: unified interface --------
def _detect_yolo(img_bgr: np.ndarray, conf_threshold: float = 0.25):
    """Return list of (x, y, w, h, conf). Empty if YOLO fails or no faces."""
    yolo = _load_yolo()
    if yolo is None:
        return []
    try:
        res = yolo.predict(img_bgr, conf=conf_threshold, verbose=False, device="cpu")
        out = []
        for r in res:
            boxes = r.boxes
            if boxes is None or boxes.xyxy is None: continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c in zip(xyxy, confs):
                out.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1), float(c)))
        return out
    except Exception:
        return []


def _detect_yunet(img_bgr: np.ndarray):
    """Return list of (x, y, w, h, conf)."""
    h_img, w_img = img_bgr.shape[:2]
    det = _get_yunet(w_img, h_img)
    if det is None: return []
    _, faces = det.detect(img_bgr)
    if faces is None: return []
    out = []
    for f in faces:
        x, y, w, h = [int(v) for v in f[0:4]]
        conf = float(f[-1]) if len(f) > 4 else 0.9
        out.append((x, y, w, h, conf))
    return out


def _detect_any(img_bgr: np.ndarray):
    """YOLO first; YuNet fallback."""
    hits = _detect_yolo(img_bgr)
    if hits: return hits
    return _detect_yunet(img_bgr)


# ========================== PUBLIC API =====================================

def detect_and_crop(img_bgr: np.ndarray, out_size: int = 256,
                    margin: float = 0.28) -> list[np.ndarray]:
    """Return 256x256 BGR uint8 face crops. If none, returns [whole-image resized]."""
    hits = _detect_any(img_bgr)
    if not hits:
        return [cv2.resize(img_bgr, (out_size, out_size))]
    return [_crop_with_margin(img_bgr, h[:4], margin, out_size) for h in hits]


def detect_with_boxes(img_bgr: np.ndarray, out_size: int = 256,
                      margin: float = 0.28
                      ) -> list[tuple[tuple[int, int, int, int], float, np.ndarray]]:
    """Return list of ((x,y,w,h), confidence, crop_256x256). Empty if no faces found."""
    hits = _detect_any(img_bgr)
    out = []
    for x, y, w, h, conf in hits:
        crop = _crop_with_margin(img_bgr, (x, y, w, h), margin, out_size)
        out.append(((x, y, w, h), conf, crop))
    return out


def sample_video_frames(video_path: str,
                        fps_sample: float = 6.0
                        ) -> list[tuple[int, np.ndarray]]:
    """Sample frames at target fps (default 6). Handles very short videos."""
    cap = cv2.VideoCapture(video_path)
    frames: list[tuple[int, np.ndarray]] = []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps): fps = 30.0
        step = max(1, int(round(fps / fps_sample)))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if 0 < total < step:
            mid = total // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ok, frame = cap.read()
            if ok and frame is not None:
                return [(mid, frame)]
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            if idx % step == 0 and frame is not None:
                frames.append((idx, frame))
            idx += 1
    finally:
        cap.release()
    return frames


def video_face_crops(video_path: str, fps_sample: float = 6.0,
                     out_size: int = 256, margin: float = 0.28
                     ) -> list[tuple[int, np.ndarray]]:
    """Every face in every sampled frame. Returns (frame_idx, crop)."""
    pairs = sample_video_frames(video_path, fps_sample)
    out = []
    for idx, frame in pairs:
        hits = _detect_any(frame)
        if not hits:
            out.append((idx, cv2.resize(frame, (out_size, out_size))))
            continue
        for x, y, w, h, _ in hits:
            out.append((idx, _crop_with_margin(frame, (x, y, w, h), margin, out_size)))
    return out


def video_face_crops_boxes(video_path: str, fps_sample: float = 6.0,
                           out_size: int = 256, margin: float = 0.28
                           ) -> list[tuple[int, tuple[int, int, int, int], float, np.ndarray]]:
    """Every face in every sampled frame WITH bboxes. For tracker + UI overlay."""
    pairs = sample_video_frames(video_path, fps_sample)
    out = []
    for idx, frame in pairs:
        hits = _detect_any(frame)
        if not hits:
            h_img, w_img = frame.shape[:2]
            out.append((idx, (0, 0, w_img, h_img), 0.0,
                        cv2.resize(frame, (out_size, out_size))))
            continue
        for x, y, w, h, conf in hits:
            crop = _crop_with_margin(frame, (x, y, w, h), margin, out_size)
            out.append((idx, (x, y, w, h), conf, crop))
    return out


def draw_face_box(img_bgr: np.ndarray, box,
                  color=(0, 0, 255), thickness: int = 3) -> np.ndarray:
    x, y, w, h = [int(v) for v in box[:4]]
    out = img_bgr.copy()
    cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out


def draw_face_boxes(img_bgr: np.ndarray,
                    items: list[tuple[tuple[int, int, int, int], float, str]],
                    ) -> np.ndarray:
    """Draw multiple face boxes with labels. items: (bbox, score, label)."""
    out = img_bgr.copy()
    for bbox, score, label in items:
        x, y, w, h = [int(v) for v in bbox[:4]]
        color = (0, 0, 255) if score >= 0.5 else (0, 255, 0)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 3)
        text = f"{label} {score*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x, y - th - 8), (x + tw + 6, y), color, -1)
        cv2.putText(out, text, (x + 3, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


# ========================== CLI demo =======================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <image_or_video_path>"); sys.exit(1)
    path = sys.argv[1]
    ext = os.path.splitext(path)[1].lower()
    if ext in [".mp4", ".mov", ".avi", ".mkv"]:
        crops = video_face_crops_boxes(path)
        print(f"Sampled {len(set(c[0] for c in crops))} frames, "
              f"total {len(crops)} face crops")
        if crops:
            cv2.imwrite("crop_first.jpg", crops[0][3])
            cv2.imwrite("crop_last.jpg",  crops[-1][3])
    else:
        img = cv2.imread(path)
        if img is None:
            print(f"could not read {path}"); sys.exit(1)
        results = detect_with_boxes(img)
        print(f"Found {len(results)} face(s) using "
              f"{'YOLO' if _load_yolo() else 'YuNet'}")
        for i, (bbox, conf, crop) in enumerate(results):
            print(f"  face {i}: bbox={bbox}  conf={conf:.3f}")
            cv2.imwrite(f"crop_{i}.jpg", crop)
