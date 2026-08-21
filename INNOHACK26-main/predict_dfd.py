"""Reversible DFD video profile. Images continue through the current predictor."""
from __future__ import annotations

import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from faces import detect_with_boxes
from predict import IoUTracker, _blur_score, _fallback_heatmap, predict as predict_current

PROJECT = Path(__file__).resolve().parent
CHECKPOINT = PROJECT / "models" / "model_dfd_b4_best.pt"
THRESHOLD = 0.515
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "DFD EfficientNet-B4 — actor-disjoint video profile"
MODEL_VERSION = "dfd-b4-actor-disjoint-v1"

TFM = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class DFDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b4(weights=None)
        features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(features, 2))
    def forward(self, x):
        return self.head(self.backbone(x))


_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        model = DFDModel()
        model.load_state_dict(checkpoint["model"], strict=True)
        _MODEL = model.to(DEVICE).eval()
    return _MODEL


@torch.inference_mode()
def _score(crops):
    result = []
    model = _model()
    batch_size = 12 if DEVICE.type == "cuda" else 4
    for start in range(0, len(crops), batch_size):
        tensors = []
        for crop in crops[start:start + batch_size]:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(TFM(Image.fromarray(rgb)))
        batch = torch.stack(tensors).to(DEVICE)
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
            logits = model(batch)
        result.extend(torch.softmax(logits.float(), 1)[:, 1].cpu().tolist())
    return np.asarray(result, dtype=np.float32)


def _verdict(score, reliable=True):
    if not reliable or abs(score - THRESHOLD) < 0.055:
        return "INCONCLUSIVE", 0.0
    if score >= THRESHOLD:
        return "FAKE", float(min(1.0, (score - THRESHOLD) / (1.0 - THRESHOLD)))
    return "REAL", float(min(1.0, (THRESHOLD - score) / THRESHOLD))


def _uniform_detections(path, samples=12):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if total <= 0:
        cap.release(); return [], fps
    indices = np.linspace(max(0, total * 0.06), max(0, total * 0.94 - 1), samples).astype(int)
    pairs = []
    for frame_index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        for bbox, confidence, crop in detect_with_boxes(frame, out_size=224, margin=0.18):
            pairs.append((int(frame_index), bbox, float(confidence), crop))
    cap.release()
    return pairs, fps


def predict_video(path):
    started = time.time()
    pairs, source_fps = _uniform_detections(path)
    if not pairs:
        return {"kind": "video", "score": 0.5, "verdict": "NO_FACE", "confidence": 0.0,
                "per_frame": [], "frame_indices": [], "fps_sampled": source_fps or 1.0,
                "top_frames": [], "tracks": [], "elapsed_ms": (time.time()-started)*1000,
                "model_name": MODEL_NAME, "model_version": MODEL_VERSION,
                "meta": {"n_frames": 0, "n_faces_total": 0, "n_tracked_ids": 0,
                         "profile": "dfd_candidate"}}
    scores = _score([p[3] for p in pairs])
    by_frame = defaultdict(list)
    for pair, score in zip(pairs, scores):
        by_frame[pair[0]].append((*pair[1:], float(score)))
    tracker = IoUTracker(iou_threshold=0.25, max_missed=3)
    track_scores, track_crops = defaultdict(list), defaultdict(list)
    frame_scores, frame_crops = {}, {}
    for frame_index in sorted(by_frame):
        items = by_frame[frame_index]
        max_area = max(b[0][2] * b[0][3] for b in items)
        kept = [item for item in items if item[1] >= 0.25 and item[0][2] * item[0][3] >= max(2500, 0.12 * max_area)]
        if not kept: kept = items
        ids = tracker.update(frame_index, [item[0] for item in kept])
        candidates = []
        for face_id, (bbox, det_conf, crop, score) in zip(ids, kept):
            sharpness = _blur_score(crop)
            if sharpness >= 8.0:
                track_scores[face_id].append(score)
                track_crops[face_id].append(crop)
                candidates.append((score, crop, bbox))
        if candidates:
            score, crop, bbox = max(candidates, key=lambda item: item[0])
            frame_scores[frame_index], frame_crops[frame_index] = score, (crop, bbox)
    summaries = {face_id: float(np.median(values)) for face_id, values in track_scores.items() if len(values) >= 2}
    reliable = bool(summaries)
    overall = max(summaries.values()) if reliable else 0.5
    verdict, confidence = _verdict(overall, reliable)
    tracks = []
    for face_id, score in sorted(summaries.items()):
        face_verdict, face_confidence = _verdict(score, True)
        tracks.append({"face_id": int(face_id), "score": score, "verdict": face_verdict,
                       "confidence": face_confidence, "observations": len(track_scores[face_id])})
    ordered_frames = sorted(frame_scores)
    per_frame = [frame_scores[index] for index in ordered_frames]
    top_frames = []
    for index in sorted(ordered_frames, key=lambda i: frame_scores[i], reverse=True)[:5]:
        crop, bbox = frame_crops[index]
        top_frames.append({"frame_index": index, "score": frame_scores[index], "face_bgr": crop,
                           "heatmap_bgr": _fallback_heatmap(crop), "bbox": bbox,
                           "n_faces_in_frame": len(by_frame[index])})
    return {"kind": "video", "score": overall, "verdict": verdict, "confidence": confidence,
            "per_frame": per_frame, "frame_indices": ordered_frames, "fps_sampled": source_fps or 1.0,
            "top_frames": top_frames, "tracks": tracks,
            "elapsed_ms": (time.time() - started) * 1000,
            "model_name": MODEL_NAME, "model_version": MODEL_VERSION,
            "meta": {"threshold": THRESHOLD, "n_frames": len(ordered_frames),
                     "n_faces_total": len(pairs), "n_tracked_ids": len(track_scores),
                     "track_summaries": summaries, "profile": "dfd_candidate",
                     "locked_test_auc": 0.9778, "locked_test_balanced_accuracy": 0.8985}}


def predict(path):
    if os.path.splitext(path)[1].lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        return predict_video(path)
    return predict_current(path)
