from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from auth import DB_PATH
from forensics import compute_phash, extract_exif


EDITOR_TERMS = (
    "photoshop", "lightroom", "premiere", "after effects", "davinci",
    "resolve", "final cut", "capcut", "filmora", "gimp", "canva",
    "ffmpeg", "lavf", "handbrake", "media encoder",
)


def inspect_evidence(file_path: str, payload: bytes, result: dict, username: str) -> dict:
    metadata = extract_exif(file_path) or {}
    phash = compute_phash(file_path)
    analyzed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sha256 = hashlib.sha256(payload).hexdigest()
    evidence_id = f"DG-{analyzed_at[:10].replace('-', '')}-{sha256[:12].upper()}"

    searchable = " ".join(f"{key}={value}" for key, value in metadata.items()).lower()
    editor_hits = sorted({term for term in EDITOR_TERMS if term in searchable})
    consistency = _consistency_checks(metadata, result.get("kind", "image"))
    has_c2pa_marker = b"c2pa" in payload.lower() or b"jumb" in payload.lower()

    if editor_hits:
        modification_status = "EDITING / RE-ENCODING INDICATOR"
        modification_detail = "Software metadata mentions: " + ", ".join(editor_hits)
    else:
        modification_status = "NO EXPLICIT EDITOR TRACE"
        modification_detail = "No known editing-software name was found in available metadata. This does not prove originality."

    return {
        "evidence_id": evidence_id,
        "sha256": sha256,
        "perceptual_hash": phash,
        "analyzed_at_utc": analyzed_at,
        "authenticated_user": username,
        "metadata": metadata,
        "metadata_available": bool(metadata and not set(metadata).issubset({"error", "note"})),
        "modification_status": modification_status,
        "modification_detail": modification_detail,
        "consistency_checks": consistency,
        "c2pa_status": (
            "CREDENTIAL MARKER PRESENT - SIGNATURE NOT VALIDATED"
            if has_c2pa_marker else
            "NO C2PA CREDENTIAL MARKER DETECTED"
        ),
        "scope_note": (
            "DeepGuard records integrity from upload onward. Without a trusted acquisition-time hash "
            "or a validated signed credential, it cannot prove the file was unchanged before upload."
        ),
    }


def _consistency_checks(metadata: dict, kind: str) -> list[dict]:
    checks: list[dict] = []
    if kind == "video":
        codec = metadata.get("video_codec")
        size = metadata.get("video_size")
        fps = metadata.get("video_fps")
        duration = metadata.get("duration_sec")
        checks.extend([
            {"label": "Video codec", "value": codec or "unavailable", "ok": bool(codec)},
            {"label": "Frame dimensions", "value": size or "unavailable", "ok": bool(size and "?" not in str(size))},
            {"label": "Frame rate", "value": fps or "unavailable", "ok": bool(fps)},
            {"label": "Duration", "value": f"{duration:.2f} s" if isinstance(duration, (int, float)) else "unavailable", "ok": bool(duration)},
        ])
    else:
        checks.extend([
            {"label": "Image format", "value": metadata.get("_image_format", "unavailable"), "ok": bool(metadata.get("_image_format"))},
            {"label": "Image dimensions", "value": metadata.get("_image_size", "unavailable"), "ok": bool(metadata.get("_image_size"))},
            {"label": "Colour mode", "value": metadata.get("_image_mode", "unavailable"), "ok": bool(metadata.get("_image_mode"))},
        ])
    return checks


def matching_hash_records(sha256: str) -> int:
    try:
        with sqlite3.connect(DB_PATH) as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM analyses WHERE sha256=?", (sha256,)
            ).fetchone()
        return int(row[0] if row else 0)
    except Exception:
        return 0


def suspicious_moments(result: dict) -> list[dict]:
    fps = float(result.get("fps_sampled") or 1.0)
    moments = []
    for item in result.get("top_frames") or []:
        frame_index = int(item.get("frame_index", 0))
        seconds = frame_index / max(fps, 0.001)
        minutes = int(seconds // 60)
        remainder = seconds - minutes * 60
        moments.append({
            "frame_index": frame_index,
            "timestamp": f"{minutes:02d}:{remainder:05.2f}",
            "score": float(item.get("score", 0.0)),
            "face_bgr": item.get("face_bgr"),
            "heatmap_bgr": item.get("heatmap_bgr"),
        })
    return moments
