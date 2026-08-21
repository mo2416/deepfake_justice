"""
Real forensics: perceptual hash, EXIF extraction, basic ELA.

Public API used by app.py:
    compute_phash(file_path) -> str
    extract_exif(file_path)  -> dict
    ela_image(file_path)     -> np.ndarray | None
"""
from __future__ import annotations
import os, subprocess, json
from typing import Optional
from PIL import Image, ImageChops, ImageEnhance, ExifTags


VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def _is_video(path: str) -> bool:
    return path.lower().endswith(VIDEO_EXTS)


def compute_phash(file_path: str) -> str:
    """
    Perceptual hash. For images, uses a fast difference-hash (dHash) implementation
    that doesn't require the `imagehash` package. For videos, hashes the middle frame.
    Returns a 16-char hex string.
    """
    try:
        if _is_video(file_path):
            img = _video_middle_frame(file_path)
            if img is None:
                return "unavailable"
        else:
            img = Image.open(file_path).convert("L")
        return _dhash(img)
    except Exception:
        return "unavailable"


def _dhash(img: Image.Image, hash_size: int = 8) -> str:
    """Difference hash — 64-bit → 16 hex chars. No numpy needed."""
    img = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = list(img.getdata())
    bits = []
    for row in range(hash_size):
        for col in range(hash_size):
            left  = pixels[row * (hash_size + 1) + col]
            right = pixels[row * (hash_size + 1) + col + 1]
            bits.append(1 if left > right else 0)
    val = 0
    for b in bits:
        val = (val << 1) | b
    return f"{val:016x}"


def _video_middle_frame(path: str) -> Optional[Image.Image]:
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 2))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        return Image.fromarray(frame[..., ::-1])
    except Exception:
        return None


def extract_exif(file_path: str) -> dict:
    """
    Return EXIF (image) or ffprobe-derived container/codec info (video).
    """
    if _is_video(file_path):
        return _ffprobe_meta(file_path)
    try:
        img = Image.open(file_path)
        raw = img.getexif()
        out = {}
        for tag_id, val in raw.items():
            tag = ExifTags.TAGS.get(tag_id, str(tag_id))
            # keep serialisable values only
            if isinstance(val, (str, int, float)):
                out[tag] = val
            elif isinstance(val, bytes):
                try:
                    out[tag] = val.decode("utf-8", errors="ignore")[:200]
                except Exception:
                    out[tag] = f"<{len(val)} bytes>"
            else:
                out[tag] = str(val)[:200]
        out["_image_size"]  = f"{img.width}x{img.height}"
        out["_image_mode"]  = img.mode
        out["_image_format"] = img.format or "unknown"
        return out
    except Exception as e:
        return {"error": str(e)}


def _ffprobe_meta(path: str) -> dict:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", path],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode != 0:
            return {"note": "ffprobe not available or failed"}
        j = json.loads(r.stdout)
        fmt = j.get("format", {}); streams = j.get("streams", [])
        v = next((s for s in streams if s.get("codec_type") == "video"), {})
        a = next((s for s in streams if s.get("codec_type") == "audio"), {})
        format_tags = fmt.get("tags", {}) or {}
        video_tags = v.get("tags", {}) or {}
        return {
            "format_name":    fmt.get("format_name"),
            "duration_sec":   float(fmt.get("duration", 0) or 0),
            "size_bytes":     int(fmt.get("size", 0) or 0),
            "bit_rate":       int(fmt.get("bit_rate", 0) or 0),
            "video_codec":    v.get("codec_name"),
            "video_size":     f"{v.get('width', '?')}x{v.get('height', '?')}",
            "video_fps":      v.get("r_frame_rate"),
            "video_pix_fmt":  v.get("pix_fmt"),
            "audio_codec":    a.get("codec_name") if a else None,
            "audio_channels": a.get("channels") if a else None,
            "encoder":         format_tags.get("encoder") or video_tags.get("encoder"),
            "creation_time":   format_tags.get("creation_time") or video_tags.get("creation_time"),
        }
    except FileNotFoundError:
        return {"note": "ffprobe (from ffmpeg) not installed"}
    except Exception as e:
        return {"error": str(e)}


def ela_image(file_path: str):
    """
    Error Level Analysis for images. Re-saves at JPEG quality 90 and returns
    an amplified difference — genuine areas fade to near-black, edited/pasted
    regions show up brighter.
    Returns a numpy array (RGB uint8) or None on failure.
    """
    if _is_video(file_path):
        return None
    try:
        import numpy as np
        orig = Image.open(file_path).convert("RGB")
        buf_path = file_path + ".ela_tmp.jpg"
        orig.save(buf_path, "JPEG", quality=90)
        try:
            resaved = Image.open(buf_path).convert("RGB")
            diff = ImageChops.difference(orig, resaved)
            extrema = diff.getextrema()
            max_diff = max((e[1] for e in extrema), default=1) or 1
            scale = 255.0 / max_diff
            diff = ImageEnhance.Brightness(diff).enhance(scale)
            return np.asarray(diff, dtype=np.uint8)
        finally:
            try: os.remove(buf_path)
            except Exception: pass
    except Exception:
        return None
