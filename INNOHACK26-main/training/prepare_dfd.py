import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
CANDIDATE = PROJECT / "artifacts" / "dfd_candidate"
RAW = PROJECT / "data" / "dfd" / "raw"
FAKE_ROOT = RAW / "DFD_manipulated_sequences"
FRAMES = PROJECT / "data" / "dfd" / "frames"
sys.path.insert(0, str(PROJECT))
from faces import detect_with_boxes  # noqa: E402

POOLS = {
    "train": set(range(1, 20)),
    "val": set(range(20, 24)),
    "test": set(range(24, 29)),
}
PAIR = re.compile(r"^(\d+)_(\d+)__")
ORIGINAL = re.compile(r"^(\d+)__")


def split_for(ids):
    for split, pool in POOLS.items():
        if all(actor in pool for actor in ids):
            return split
    return None


def inventory():
    rows = []
    for path in sorted(RAW.glob("*.mp4")):
        match = ORIGINAL.match(path.stem)
        if match:
            split = split_for([int(match.group(1))])
            rows.append({"path": str(path), "label": 0, "split": split, "actors": [int(match.group(1))]})
    for path in sorted(FAKE_ROOT.glob("*.mp4")):
        match = PAIR.match(path.stem)
        if match:
            actors = [int(match.group(1)), int(match.group(2))]
            split = split_for(actors)
            if split is not None:
                rows.append({"path": str(path), "label": 1, "split": split, "actors": actors})
    return rows


def extract_video(row, frames_per_video=4):
    path = Path(row["path"])
    label_name = "fake" if row["label"] else "real"
    out_dir = FRAMES / row["split"] / label_name
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = [out_dir / f"{path.stem}_f{i:02d}.jpg" for i in range(frames_per_video)]
    if all(p.exists() for p in expected):
        return len(expected)
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release(); return 0
    indices = np.linspace(max(0, total * 0.08), max(0, total * 0.92 - 1), frames_per_video).astype(int)
    written = 0
    for index, output in zip(indices, expected):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        detected = detect_with_boxes(frame, out_size=256, margin=0.18)
        if not detected:
            continue
        _box, _confidence, crop = max(detected, key=lambda item: item[0][2] * item[0][3])
        if cv2.imwrite(str(output), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            written += 1
    cap.release()
    return written


def main():
    rows = inventory()
    (CANDIDATE / "split_manifest.json").write_text(json.dumps(rows, indent=2))
    started = time.time(); written = 0
    for number, row in enumerate(rows, 1):
        written += extract_video(row)
        if number % 50 == 0:
            print(f"progress={number}/{len(rows)} frames_written_or_present={written} elapsed={time.time()-started:.1f}s", flush=True)
    counts = {}
    for split in POOLS:
        counts[split] = {}
        for label in ("real", "fake"):
            counts[split][label] = len(list((FRAMES / split / label).glob("*.jpg")))
    report = {"videos": len(rows), "frames": counts, "seconds": time.time() - started,
              "policy": "actor-disjoint; cross-pool manipulated pairs excluded"}
    (CANDIDATE / "preprocess_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
