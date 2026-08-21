"""Generate the notebook-first walkthroughs shipped with the repository."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"
OUT.mkdir(exist_ok=True)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(True)}


def write(name, cells):
    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    (OUT / name).write_text(json.dumps(notebook, indent=1), encoding="utf-8")


write("04_dfd_video_training.ipynb", [
    md("""# DFD video training — actor-disjoint and rollback-safe

This notebook is the preferred walkthrough for the high-accuracy video candidate. It never overwrites the active model. Actors 01–19 are training, 20–23 validation, and 24–28 locked test. Manipulated pairs crossing actor pools are discarded to prevent identity leakage.

Locked test result: **89.2% accuracy**, **89.9% balanced accuracy**, **97.8% AUC** on 130 videos."""),
    code("""from pathlib import Path
import json

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
RAW = ROOT / "data" / "dfd" / "raw"
FRAMES = ROOT / "data" / "dfd" / "frames"
ARTIFACTS = ROOT / "artifacts" / "dfd_candidate"
print("Repository:", ROOT)
print("Raw DFD directory:", RAW)"""),
    md("""## Dataset layout

Extract the original videos directly into `data/dfd/raw/` and the manipulated archive so it creates `data/dfd/raw/DFD_manipulated_sequences/`. The archives and extracted media are intentionally excluded from Git."""),
    code("""POOLS = {
    "train": set(range(1, 20)),
    "val": set(range(20, 24)),
    "test": set(range(24, 29)),
}

def split_for(actor_ids):
    for split, pool in POOLS.items():
        if all(actor in pool for actor in actor_ids):
            return split
    return None

assert split_for([1, 2]) == "train"
assert split_for([24, 28]) == "test"
assert split_for([19, 20]) is None  # cross-pool pair is rejected"""),
    md("## Extract four YOLO face crops per included video"),
    code("""# Runs the complete reproducible preprocessing module.
%run ../training/prepare_dfd.py"""),
    md("""## Training recipe

EfficientNet-B4 starts from a FaceForensics++ checkpoint. Epoch 1 calibrates the head; epochs 2–4 fine-tune end-to-end. Sampling is class-balanced. Blur, JPEG recompression, crops, color variation and horizontal flips are applied only to training."""),
    code("""import torch
from torch import nn
from torchvision import models

class DFDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b4(weights=None)
        features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(features, 2))

    def forward(self, x):
        return self.head(self.backbone(x))

DFDModel()"""),
    code("""# Trains, calibrates threshold on validation, then evaluates the locked test once.
%run ../training/train_dfd_b4.py"""),
    md("## Inspect the saved deployment-gate report"),
    code("""report_path = ROOT / "reports" / "dfd_training_report.json"
report = json.loads(report_path.read_text())
report["passed_deployment_gate"], report["test"]"""),
])

write("05_website_and_rollback.ipynb", [
    md("""# Website, inference profiles and instant rollback

The website exposes two video profiles:

- **DFD High-Accuracy (Recommended)** — EfficientNet-B4 candidate that passed the locked test.
- **Current / Rollback** — the previous B0 ensemble.

Images continue through the existing image pipeline in both profiles."""),
    code("""from pathlib import Path
import sys

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from predict import predict as predict_current
from predict_dfd import predict as predict_dfd
print("Profiles imported successfully")"""),
    md("## Run either profile directly"),
    code("""# Replace with a local image/video path.
media_path = None

if media_path:
    dfd_result = predict_dfd(media_path)
    current_result = predict_current(media_path)
    print("DFD:", dfd_result["verdict"], dfd_result["score"])
    print("Current:", current_result["verdict"], current_result["score"])"""),
    md("## Verify model artifacts before launching"),
    code("""required = [
    ROOT / "models" / "model_v2_best.pt",
    ROOT / "models" / "model_v3_best.pt",
    ROOT / "models" / "model_dfd_b4_best.pt",
    ROOT / "models" / "yolo" / "yolov8n-face.pt",
]
for path in required:
    print(path.relative_to(ROOT), "OK" if path.exists() else "MISSING")"""),
    md("## Launch the simple black/green tester on port 4747"),
    code("""# Run this cell and open http://localhost:4747
!streamlit run ../tester_app.py --server.port 4747 --server.headless true"""),
    md("""## Full authenticated forensic application

The full app adds analyst accounts, history, hashes, metadata, timelines and report generation. It uses the same reversible profile selector."""),
    code("""# Alternative launch command:
# !streamlit run ../app.py --server.port 8502 --server.headless true"""),
])

print("Generated notebook-first walkthroughs in", OUT)
