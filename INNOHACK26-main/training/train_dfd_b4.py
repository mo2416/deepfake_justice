import io
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

PROJECT = Path(__file__).resolve().parents[1]
CANDIDATE = PROJECT / "artifacts" / "dfd_candidate"
FRAMES = PROJECT / "data" / "dfd" / "frames"
SOURCE_WEIGHTS = PROJECT / "models" / "efficientnet_b4_ffpp_source.pth"
OUTPUT = CANDIDATE / "model_dfd_b4_best.pt"
SEED = 20260821


class RandomJPEG:
    def __init__(self, probability=0.55, quality=(28, 88)):
        self.probability, self.quality = probability, quality
    def __call__(self, image):
        if random.random() >= self.probability:
            return image
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=random.randint(*self.quality))
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB")


NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.76, 1.0), ratio=(0.92, 1.08)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.15, 2.4))], p=0.40),
    transforms.ColorJitter(0.16, 0.16, 0.10, 0.025),
    RandomJPEG(), transforms.ToTensor(), NORMALIZE,
])
EVAL_TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), NORMALIZE])


def rows_for(split):
    rows = []
    for label, folder in [(0, "real"), (1, "fake")]:
        rows.extend((path, label) for path in sorted((FRAMES / split / folder).glob("*.jpg")))
    return rows


class Frames(Dataset):
    def __init__(self, rows, transform): self.rows, self.transform = rows, transform
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        path, label = self.rows[index]
        with Image.open(path) as image:
            x = self.transform(image.convert("RGB"))
        return x, torch.tensor(label, dtype=torch.long), str(path)


class DFDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b4(weights=None)
        features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(features, 2))
    def forward(self, x): return self.head(self.backbone(x))


def video_key(path):
    return re.sub(r"_f\d+$", "", Path(path).stem)


@torch.inference_mode()
def evaluate(model, loader, device, threshold=None):
    model.eval(); scores, labels, paths = [], [], []
    for x, y, p in loader:
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(x.to(device))
        scores.extend(torch.softmax(logits.float(), 1)[:, 1].cpu().tolist())
        labels.extend(y.tolist()); paths.extend(p)
    groups = defaultdict(lambda: [[], None])
    for score, label, path in zip(scores, labels, paths):
        groups[video_key(path)][0].append(score); groups[video_key(path)][1] = label
    video_scores = np.array([float(np.median(v[0])) for v in groups.values()])
    video_labels = np.array([v[1] for v in groups.values()])
    if threshold is None:
        best = (0.5, -1.0)
        for value in np.linspace(0.02, 0.98, 193):
            metric = balanced_accuracy_score(video_labels, video_scores >= value)
            if metric > best[1]: best = (float(value), float(metric))
        threshold = best[0]
    predictions = video_scores >= threshold
    tn, fp, fn, tp = confusion_matrix(video_labels, predictions, labels=[0, 1]).ravel()
    return {
        "frame_auc": float(roc_auc_score(labels, scores)),
        "video_auc": float(roc_auc_score(video_labels, video_scores)),
        "video_accuracy": float(accuracy_score(video_labels, predictions)),
        "video_balanced_accuracy": float(balanced_accuracy_score(video_labels, predictions)),
        "threshold": float(threshold), "videos": len(video_labels),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows, val_rows, test_rows = rows_for("train"), rows_for("val"), rows_for("test")
    counts = np.bincount([label for _, label in train_rows], minlength=2)
    sampler = WeightedRandomSampler([1 / counts[y] for _, y in train_rows], num_samples=4000, replacement=True)
    train_loader = DataLoader(Frames(train_rows, TRAIN_TF), batch_size=12, sampler=sampler,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(Frames(val_rows, EVAL_TF), batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(Frames(test_rows, EVAL_TF), batch_size=32, shuffle=False, num_workers=0)
    model = DFDModel()
    source = torch.load(SOURCE_WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict(source["state_dict"], strict=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_auc, best_val = -1.0, None
    started = time.time()
    print(json.dumps({"device": str(device), "train": len(train_rows), "val": len(val_rows),
                      "test_locked": len(test_rows), "train_counts": counts.tolist()}), flush=True)
    for epoch in range(1, 5):
        if epoch == 1:
            for p in model.backbone.parameters(): p.requires_grad = False
            optimizer = torch.optim.AdamW(model.head.parameters(), lr=8e-4, weight_decay=1e-4)
        elif epoch == 2:
            for p in model.backbone.parameters(): p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=2e-4)
        model.train(); loss_sum = 0.0; seen = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            loss_sum += loss.item() * len(y); seen += len(y)
        val = evaluate(model, val_loader, device)
        print(json.dumps({"epoch": epoch, "loss": loss_sum / seen, "val": val}), flush=True)
        if val["video_auc"] > best_auc:
            best_auc, best_val = val["video_auc"], val
            torch.save({"model": model.state_dict(), "threshold": val["threshold"],
                        "val": val, "version": "dfd-b4-actor-disjoint-v1",
                        "source": "abraraltaf92 FFPP B4"}, OUTPUT)
    checkpoint = torch.load(OUTPUT, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True); model.to(device)
    test = evaluate(model, test_loader, device, threshold=checkpoint["threshold"])
    passed = test["video_balanced_accuracy"] >= 0.85 and test["video_auc"] >= 0.85
    report = {"passed_deployment_gate": passed, "gate": "test balanced accuracy and AUC >= 0.85",
              "best_val": best_val, "test": test, "seconds": time.time() - started,
              "candidate": str(OUTPUT), "active_model_changed": False}
    (CANDIDATE / "dfd_training_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
