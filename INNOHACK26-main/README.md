# DeepGuard — Deepfake Detection and Digital Evidence Authentication

## [Open the public DeepGuard website](https://raw.githack.com/RAGU-NAN-DHAN/INNOHACK26/main/site/index.html)

DeepGuard is a notebook-first hackathon project for screening images and videos for facial manipulation. It combines multi-face detection, video tracking, model uncertainty, evidence hashing, metadata checks, suspicious-frame timestamps, and a black/green Streamlit interface.

> This is an automated screening and research aid, not a substitute for qualified forensic examination or a standalone basis for legal conclusions.

## Verified DFD video result

The DFD EfficientNet-B4 profile was evaluated once on an actor-disjoint locked test split:

| Metric | Result |
|---|---:|
| Test videos | 130 |
| Accuracy | **89.2%** |
| Balanced accuracy | **89.9%** |
| ROC AUC | **97.8%** |
| False positives | 2 / 59 real videos |
| False negatives | 12 / 71 fake videos |

These numbers describe the included DFD-style benchmark—not universal accuracy on every manipulation, camera, demographic, or compression pipeline. See [the model card](docs/MODEL_CARD.md) and [raw evaluation report](reports/dfd_training_report.json).

## Notebook-first walkthrough

Run the notebooks in order:

1. [`01_datasets_and_training.ipynb`](notebooks/01_datasets_and_training.ipynb)
2. [`02_inference_pipeline.ipynb`](notebooks/02_inference_pipeline.ipynb)
3. [`03_streamlit_app.ipynb`](notebooks/03_streamlit_app.ipynb)
4. [`04_dfd_video_training.ipynb`](notebooks/04_dfd_video_training.ipynb)
5. [`05_website_and_rollback.ipynb`](notebooks/05_website_and_rollback.ipynb)

The Python modules remain available for repeatable CLI and website execution.

## Quick start

### 1. Clone with Git LFS

```powershell
git lfs install
git clone https://github.com/RAGU-NAN-DHAN/INNOHACK26.git
cd INNOHACK26
git lfs pull
```

### 2. Create an environment

Python 3.11 and an NVIDIA GPU are recommended.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Start the simple tester

```powershell
streamlit run tester_app.py --server.port 4747
```

Open <http://localhost:4747>.

For the authenticated forensic interface:

```powershell
streamlit run app.py --server.port 8502
```

## Reversible video profiles

The website includes a `VIDEO MODEL PROFILE` control:

- **DFD High-Accuracy (Recommended):** actor-disjoint EfficientNet-B4 video model.
- **Current / Rollback:** previous EfficientNet-B0 ensemble.

Switching profiles does not delete or overwrite either checkpoint. Image uploads continue through the existing image pipeline in both modes. See [rollback instructions](docs/ROLLBACK.md).

## Repository layout

```text
INNOHACK26/
├── app.py                         # Full authenticated forensic application
├── tester_app.py                  # Simple black/green tester (port 4747)
├── predict.py                     # Previous/current image and video pipeline
├── predict_dfd.py                 # DFD B4 video profile; delegates images to predict.py
├── faces.py                       # YOLO face detection + YuNet fallback
├── auth.py                        # Local analyst accounts/history (SQLite)
├── forensics.py                   # EXIF, hashes, error-level and video metadata checks
├── report.py                      # Exportable forensic PDF report
├── notebooks/                     # Preferred cell-by-cell walkthroughs
├── training/                      # Reproducible preprocessing/training scripts
├── models/                        # Git-LFS checkpoints
├── reports/                       # Recorded evaluation outputs
└── docs/                          # Dataset, model and rollback documentation
```

## DFD dataset

The raw DFD archives and extracted frames are not committed: they are roughly 24 GB, have separate dataset terms, and do not belong in normal Git history. Follow [docs/DATASETS.md](docs/DATASETS.md) to reproduce the actor-disjoint split and face crops.

## Important limitations

- High benchmark performance does not imply universal forensic reliability.
- Very small, dark, occluded, non-frontal, or severely blurred faces may be inconclusive.
- Images and videos use different optimized profiles.
- A manipulated person in a group is handled independently, but missed face detections can still cause misses.
- Preserve the original evidence file and SHA-256 hash; never rely on a screenshot of the result alone.

