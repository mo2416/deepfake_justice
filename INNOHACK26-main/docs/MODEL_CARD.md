# DeepGuard model card

## DFD video profile

- Architecture: EfficientNet-B4, two-class head.
- Initialization: FaceForensics++ checkpoint.
- Training: balanced video-frame sampling with random crops, flips, color variation, Gaussian blur and JPEG recompression.
- Validation threshold: 0.515.
- Deployment gate: locked-test ROC AUC and balanced accuracy must both be at least 0.85.

### Locked actor-disjoint test

| Metric | Value |
|---|---:|
| Videos | 130 |
| Accuracy | 0.8923 |
| Balanced accuracy | 0.8985 |
| ROC AUC | 0.9778 |
| TN / FP / FN / TP | 57 / 2 / 12 / 59 |

## Current / rollback profile

The earlier profile ensembles two EfficientNet-B0 checkpoints and uses multi-scale face crops, per-face tracking, blur gating and explicit inconclusive results. Its very high in-domain image score did not transfer reliably to the external sanity samples, so it is retained as a rollback and comparison profile—not advertised as universally accurate.

## Intended use

Research, demonstrations and automated evidence triage. Results should guide further examination, not replace it.

## Out-of-scope and failure modes

- non-face manipulation;
- faces too small for reliable detection;
- extreme blur, darkness, occlusion or profile pose;
- manipulation methods outside training distribution;
- claims of source identity or legal authenticity;
- calibrated confidence across every camera and compression pipeline.

## Reproducibility

The split and preprocessing code is in `training/prepare_dfd.py`. Training and the locked evaluation are in `training/train_dfd_b4.py`. Recorded JSON output is under `reports/`.

