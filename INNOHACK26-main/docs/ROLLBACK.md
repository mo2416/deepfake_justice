# Model rollback

No checkpoint is overwritten when profiles are changed.

## Website rollback

In either Streamlit application, set `VIDEO MODEL PROFILE` to:

- `DFD High-Accuracy (Recommended)` for `models/model_dfd_b4_best.pt`; or
- `Current / Rollback` for `models/model_v3_best.pt` + `models/model_v2_best.pt`.

The switch takes effect on the next scan. Images always route through the current image pipeline.

## Code-level rollback

```python
from predict import predict as predict_current
from predict_dfd import predict as predict_dfd

result = predict_current("video.mp4")  # previous profile
result = predict_dfd("video.mp4")      # passing DFD profile
```

## Rejection rule for future candidates

Keep a new checkpoint isolated until it passes both ROC AUC and balanced accuracy gates on a locked identity/video-disjoint test. Do not select a threshold on the locked test. If the gate fails, do not change the active website default.

