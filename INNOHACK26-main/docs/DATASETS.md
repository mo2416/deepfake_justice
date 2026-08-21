# Dataset setup

## DFD videos

The repository does not redistribute the raw DFD archives or extracted media. Obtain the dataset under its original terms. Place the extracted files as follows:

```text
data/dfd/raw/
├── 01__exit_phone_room.mp4
├── ... original videos ...
└── DFD_manipulated_sequences/
    ├── 01_02__exit_phone_room__....mp4
    └── ... manipulated videos ...
```

If you already have the two ZIP files, from the repository root:

```powershell
New-Item -ItemType Directory -Force data\dfd\raw
tar -xf "DFD_original sequences.zip" -C data\dfd\raw
tar -xf "DFD_manipulated_sequences.zip" -C data\dfd\raw
```

Then extract face crops:

```powershell
python training\prepare_dfd.py
```

The split policy is fixed:

- actors 01–19: training;
- actors 20–23: validation and threshold selection;
- actors 24–28: locked test;
- manipulated pairs whose two actors cross pools: excluded.

This prevents identities from crossing train/validation/test. Four uniformly spaced YOLO face crops are extracted per included video. Expected counts from the supplied archives:

| Split | Real frames | Fake frames |
|---|---:|---:|
| Train | 988 | 6,536 |
| Validation | 228 | 188 |
| Test | 236 | 284 |

## Other documented datasets

The earlier notebooks discuss CIFAKE, synthetic-face collections, image deepfake datasets and social-media videos. They are not bundled. Follow each dataset's official license and access requirements before downloading or redistributing it.

