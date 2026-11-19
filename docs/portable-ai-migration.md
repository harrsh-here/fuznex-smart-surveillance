# Portable AI integration

## Scope

This branch consolidates reusable AI code into `ai/` while retaining the
original AI Lab, Stage A workspace, and Combined Console workspace as evidence
and development history. It does not migrate biometrics, recordings, model
files, secrets, caches, or generated reports.

## Source evidence

| Source | Verified commit | Reused scope |
|---|---|---|
| Monorepo Face AI | `3f31a7b` | SCRFD/SFace pipeline and safety rules |
| Face AI Lab | `9a96186` | identity disclosure safeguards |
| Stage A | `534b0fd` | YOLO person/laptop adapter design |
| Combined console | `e068054` | identity-disabled output contract and replay safety |

Stage A contained additional uncommitted work at audit time. It is intentionally
excluded from this migration.

## Layout

- `ai/pipeline.py`: reusable SCRFD/SFace processing.
- `ai/object_detection.py`: YOLO11 COCO person/laptop adapter.
- `ai/combined.py`: headless frame contract; never links a face to a person.
- `ai/console.py`: optional Windows Tkinter camera viewer.
- `ai/notebooks/kaggle_portable_ai.ipynb`: repository-based Kaggle bootstrap.
- `scripts/benchmark_headless.py`: safe benchmark-record scaffold.

## Setup

Windows CPU:

```powershell
./scripts/setup_windows_cpu.ps1
python -m unittest discover -s tests -v
```

Windows NVIDIA GPU: install a PyTorch CUDA build compatible with the local
driver, then install `ai/requirements/windows-nvidia.txt`. Kaggle: run the
notebook with GPU enabled, but use its repository checkout rather than copying
notebook code into a new implementation.

Models are always outside Git. Obtain only approved artifacts, calculate their
SHA-256 values, compare them with `ai/model-manifest.json`, and pass their paths
explicitly. A model with a missing checksum or unresolved license is not
approved for automatic download.

## Verification and troubleshooting

`python -m unittest discover -s tests -v` runs dependency-light contract tests.
Model-backed tests require explicit local approved weights and their applicable
runtime. If `ultralytics` is unavailable, install `ai/requirements/yolo.txt`.
If CUDA is unavailable, set `device="cpu"`; do not assume GPU use merely because
a GPU is present.

The benchmark scaffold refuses to overwrite output files. Record camera capture,
queueing, rendering, model paths, hardware, and consent conditions separately;
the earlier combined replay did not measure them.

## Safety boundary

`verified_identity` and `track` are always `null` for combined detections.
Temporary face tracking inside the face pipeline must not be treated as a
person-track association. Do not enable association until controlled multi-person
crossing, occlusion, false-match, stale-output, reconnect, and consented live
camera validation are reviewed and approved.
