# Face-detection research

This archive records the useful, non-biometric outcome of the prior detector
evaluation. Original benchmark scripts, downloaded weights, WIDERFace data,
live camera results, and run output remain only in the old workspace and are
excluded from this repository by design.

## Retained result

The [2026-11-03 benchmark summary](benchmark_summary_2026-11-03.md) is a
hardware-specific live-camera measurement, not an accuracy claim. It found
YuNet at 10.46 baseline FPS (58.28 ms median inference) and 14.91 max FPS
(34.30 ms median); SCRFD-2.5G-KPS measured 4.15 and 9.24 FPS respectively.
The source summary notes that labeled ground truth is required for accuracy,
recall, or false-positive-rate claims.

The old workspace's offline source audit covered OpenCV Zoo, InsightFace, YOLO
face variants, CenterFace, RetinaFace, MTCNN, and MediaPipe. Re-run it from a
clean environment if needed; do not commit third-party READMEs or downloaded
artifacts without reviewing licenses.
