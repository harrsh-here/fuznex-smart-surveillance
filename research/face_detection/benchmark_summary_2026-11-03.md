# Consolidated live face-detector benchmark — 2026-11-03

Migrated from the old workspace's lightweight `summary.txt` only. The source
used camera `0`, requested 640×480 input, detector input size 640, threshold
0.5, and 20 seconds per model/pass.

| Model | Baseline FPS | Max FPS | Baseline median inference | Max median inference |
|---|---:|---:|---:|---:|
| YuNet | 10.46 | 14.91 | 58.28 ms | 34.30 ms |
| SCRFD-500M-KPS | skipped | skipped | — | — |
| SCRFD-2.5G-KPS | 4.15 | 9.24 | 235.54 ms | 101.53 ms |
| YOLOv5n-face | skipped | skipped | — | — |
| YOLOv5s-face | skipped | skipped | — | — |

FPS includes capture, inference, and display; latency is inference-only.
Skipped entries lacked compatible ONNX weights. No accuracy, recall, or
false-positive-rate measurement was made because no labeled ground truth was
supplied. Provider availability does not establish GPU execution.
