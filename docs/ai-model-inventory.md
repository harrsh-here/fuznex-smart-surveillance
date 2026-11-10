# Model inventory and licensing notes

See `ai/model-manifest.json` for exact filenames, SHA-256 values, and byte sizes.

| Model | Status |
|---|---|
| `det_500m.onnx` | SCRFD-500M-KPS local artifact; provenance and license are unresolved. Do not upload, redistribute, auto-download, or propose LFS. |
| `face_recognition_sface_2021dec.onnx` | SHA verified against the local lab's prior official OpenCV Zoo acquisition. Pretrained-weight commercial and redistribution permissions remain unresolved. Do not bundle, auto-fetch, or claim clearance. |

Weights are deliberately ignored by Git. A verified source and explicit rights
review are mandatory before any future artifact distribution decision.
