# AI integration module

`ai/` exposes reusable local face, person/laptop, and identity-disabled combined
frame pipelines. It has no database, alerts, cloud, or persistence dependency.

```python
from ai.config import SFaceConfig
from ai.detection import SCRFD500MDetector
from ai.pipeline import SFacePipeline
from ai.recognition import SFaceRecognizer

config = SFaceConfig(detector_model=..., recognizer_model=...)
pipeline = SFacePipeline(config, SCRFD500MDetector(config.detector_model), SFaceRecognizer(config.recognizer_model))
pipeline.set_templates({"backend-person-id": normalized_embedding})
result = pipeline.process_frame(frame)  # OpenCV BGR ndarray
payload = result.to_dict()
```

`match_score` is cosine similarity, not a confidence percentage. The default
SFace reference threshold is experimental and is not a FuzNex accuracy claim.
See `../docs/ai-architecture.md` and `model-manifest.json` before integration.

Portable integration, Windows/Kaggle setup, safety limits, and benchmark guidance
are in [`../docs/portable-ai-migration.md`](../docs/portable-ai-migration.md).
`object_detection.py` accepts only a supplied local, verified YOLO COCO model.
`combined.py` deliberately does not associate a face with a person or publish a
face identity. `console.py` is optional and intended for local Windows testing;
headless consumers should call `CombinedPipeline.process_frame` directly.

## Integration handoff

Backend and independent consumers must start with
[`INTEGRATION_HANDOFF.md`](INTEGRATION_HANDOFF.md). It documents the actual
`1.1.0-draft` combined-observation output, sanitized examples, verification,
failure boundaries, runtime setup, and the privacy/unsupported-field rules.

## Private SFace enrollment (operator workstation only)

The archived LBPH data is not compatible with SFace and is never read or
converted. Start a new, explicitly consented gallery only from fresh samples;
the commands below create an ignored local JSON file containing embeddings and
model metadata, not source images. The participant IDs are intentionally fixed
to `participant_1`, `participant_2`, and `participant_3`.

```powershell
python -m ai.enrollment --gallery local_data/sface/private-gallery.json --create --participant-id participant_1 --consent I_CONSENT --scrfd C:\models\det_500m.onnx --sface C:\models\face_recognition_sface_2021dec.onnx --camera 0 --samples 5
```

For headless operation, replace `--camera` with at least three repeated
`--image` paths supplied by the consenting participant. Each sample must
contain exactly one sufficiently large, sharp face; rejected frames and all
source images stay out of the gallery. Updating an existing participant creates
a timestamped sibling `.bak` backup. Delete only through an explicit command:

```powershell
python -m ai.enrollment --gallery local_data/sface/private-gallery.json --participant-id participant_1 --delete --confirm-delete participant_1
```

SPECTRA DIAG remains identity-disabled by default and its JSONL/CSV output
never contains identity data. A local terminal-only check is available after
enrollment; it prints a participant ID only for `known`, never for `unknown` or
`uncertain`, and writes no export:

```powershell
python -m ai.console --local-diagnostic --scrfd C:\models\det_500m.onnx --sface C:\models\face_recognition_sface_2021dec.onnx --gallery local_data/sface/private-gallery.json --private-recognition-image C:\consented\check.jpg
```

Use the official [OpenCV Zoo SFace model](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface), whose model directory declares Apache-2.0, and compare its SHA-256 with `ai/model-manifest.json` before use. This does not replace the operator's consent, retention, security, or applicable biometric-law obligations.
