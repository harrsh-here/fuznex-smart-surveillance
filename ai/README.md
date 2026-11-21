# AI integration module

`ai/` exposes reusable local face, person/laptop/phone, and identity-disabled combined
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

## Private SFace enrollment (operator workstation only)

The archived LBPH data is not compatible with SFace and is never read or
converted. Start a new, explicitly consented gallery only from fresh samples;
the commands below create an ignored local JSON file containing embeddings and
model metadata, not source images. There is no participant-number cap; generated
local IDs and display names stay in the ignored private gallery.

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

## SPECTRA pipeline sessions

## Minimal Windows live camera

`ai.live_camera` is the direct OpenCV live preview. It uses the same runtime
plan and `CombinedPipeline` as SPECTRA, but it does not write a diagnostic
export, save frames, or copy a gallery. Press `s` to stop/start and `q` or
Escape to close. `--private-recognition` is a local-only opt-in: display names
are shown only for a `known` SFace result and stay in the window memory.

```powershell
python -m ai.live_camera --source 0 --yolo C:\models\yolo11n.pt --scrfd C:\models\det_500m.onnx --sface C:\models\face_recognition_sface_2021dec.onnx --gallery .\local_data\spectra\private-gallery.json --private-recognition
```

For an IP stream, replace `--source 0` with a supported URL such as
`--source "rtsp://user:password@camera.example/live"`; keep credentials out of
Git and shell history where possible. Without `--private-recognition`, faces
remain detection-only and no names are displayed.

`python -m ai.spectra_console` starts the local Windows console. Its Integrated
Pipeline is the locked default: it requires the discovered YOLO and SCRFD
models, creates a unique session on every test, and blocks instead of silently
substituting a missing core layer. Private SFace recognition is available only
after local opt-in and a valid consented gallery.

The same planner and production components are available headlessly for local
images (a remote Codex session is not assumed to have a camera):

```powershell
python -m ai.spectra_runner --mode integrated --image C:\test\frame.jpg
python -m ai.spectra_runner --mode layer --layers yolo,scrfd --image C:\test\frame.jpg
```

Every session is stored under the ignored `diagnostics/spectra-*` directory:
`events.jsonl`, `detections.csv`, `provenance.json`, and
`session-summary.json`. Ordinary exports exclude images, embeddings, gallery
contents, display names, identity candidates, and face-to-person associations.
