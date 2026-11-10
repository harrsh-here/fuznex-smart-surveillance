# AI integration module

`ai/` exposes a reusable, local SCRFD-500M-KPS + SFace frame pipeline. It has no
camera capture, database, alerts, UI, cloud, or persistence dependency.

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
