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
