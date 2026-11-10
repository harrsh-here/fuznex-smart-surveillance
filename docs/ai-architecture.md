# AI architecture and processing flow

`ai/` is a reusable local inference module. It has no camera capture, database,
alert, cloud, or frontend dependency.

```text
OpenCV BGR frame -> SCRFD-500M-KPS detection -> IoU track association
  -> five-landmark SFace alignment -> quality gate -> SFace embedding
  -> in-memory template match -> structured FrameResult
```

`SFacePipeline.process_frame(frame)` returns `FrameResult` with one `FaceResult`
per visible track. Each has display and raw-style boxes, track ID, detector score,
identity ID where available, state, cosine match score, score age, threshold
source, and quality reason.

The backend supplies normalized, model-compatible active templates and owns
persistence. `ai/` never writes biometric data. Tracks are in-memory, bounded by
missed detector updates, and reset/re-measure after a temporary miss.
