# FuzNex AI integration handoff

**Audience:** Govind/backend and any independent consumer of the combined local AI output.  
**Implementation source:** [`combined.py`](combined.py), current contract version `1.1.0-draft`.  
**Status:** provisional local observation contract, not a final API or an event protocol.

This is the central handoff for the identity-disabled combined pipeline. Read it before consuming observations, designing backend storage, or adding frontend fields.

## Non-negotiable boundary

`CombinedPipeline.process_frame(...)` produces **one observation for one supplied frame**. It does not create alerts, write a database, own camera transport, store video, persist a gallery, publish an identity, associate faces with person boxes, or establish a distance/tracking/consensus result.

`events` is currently always `[]`. Do not convert it, a raw detection, a count, or a timing value into a backend event or alert without a separately reviewed, versioned event contract.

## Invocation and ordering

```python
result = combined.process_frame(
    frame,                         # OpenCV BGR ndarray
    camera_id="camera-owned-id",   # nonempty caller value
    stream_epoch="opaque-session", # nonempty caller value
    frame_seq=0,                   # non-negative caller value
    captured_at=None,              # optional capture-host timestamp
    acquired_at=None,              # optional perf-counter timestamp
)
```

Use a new opaque `stream_epoch` when the producer process/session restarts. Consumers may order `frame_seq` only within the same `(camera_id, stream_epoch)` pair. A gap is a capture/transport fact, not an event or a no-detection observation.

## Actual emitted schema

The implementation returns these top-level fields on every successful frame-processing call:

| Field | Actual behavior |
|---|---|
| `schema_version` | Literal `"1.1.0-draft"`. Reject other versions until explicitly supported. |
| `camera_id`, `stream_epoch`, `frame_seq` | Caller values after nonempty/non-negative validation. |
| `captured_at` | Caller value or current UTC ISO-8601 timestamp with millisecond precision and `Z`. |
| `timestamp_source` | Literal `"capture_host"`. |
| `frame_size` | `{ "width": int, "height": int }` derived from the input frame. |
| `models` | `{ "face": "scrfd-500m+sface", "object": "yolo11n" }`; labels only, not an approval/provenance record. |
| `detections` | Face detections first, followed by object detections. `detection_id` is assigned as `d0`, `d1`, ... in that final order. |
| `events` | Always `[]` in this version. No event schema exists. |
| `processing_errors` | Sanitized branch failures only: `{ "stage": "face" | "object", "code": "INFERENCE_FAILED", "retryable": true }`. One branch can succeed while the other fails. |
| `detection_trace` | `{ "raw_emitted_counts": { "face": int, "person": int, "laptop": int } }`. This is diagnostic evidence, never identity evidence. |
| `processing` | Numeric branch/overall timing: `face_ms`, `object_ms`, `total_ms`, `capture_to_output_ms`. Local telemetry only; no performance/SLA claim. |

The producer clamps each `bbox_xyxy` to the input frame and rejects non-finite, empty, or inverted boxes. It does not itself declare a contract-wide confidence range, so consumers must not add an unsupported threshold or probability interpretation.

### Detection fields

Every emitted detection has `detection_id`, `kind`, `bbox_xyxy`, `detection_confidence`, `track: null`, `verified_identity: null`, and `identity_state`.

| Kind | Additional emitted fields |
|---|---|
| `face` | `identity_state: "identity_disabled"`; `quality` with `state`, `reason`, `measurement_age_ms`, `pixel_width`, `pixel_height`, plus literal `blur: "not_measured_by_combined_contract"` and `landmarks: "not_published_by_combined_contract"`; `uncertainty_reasons: ["IDENTITY_ASSOCIATION_DISABLED"]`. No `class_id`. |
| `person` | `class_id: 0` from the supplied COCO adapter; `identity_state: "not_applicable"`; `quality: null`; `uncertainty_reasons: []`. |
| `laptop` | `class_id: 63` from the supplied COCO adapter; `identity_state: "not_applicable"`; `quality: null`; `uncertainty_reasons: []`. |

The bundled YOLO adapter only invokes COCO classes 0 (`person`) and 63 (`laptop`) and checks that mapping while loading. See [`object_detection.py`](object_detection.py). A custom detector passed into `CombinedPipeline` must preserve the same restricted kinds.

## Sanitized adapter-shaped outputs

The numbers below are illustrative only; they are structurally derived from `CombinedPipeline`, contain no image, template, credential, enrolled identity, or real camera identifier, and are not benchmark data.

### Person

```json
{"schema_version":"1.1.0-draft","camera_id":"cam-demo","stream_epoch":"epoch-a","frame_seq":7,"captured_at":"2026-11-19T09:00:00.000Z","timestamp_source":"capture_host","frame_size":{"width":640,"height":480},"models":{"face":"scrfd-500m+sface","object":"yolo11n"},"detections":[{"kind":"person","bbox_xyxy":[10.0,20.0,110.0,300.0],"detection_confidence":0.88,"track":null,"class_id":0,"verified_identity":null,"identity_state":"not_applicable","quality":null,"uncertainty_reasons":[],"detection_id":"d0"}],"events":[],"processing_errors":[],"detection_trace":{"raw_emitted_counts":{"face":0,"person":1,"laptop":0}},"processing":{"face_ms":2.1,"object_ms":8.4,"total_ms":10.7,"capture_to_output_ms":12.0}}
```

### Face, intentionally identity-disabled

```json
{"schema_version":"1.1.0-draft","camera_id":"cam-demo","stream_epoch":"epoch-a","frame_seq":8,"captured_at":"2026-11-19T09:00:00.100Z","timestamp_source":"capture_host","frame_size":{"width":640,"height":480},"models":{"face":"scrfd-500m+sface","object":"yolo11n"},"detections":[{"kind":"face","bbox_xyxy":[80.0,40.0,160.0,140.0],"detection_confidence":0.93,"track":null,"verified_identity":null,"identity_state":"identity_disabled","quality":{"state":"unknown","reason":null,"measurement_age_ms":0,"pixel_width":80.0,"pixel_height":100.0,"blur":"not_measured_by_combined_contract","landmarks":"not_published_by_combined_contract"},"uncertainty_reasons":["IDENTITY_ASSOCIATION_DISABLED"],"detection_id":"d0"}],"events":[],"processing_errors":[],"detection_trace":{"raw_emitted_counts":{"face":1,"person":0,"laptop":0}},"processing":{"face_ms":3.3,"object_ms":8.2,"total_ms":11.7,"capture_to_output_ms":13.0}}
```

### No detection and a partial branch failure

```json
{"schema_version":"1.1.0-draft","camera_id":"cam-demo","stream_epoch":"epoch-a","frame_seq":9,"captured_at":"2026-11-19T09:00:00.200Z","timestamp_source":"capture_host","frame_size":{"width":640,"height":480},"models":{"face":"scrfd-500m+sface","object":"yolo11n"},"detections":[],"events":[],"processing_errors":[],"detection_trace":{"raw_emitted_counts":{"face":0,"person":0,"laptop":0}},"processing":{"face_ms":2.0,"object_ms":8.0,"total_ms":10.2,"capture_to_output_ms":11.4}}
```

```json
{"schema_version":"1.1.0-draft","camera_id":"cam-demo","stream_epoch":"epoch-a","frame_seq":10,"captured_at":"2026-11-19T09:00:00.300Z","timestamp_source":"capture_host","frame_size":{"width":640,"height":480},"models":{"face":"scrfd-500m+sface","object":"yolo11n"},"detections":[],"events":[],"processing_errors":[{"stage":"object","code":"INFERENCE_FAILED","retryable":true}],"detection_trace":{"raw_emitted_counts":{"face":0,"person":0,"laptop":0}},"processing":{"face_ms":2.1,"object_ms":2.2,"total_ms":4.4,"capture_to_output_ms":5.1}}
```

`contracts/observation.example.json` is a deliberately smaller independent-consumer sample. It is not a complete runtime output: it omits capture/timing/model/error fields and includes a human `notes` field which `CombinedPipeline` itself does not emit.

## Setup and verification

The current `scripts/setup_windows_cpu.ps1` is the optional **YOLO runtime**
installer: it creates `.venv` and installs `ai/requirements/yolo.txt`. It is
not the dependency-light contract-test setup and it does not install `pytest`.
Use it only when a locally approved YOLO weight will be exercised.

For the independent consumer/contract checks, install the explicit test
dependencies in an isolated environment from the repository root:

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install numpy pytest
python -m unittest discover -s tests -p "test_combined.py" -v
python -m pytest tests/test_observation_contract.py -q
```

The broader diagnostic test is optional for a contract consumer but can run in
the same environment:

```powershell
python -m pytest -q
```

Model-backed inference additionally needs explicit, locally approved files outside Git. Install `ai/requirements/yolo.txt` for the YOLO path. CUDA is optional: first install the PyTorch build matching the local NVIDIA driver, then `ai/requirements/windows-nvidia.txt`; otherwise use `device="cpu"`. Do not infer camera FPS, latency, hardware sizing, or accuracy from unit tests.

Before a real run, compute model hashes and check [`model-manifest.json`](model-manifest.json). The manifest says SCRFD provenance/license is unresolved and YOLO11n has no approved checksum; neither may be automatically downloaded or redistributed. SFace has the stated official OpenCV Zoo source and SHA-256, but deployment consent, retention, security, and applicable biometric-law obligations remain with the operator.

## Failure and source-health semantics

- Invalid call inputs (`camera_id`, `stream_epoch`, or `frame_seq`) raise `ValueError`; no observation is produced.
- A face or object branch exception produces the sanitized `processing_errors` entry and permits the other branch’s detections to remain.
- The observation contract has **no** `stale`, `disconnect`, `reconnect`, dropped-frame, camera-open, or capture-error field. If the camera cannot yield a frame, do not fabricate a `1.1.0-draft` observation.
- Capture/source status belongs to a separately versioned camera/transport status channel owned by camera/backend code. The SPECTRA diagnostic UI may show local retry status, but that is not an observation schema extension.

## Privacy and unsupported fields

Never place raw frames, crops, embeddings, gallery JSON, templates, real identity labels, credentials, recordings, local databases, or model files in Git, frontend payloads, or this integration contract.

The following are unsupported in `1.1.0-draft` and must be rejected/deferred rather than inferred: `track` values, `verified_identity` values, identity candidates, face-to-person association, person identity, distance, multi-frame consensus, alert/event semantics, stale/disconnect observations, and camera persistence. A recognition pipeline may exist separately for consented local diagnostic workflows, but it is not part of this combined production handoff.

## Consumer acceptance rules

1. Reject unknown schema versions.
2. Require nonempty camera/epoch values and a non-negative sequence; scope ordering to a single epoch.
3. Validate boxes against `frame_size`; preserve null identity and track fields.
4. Treat `processing_errors` as diagnostic health, `detection_trace` as evidence only, and `events` as empty.
5. Keep camera-ingest health, alerting, persistence, and retention design outside this contract until their own review/contract is approved.
