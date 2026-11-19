# AI testing results and limitations

Run:

```powershell
python -m unittest discover -s tests -v
```

The integration tests use injected detector/recognizer doubles; no model weights,
camera, images, or biometric templates are required. The existing AI-lab webcam
workflow remains separate and was not modified by this integration preparation.

Limitations: no authorized identity-disjoint dataset is available in this repo,
so no FuzNex accuracy, FAR/FRR, unknown-rejection, low-light, or pose claim is
made. Real model loading needs externally supplied, verified weights and a
rights review.
# SPECTRA DIAG laptop-path check

The combined path can be checked without a camera, using a locally verified
YOLO11 COCO weight and a laptop-positive image:

```powershell
python -m ai.console --yolo C:\models\yolo11n.pt --image C:\tests\office_laptop.jpg
```

This invokes the same `CombinedPipeline` as live capture and exports a new
timestamped `diagnostics/spectra-*` JSONL/CSV trace. It reports class IDs, so a
laptop must be class `63` and a person class `0`; output is never a substitute
for a camera-scene evaluation. `diagnostics/` is ignored by Git and frames are
not stored by this tool.

For a live Windows source (camera number or RTSP/IP URL):

```powershell
python -m ai.console --yolo C:\models\yolo11n.pt --camera 0
```

The viewer independently toggles face/person/laptop overlays, exposes capture
to output timing, dropped frames, errors and reconnects, and retries a lost
source. It does not expose identity candidates or associate faces to people.

