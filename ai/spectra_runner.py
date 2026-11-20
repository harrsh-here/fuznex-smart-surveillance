"""Headless SPECTRA runner using the same execution plan as the desktop console.

Example:
``python -m ai.spectra_runner --mode integrated --image C:\\test\\frame.jpg``
``python -m ai.spectra_runner --mode layer --layers yolo,scrfd --image C:\\test\\frame.jpg``
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ai.runtime_discovery import discover_models
from ai.spectra_diag import DiagnosticExport, RunMetrics, replay_frames
from ai.spectra_runtime import build_pipeline, gallery_is_ready, make_plan

APP_DIR = Path(__file__).resolve().parent.parent
GALLERY_PATH = APP_DIR / "local_data" / "spectra" / "private-gallery.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("integrated", "layer"), default="integrated")
    parser.add_argument("--layers", default="yolo,scrfd", help="comma-separated layer selection for --mode layer")
    parser.add_argument("--private-recognition", action="store_true", help="local-only; requires SFace + a consented gallery")
    parser.add_argument("--image", type=Path, required=True, help="local BGR-readable image; no camera access is assumed")
    parser.add_argument("--sessions-dir", type=Path, default=APP_DIR / "diagnostics")
    args = parser.parse_args()
    import cv2
    frame = cv2.imread(str(args.image))
    if frame is None:
        parser.error("--image must name a readable local image")
    discovered = discover_models(APP_DIR, {})
    requested = {item.strip() for item in args.layers.split(",") if item.strip()}
    ready = gallery_is_ready(GALLERY_PATH, discovered.config.get("sface"))
    plan = make_plan(args.mode, requested, discovered.availability, ready, args.private_recognition)
    if plan.blocked:
        print(json.dumps({"state": "blocked", "reason": plan.explanation(), "decisions": [item.__dict__ for item in plan.decisions]}, indent=2))
        return 2
    pipeline = build_pipeline(plan, discovered.config, GALLERY_PATH)
    provenance = {"session_mode": plan.mode, "enabled_layers": plan.active_layers,
                  "layer_decisions": [item.__dict__ for item in plan.decisions],
                  "privacy": {"identity_publication": "disabled", "face_person_association": "disabled", "captures": "not_retained"}}
    exporter = DiagnosticExport(args.sessions_dir, provenance)
    exporter.record_event("headless_started", {"input": args.image.name, "mode": plan.mode, "enabled_layers": plan.active_layers})
    results, metrics = replay_frames(pipeline, [frame], f"image:{args.image.name}", exporter=exporter)
    exporter.finalize("completed", {"mode": plan.mode, "layers": plan.active_layers})
    exporter.close()
    result = results[0]
    print(json.dumps({"state": "completed", "session": str(exporter.directory), "layers": result["layers"],
                      "counts": result["detection_trace"]["raw_emitted_counts"], "metrics": metrics.snapshot()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
