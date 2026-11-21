"""Minimal Windows live-camera runner for the shared FuzNex AI pipeline.

Press ``s`` to stop/start capture, and ``q`` or Escape to exit.  This command
does not create a diagnostic export, save frames, or copy a private gallery.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import uuid
from typing import Any

from ai.local_gallery import sha256_file
from ai.spectra_runtime import build_pipeline, gallery_is_ready, make_plan


APP_ROOT = Path(__file__).resolve().parent.parent


def source_value(value: str) -> int | str:
    """Allow webcam indices and OpenCV-supported IP stream URLs."""
    return int(value) if value.isdecimal() else value


def explicit_availability(paths: dict[str, Path | None]) -> tuple[dict[str, str], dict[str, str]]:
    """Check only caller-provided files; never scan disks or download weights."""
    manifest = json.loads((APP_ROOT / "ai" / "model-manifest.json").read_text(encoding="utf-8"))
    scrfd = next(item for item in manifest["models"] if item["id"] == "scrfd-500m-kps")
    availability: dict[str, str] = {}; config: dict[str, str] = {}
    for role, path in paths.items():
        if path is None or not path.is_file():
            availability[role] = "not_found"; continue
        if role == "scrfd" and sha256_file(path).lower() != str(scrfd["sha256"]).lower():
            availability[role] = "hash_mismatch"; continue
        availability[role] = "available"; config[role] = str(path.resolve())
    return availability, config


def face_label(face: Any, private_names: dict[Any, str]) -> str:
    """Return the state honestly; show a name only for an exact known match."""
    raw_state = str(getattr(face, "state", "unevaluable"))
    if raw_state == "detection_only":
        return "Face detected (recognition off)"
    state = raw_state.capitalize()
    if raw_state == "known":
        name = private_names.get(getattr(face, "identity_id", None))
        return f"Known: {name}" if name else "Known"
    reason = getattr(face, "quality_reason", None)
    return f"{state}: {reason}" if reason else state


def annotate(frame: Any, result: dict[str, Any], pipeline: Any, show_private_names: bool) -> Any:
    """Overlay independent detections; face/person boxes are never associated."""
    import cv2
    colours = {"face": (0, 200, 255), "person": (0, 220, 0), "laptop": (255, 160, 0), "phone": (255, 80, 180)}
    private_names = getattr(pipeline, "_private_display_names", {}) if show_private_names else {}
    face_rows = list(getattr(pipeline, "last_private_faces", ()))
    for item in result["detections"]:
        x1, y1, x2, y2 = (round(value) for value in item["bbox_xyxy"])
        kind = item["kind"]; colour = colours[kind]
        label = f"{kind} {item['detection_confidence']:.2f}"
        if kind == "face":
            # The shared pipeline emits faces in the same order it stores them;
            # no person association is inferred here.
            face = face_rows.pop(0) if face_rows else None
            label = face_label(face, private_names) if face is not None else "Unevaluable"
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(frame, label, (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .48, colour, 1, cv2.LINE_AA)
    return frame


def run(args: argparse.Namespace) -> int:
    import cv2
    paths = {"yolo": args.yolo, "scrfd": args.scrfd, "sface": args.sface}
    availability, config = explicit_availability(paths)
    private_ready = bool(args.private_recognition and gallery_is_ready(args.gallery, config.get("sface")))
    plan = make_plan("integrated", None, availability, private_ready, args.private_recognition)
    if plan.blocked:
        print(json.dumps({"state": "blocked", "availability": availability, "reason": plan.explanation()}, indent=2)); return 2
    if args.private_recognition and "sface" not in plan.enabled:
        print(json.dumps({"state": "blocked", "reason": "Private recognition needs a valid consented local gallery and matching SFace model."}, indent=2)); return 2
    pipeline = build_pipeline(plan, config, args.gallery, {"input_size": args.input_size, "confidence": args.confidence,
                                                           "device": args.device, "threads": args.threads})
    source = source_value(args.source); capture = None; running = True; frame_seq = 0; epoch = uuid.uuid4().hex
    samples: list[float] = []; errors = 0
    print("Live camera ready. Press s to stop/start; q or Escape to quit.")
    try:
        while True:
            if running and (capture is None or not capture.isOpened()):
                capture = cv2.VideoCapture(source, cv2.CAP_DSHOW if isinstance(source, int) else cv2.CAP_ANY)
                if not capture.isOpened():
                    errors += 1; canvas = None
                else:
                    epoch = uuid.uuid4().hex
            if running and capture is not None and capture.isOpened():
                acquired = time.perf_counter(); ok, frame = capture.read()
                if not ok:
                    errors += 1; capture.release(); capture = None; canvas = None
                else:
                    result = pipeline.process_frame(frame, f"live:{args.source}", epoch, frame_seq, acquired_at=acquired); frame_seq += 1
                    total_ms = result["processing"]["total_ms"]; samples.append(total_ms); samples = samples[-30:]
                    errors += len(result["processing_errors"]); fps = 1000 / (sum(samples) / len(samples)) if samples else 0
                    canvas = annotate(frame.copy(), result, pipeline, args.private_recognition)
                    text = f"{fps:.1f} FPS | YOLO {result['processing'].get('yolo_ms', 0):.1f} ms | SCRFD {result['processing'].get('scrfd_ms', 0):.1f} ms | SFace {result['processing'].get('sface_ms', 0):.1f} ms | errors {errors}"
                    cv2.rectangle(canvas, (8, 8), (min(canvas.shape[1] - 8, 950), 36), (20, 20, 20), -1)
                    cv2.putText(canvas, text, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, .45, (245, 245, 245), 1, cv2.LINE_AA)
            else:
                canvas = None
            if canvas is None:
                canvas = __import__("numpy").zeros((180, 820, 3), dtype="uint8")
                status = "Stopped — press s to start" if not running else "Camera/IP stream unavailable — press q to exit"
                cv2.putText(canvas, status, (18, 85), cv2.FONT_HERSHEY_SIMPLEX, .65, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.imshow("FuzNex Live Camera — s start/stop, q exit", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27): break
            if key == ord("s"):
                running = not running
                if not running and capture is not None: capture.release(); capture = None
    finally:
        if capture is not None: capture.release()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="Webcam index (default 0) or an IP stream URL")
    parser.add_argument("--yolo", required=True, type=Path); parser.add_argument("--scrfd", required=True, type=Path)
    parser.add_argument("--sface", type=Path); parser.add_argument("--gallery", type=Path)
    parser.add_argument("--private-recognition", action="store_true", help="Explicit local-only opt-in; needs --sface and --gallery")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu"); parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--confidence", type=float, default=.25); parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.private_recognition and (args.sface is None or args.gallery is None): parser.error("--private-recognition requires --sface and --gallery")
    return run(args)


if __name__ == "__main__": raise SystemExit(main())
