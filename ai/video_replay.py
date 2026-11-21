"""Deterministic, identity-disabled video replay using the production AI package.

Example (from the repository root)::

    python -m ai.video_replay --input /kaggle/input/example/clip.mp4 \
      --output-dir /kaggle/working/fuznex-output --yolo /kaggle/input/models/yolo11n.pt \
      --scrfd /kaggle/input/models/det_500m.onnx --device cuda

The source clip is read sequentially once.  The generated video, detections and
session log contain no face embeddings, gallery entries, names, or identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

from ai.spectra_diag import DiagnosticExport, RunMetrics
from ai.spectra_runtime import build_pipeline, gallery_is_ready, make_plan


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _face_state(value: str | None) -> str:
    """State-only label: identities and similarity scores never leave memory."""
    return {"known": "Known", "unknown": "Unknown", "uncertain": "Uncertain",
            "unevaluable": "Unevaluable", "detection_only": "Face detected"}.get(value or "", "Unevaluable")


def _draw(frame: Any, result: dict[str, Any]) -> Any:
    """Draw only detection/quality state; never render a gallery name or ID."""
    import cv2
    colours = {"face": (0, 200, 255), "person": (0, 220, 0), "laptop": (255, 160, 0), "phone": (255, 80, 180)}
    for detection in result["detections"]:
        x1, y1, x2, y2 = (round(value) for value in detection["bbox_xyxy"])
        kind = detection["kind"]
        label = f"{kind} {detection['detection_confidence']:.2f}"
        if kind == "face":
            quality = detection.get("quality") or {}
            label = _face_state(quality.get("state"))
            if quality.get("reason"):
                label += f": {quality['reason']}"
        colour = colours.get(kind, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(frame, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    return frame


def _h264_writer(cv2: Any, directory: Path, fps: float, width: int, height: int) -> tuple[Any, Path, Path | None]:
    """Prefer direct H.264; use ffmpeg transcoding only if OpenCV lacks it."""
    final_path = directory / "annotated_h264.mp4"
    writer = cv2.VideoWriter(str(final_path), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
    if writer.isOpened():
        return writer, final_path, None
    writer.release()
    temporary = directory / "annotated_intermediate.mp4"
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        writer.release(); raise RuntimeError("Could not create an annotated video writer")
    return writer, final_path, temporary


def _finish_h264(temporary: Path | None, final_path: Path) -> None:
    if temporary is None:
        return
    completed = subprocess.run(["ffmpeg", "-y", "-i", str(temporary), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(final_path)],
                               capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError("H.264 conversion failed: " + (completed.stderr[-500:] or "ffmpeg unavailable"))
    temporary.unlink(missing_ok=True)


def _model_config(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate only explicitly supplied paths; never scan a user's media folders."""
    manifest = json.loads((PROJECT_ROOT / "ai" / "model-manifest.json").read_text(encoding="utf-8"))
    expected = {"scrfd": next(item for item in manifest["models"] if item["id"] == "scrfd-500m-kps").get("sha256")}
    config: dict[str, Any] = {}; availability: dict[str, str] = {}
    for role, path in {"yolo": args.yolo, "scrfd": args.scrfd, "sface": args.sface}.items():
        if path is None or not path.is_file():
            availability[role] = "not_found"; continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected.get(role) and digest.lower() != expected[role].lower():
            availability[role] = "hash_mismatch"; continue
        config[role] = str(path.resolve()); availability[role] = "available"
    return config, availability


def replay(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be positive")
    import cv2
    config, availability = _model_config(args)
    private_ready = bool(args.private_recognition and args.private_gallery and gallery_is_ready(args.private_gallery, config.get("sface")))
    plan = make_plan("integrated", None, availability, gallery_ready=private_ready, private_recognition=args.private_recognition)
    if plan.blocked:
        return {"state": "blocked", "availability": availability, "reason": plan.explanation(),
                "decisions": [item.__dict__ for item in plan.decisions]}
    if args.private_recognition and "sface" not in plan.enabled:
        return {"state": "blocked", "availability": availability,
                "reason": "Private recognition needs a compatible consented gallery and matching local SFace model."}
    # The adapter's configuration is deliberately explicit: Kaggle can use CUDA,
    # while the exact same command works locally with --device cpu.
    pipeline = build_pipeline(plan, config, args.private_gallery, yolo_options={"input_size": args.input_size,
                                                           "confidence": args.confidence,
                                                           "device": args.device, "threads": args.threads})
    input_path = args.input.resolve()
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    output_root = args.output_dir.resolve(); output_root.mkdir(parents=True, exist_ok=True)
    exporter = DiagnosticExport(output_root, {
        "source": {"filename": input_path.name, "replay": "sequential", "frame_start": args.start_frame, "max_frames": args.max_frames},
        "models": {"availability": availability, "selected": {key: Path(value).name for key, value in config.items() if key in {"yolo", "scrfd", "sface"}}},
        "layers": [item.__dict__ for item in plan.decisions],
        "privacy": {"identity_publication": "disabled", "private_gallery": "used_in_memory_only" if args.private_recognition else "not_used", "face_person_association": "disabled", "source_video_retained": "not copied"},
    })
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer, video_path, intermediate = _h264_writer(cv2, exporter.directory, fps, width, height)
    metrics = RunMetrics(); epoch = "replay-" + input_path.stem
    seq = 0; written = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok: break
            if seq < args.start_frame:
                seq += 1; continue
            if args.max_frames is not None and written >= args.max_frames: break
            acquired = time.perf_counter()
            result = pipeline.process_frame(frame, f"video:{input_path.name}", epoch, written, acquired_at=acquired)
            metrics.frames += 1; metrics.inference_ms.append(result["processing"]["total_ms"])
            metrics.face_ms.append(result["processing"].get("face_ms", 0.0)); metrics.object_ms.append(result["processing"].get("object_ms", 0.0))
            metrics.end_to_end_ms.append(result["processing"]["capture_to_output_ms"]); metrics.errors += len(result["processing_errors"])
            exporter.record(result, metrics)
            writer.write(_draw(frame.copy(), result)); written += 1; seq += 1
    finally:
        capture.release(); writer.release()
    _finish_h264(intermediate, video_path)
    exporter.finalize("completed", {"annotated_video": video_path.name, "video_codec": "H.264", "metrics": metrics.snapshot(), "frames_written": written})
    exporter.close()
    return {"state": "completed", "session": str(exporter.directory), "annotated_video": str(video_path),
            "availability": availability, "metrics": metrics.snapshot()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="User-supplied consented or openly licensed video")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--yolo", type=Path); parser.add_argument("--scrfd", type=Path); parser.add_argument("--sface", type=Path)
    parser.add_argument("--private-gallery", type=Path, help="Private, consented gallery input; never copied or exported")
    parser.add_argument("--private-recognition", action="store_true", help="Explicit opt-in: enables SFace only with --sface and --private-gallery")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu"); parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--confidence", type=float, default=0.25); parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--start-frame", type=int, default=0); parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if args.private_recognition and (args.sface is None or args.private_gallery is None):
        parser.error("--private-recognition requires --sface and --private-gallery")
    try:
        print(json.dumps(replay(args), indent=2, sort_keys=True))
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"state": "failed", "reason": str(exc)}, indent=2)); return 2


if __name__ == "__main__":
    raise SystemExit(main())
