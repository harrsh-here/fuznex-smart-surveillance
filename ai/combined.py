"""Safe, headless-friendly combined frame contract.

Face and object detections share a frame only. This module never associates a
face with a person box and never publishes an identity candidate.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
import time
from typing import Any


SCHEMA_VERSION = "1.1.0-draft"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _box(value: Any, width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = (float(item) for item in value)
    if not all(math.isfinite(item) for item in (x1, y1, x2, y2)):
        raise ValueError("invalid bounding box")
    result = [max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)]
    if result[0] >= result[2] or result[1] >= result[3]:
        raise ValueError("invalid bounding box")
    return result


class CombinedPipeline:
    def __init__(self, face_pipeline: Any | None, object_detector: Any | None, clock=time.perf_counter,
                 enabled_layers: frozenset[str] | set[str] | None = None,
                 layer_decisions: tuple[Any, ...] | list[Any] = (),
                 private_display_names: dict[str, str] | None = None) -> None:
        self.face_pipeline = face_pipeline
        self.object_detector = object_detector
        self.clock = clock
        self.enabled_layers = frozenset(enabled_layers or {"capture", "yolo", "scrfd"})
        self.layer_decisions = tuple(layer_decisions)
        # Deliberately transient: this is for the local GUI's explicitly opted-in
        # recognition indicator only. It is never copied into the frame contract,
        # event log, diagnostic export, or session summary.
        self.last_private_faces: tuple[Any, ...] = ()
        # Memory-only data for the explicitly opted-in local preview. This
        # mapping never enters the frame contract, a log, or an export.
        self._private_display_names = dict(private_display_names or {})

    def private_known_labels(self) -> tuple[tuple[Any, str], ...]:
        """Return display labels for exact known matches, never candidates."""
        labels: list[tuple[Any, str]] = []
        for face in self.last_private_faces:
            if getattr(face, "state", None) != "known":
                continue
            display_name = self._private_display_names.get(getattr(face, "identity_id", None))
            if display_name:
                labels.append((face, display_name))
        return tuple(labels)

    def process_frame(self, frame: Any, camera_id: str, stream_epoch: str, frame_seq: int,
                      captured_at: str | None = None, acquired_at: float | None = None) -> dict[str, Any]:
        if not camera_id or not stream_epoch or frame_seq < 0:
            raise ValueError("camera_id, stream_epoch, and non-negative frame_seq are required")
        height, width = frame.shape[:2]
        start = self.clock()
        detections: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        timings: dict[str, float] = {}
        if "scrfd" not in self.enabled_layers:
            self.last_private_faces = ()
            timings["scrfd_ms"] = 0.0
            timings["sface_ms"] = 0.0
        elif self.face_pipeline is None:
            self.last_private_faces = ()
            errors.append({"stage": "scrfd", "code": "LAYER_NOT_CONSTRUCTED", "retryable": False})
            timings["scrfd_ms"] = 0.0
            timings["sface_ms"] = 0.0
        else:
            try:
                branch_start = self.clock()
                private_faces = tuple(self.face_pipeline.process_frame(frame).faces)
                self.last_private_faces = private_faces
                for face in private_faces:
                    box = _box(face.bbox_xyxy, width, height)
                    detections.append({
                        "kind": "face", "bbox_xyxy": box,
                        "detection_confidence": float(face.detection_score), "track": None,
                        "verified_identity": None, "identity_state": "identity_disabled",
                        "quality": {"state": face.state, "reason": face.quality_reason,
                                    "measurement_age_ms": face.score_age_ms,
                                    "pixel_width": round(box[2] - box[0], 1),
                                    "pixel_height": round(box[3] - box[1], 1),
                                    "blur": "not_measured_by_combined_contract",
                                    "landmarks": "not_published_by_combined_contract"},
                        "uncertainty_reasons": ["IDENTITY_ASSOCIATION_DISABLED"],
                    })
                timings["face_ms"] = (self.clock() - branch_start) * 1000
                reported = getattr(self.face_pipeline, "last_timings", {})
                timings["scrfd_ms"] = float(reported.get("scrfd_ms", timings["face_ms"]))
                timings["sface_ms"] = float(reported.get("sface_ms", 0.0))
            except Exception:
                self.last_private_faces = ()
                errors.append({"stage": "face", "code": "INFERENCE_FAILED", "retryable": True})
                timings["face_ms"] = (self.clock() - start) * 1000
                timings["scrfd_ms"] = timings["face_ms"]
                timings["sface_ms"] = 0.0
        if "yolo" not in self.enabled_layers:
            timings["yolo_ms"] = 0.0
            timings["object_ms"] = 0.0
        elif self.object_detector is None:
            errors.append({"stage": "yolo", "code": "LAYER_NOT_CONSTRUCTED", "retryable": False})
            timings["yolo_ms"] = 0.0
            timings["object_ms"] = 0.0
        else:
            try:
                branch_start = self.clock()
                for item in self.object_detector.detect(frame):
                    kind = item["kind"]
                    if kind not in {"person", "laptop", "phone"}:
                        raise ValueError("object detector returned unsupported class")
                    detections.append({"kind": kind, "bbox_xyxy": _box(item["bbox_xyxy"], width, height),
                                       "detection_confidence": float(item["detection_confidence"]), "track": None,
                                       "class_id": int(item.get("class_id", {"person": 0, "laptop": 63, "phone": 67}[kind])),
                                       "verified_identity": None, "identity_state": "not_applicable",
                                       "quality": None, "uncertainty_reasons": []})
                timings["object_ms"] = (self.clock() - branch_start) * 1000
                timings["yolo_ms"] = timings["object_ms"]
            except Exception:
                errors.append({"stage": "object", "code": "INFERENCE_FAILED", "retryable": True})
                timings["object_ms"] = (self.clock() - start) * 1000
                timings["yolo_ms"] = timings["object_ms"]
        for index, detection in enumerate(detections):
            detection["detection_id"] = f"d{index}"
        finished = self.clock()
        # Counts allow a duplicate-box report to be tied to a specific frame.
        counts = {kind: sum(item["kind"] == kind for item in detections) for kind in ("face", "person", "laptop", "phone")}
        return {"schema_version": SCHEMA_VERSION, "camera_id": camera_id, "stream_epoch": stream_epoch,
                "frame_seq": frame_seq, "captured_at": captured_at or utc_now(),
                "timestamp_source": "capture_host", "frame_size": {"width": width, "height": height},
                "models": {"face": "scrfd-500m+sface", "object": "yolo11n"}, "detections": detections,
                "events": [], "processing_errors": errors, "detection_trace": {"raw_emitted_counts": counts},
                "layers": {"enabled": sorted(self.enabled_layers),
                           "decisions": [{"layer": item.layer, "state": item.state, "reason": item.reason}
                                         for item in self.layer_decisions]},
                "processing": {**timings, "total_ms": (finished - start) * 1000,
                               "capture_to_output_ms": (finished - (acquired_at or start)) * 1000}}
