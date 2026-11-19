"""Safe, headless-friendly combined frame contract.

Face and object detections share a frame only. This module never associates a
face with a person box and never publishes an identity candidate.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
import time
from typing import Any


SCHEMA_VERSION = "1.0.0"


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
    def __init__(self, face_pipeline: Any, object_detector: Any, clock=time.perf_counter) -> None:
        self.face_pipeline = face_pipeline
        self.object_detector = object_detector
        self.clock = clock

    def process_frame(self, frame: Any, camera_id: str, stream_epoch: str, frame_seq: int,
                      captured_at: str | None = None, acquired_at: float | None = None) -> dict[str, Any]:
        if not camera_id or not stream_epoch or frame_seq < 0:
            raise ValueError("camera_id, stream_epoch, and non-negative frame_seq are required")
        height, width = frame.shape[:2]
        start = self.clock()
        detections: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        timings: dict[str, float] = {}
        try:
            branch_start = self.clock()
            for face in self.face_pipeline.process_frame(frame).faces:
                detections.append({
                    "kind": "face", "bbox_xyxy": _box(face.bbox_xyxy, width, height),
                    "detection_confidence": float(face.detection_score), "track": None,
                    "verified_identity": None, "identity_state": "identity_disabled",
                    "quality": {"state": face.state, "reason": face.quality_reason,
                                "measurement_age_ms": face.score_age_ms},
                    "uncertainty_reasons": ["IDENTITY_ASSOCIATION_DISABLED"],
                })
            timings["face_ms"] = (self.clock() - branch_start) * 1000
        except Exception:
            errors.append({"stage": "face", "code": "INFERENCE_FAILED", "retryable": True})
            timings["face_ms"] = (self.clock() - start) * 1000
        try:
            branch_start = self.clock()
            for item in self.object_detector.detect(frame):
                kind = item["kind"]
                if kind not in {"person", "laptop"}:
                    raise ValueError("object detector returned unsupported class")
                detections.append({"kind": kind, "bbox_xyxy": _box(item["bbox_xyxy"], width, height),
                                   "detection_confidence": float(item["detection_confidence"]), "track": None,
                                   "verified_identity": None, "identity_state": "not_applicable",
                                   "quality": None, "uncertainty_reasons": []})
            timings["object_ms"] = (self.clock() - branch_start) * 1000
        except Exception:
            errors.append({"stage": "object", "code": "INFERENCE_FAILED", "retryable": True})
            timings["object_ms"] = (self.clock() - start) * 1000
        for index, detection in enumerate(detections):
            detection["detection_id"] = f"d{index}"
        finished = self.clock()
        return {"schema_version": SCHEMA_VERSION, "camera_id": camera_id, "stream_epoch": stream_epoch,
                "frame_seq": frame_seq, "captured_at": captured_at or utc_now(),
                "timestamp_source": "capture_host", "frame_size": {"width": width, "height": height},
                "models": {"face": "scrfd-500m+sface", "object": "yolo11n"}, "detections": detections,
                "events": [], "processing_errors": errors,
                "processing": {**timings, "total_ms": (finished - start) * 1000,
                               "capture_to_output_ms": (finished - (acquired_at or start)) * 1000}}
