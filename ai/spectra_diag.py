"""Privacy-preserving SPECTRA DIAG run accounting and headless replay.

This module deliberately records only manually supplied, consented participant
codes (participant_1..participant_3), never recognition candidates or face to
person associations.  Frames are never written unless ``store_frames`` is set.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time
import uuid
from typing import Any, Iterable

VALID_BANDS = {"near", "middle", "far"}
VALID_PARTICIPANTS = {"participant_1", "participant_2", "participant_3"}
VALID_KINDS = {"face", "person", "laptop"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class GroundTruth:
    """A manual label; absent labels always make a decision unevaluable."""
    present: frozenset[str] = frozenset()
    distance_band: str | None = None
    participant: str | None = None
    consented: bool = False

    def __post_init__(self) -> None:
        if not self.present <= VALID_KINDS:
            raise ValueError("ground truth only supports face, person, laptop")
        if self.distance_band is not None and self.distance_band not in VALID_BANDS:
            raise ValueError("distance band must be near, middle, or far")
        if self.participant is not None and self.participant not in VALID_PARTICIPANTS:
            raise ValueError("participant must be participant_1, participant_2, or participant_3")
        if self.participant and not self.consented:
            raise ValueError("participant labels require explicit consent")


def decision(kind: str, detected: bool, truth: GroundTruth | None) -> str:
    if truth is None:
        return "unevaluable"
    expected = kind in truth.present
    return "true_positive" if detected and expected else "false_positive" if detected else "false_negative" if expected else "true_negative"


@dataclass
class RunMetrics:
    frames: int = 0
    dropped_frames: int = 0
    errors: int = 0
    reconnects: int = 0
    capture_ms: list[float] = field(default_factory=list)
    inference_ms: list[float] = field(default_factory=list)
    end_to_end_ms: list[float] = field(default_factory=list)
    face_ms: list[float] = field(default_factory=list)
    object_ms: list[float] = field(default_factory=list)
    serialization_ms: list[float] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        def avg(values: list[float]) -> float | None: return round(sum(values) / len(values), 2) if values else None
        latency = avg(self.end_to_end_ms)
        return {"frames": self.frames, "dropped_frames": self.dropped_frames, "errors": self.errors,
                "reconnects": self.reconnects, "capture_ms_avg": avg(self.capture_ms),
                "inference_ms_avg": avg(self.inference_ms), "end_to_end_ms_avg": latency,
                "face_ms_avg": avg(self.face_ms), "object_ms_avg": avg(self.object_ms), "serialization_ms_avg": avg(self.serialization_ms),
                "fps": round(1000 / latency, 2) if latency and latency > 0 else None,
                "platform": platform.platform()}


class DiagnosticExport:
    """Append-only per-run JSONL/CSV export; never overwrites an older run."""
    def __init__(self, output_dir: str | Path, provenance: dict[str, Any], store_frames: bool = False) -> None:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        self.directory = Path(output_dir) / f"spectra-{stamp}-{uuid.uuid4().hex[:8]}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.store_frames = store_frames
        self.events = (self.directory / "events.jsonl").open("x", encoding="utf-8")
        self.csv_file = (self.directory / "detections.csv").open("x", newline="", encoding="utf-8")
        self.csv = csv.DictWriter(self.csv_file, fieldnames=["timestamp", "frame_seq", "kind", "class_id", "confidence", "decision", "distance_band", "participant"])
        self.csv.writeheader(); self.csv_file.flush()
        (self.directory / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")

    def record(self, result: dict[str, Any], metrics: RunMetrics, truth: GroundTruth | None = None, event: str = "frame") -> float:
        started = time.perf_counter()
        seen = {d["kind"] for d in result.get("detections", [])}
        item = {"event": event, "at": utc_now(), "result": result, "metrics": metrics.snapshot(),
                "ground_truth": None if truth is None else {"present": sorted(truth.present), "distance_band": truth.distance_band,
                "participant": truth.participant if truth.consented else None}}
        self.events.write(json.dumps(item, separators=(",", ":")) + "\n"); self.events.flush()
        for kind in VALID_KINDS:
            matching = [d for d in result.get("detections", []) if d["kind"] == kind] or [{}]
            for detected in matching:
                self.csv.writerow({"timestamp": result.get("captured_at"), "frame_seq": result.get("frame_seq"), "kind": kind,
                    "class_id": detected.get("class_id"), "confidence": detected.get("detection_confidence"),
                    "decision": decision(kind, kind in seen, truth), "distance_band": truth.distance_band if truth else None,
                    "participant": truth.participant if truth and truth.consented else None})
        self.csv_file.flush()
        elapsed = (time.perf_counter() - started) * 1000
        metrics.serialization_ms.append(elapsed)
        return elapsed

    def close(self) -> None:
        self.events.close(); self.csv_file.close()


def replay_frames(pipeline: Any, frames: Iterable[Any], camera_id: str, truth: GroundTruth | None = None,
                  exporter: DiagnosticExport | None = None) -> tuple[list[dict[str, Any]], RunMetrics]:
    """Run the real combined pipeline on supplied frames, suitable for CI/Kaggle."""
    epoch = uuid.uuid4().hex; metrics = RunMetrics(); results = []
    for seq, frame in enumerate(frames):
        acquired = time.perf_counter()
        result = pipeline.process_frame(frame, camera_id, epoch, seq, acquired_at=acquired)
        metrics.frames += 1; metrics.inference_ms.append(result["processing"]["total_ms"])
        metrics.face_ms.append(result["processing"].get("face_ms", 0.0)); metrics.object_ms.append(result["processing"].get("object_ms", 0.0))
        metrics.end_to_end_ms.append(result["processing"]["capture_to_output_ms"])
        metrics.errors += len(result["processing_errors"]); results.append(result)
        if exporter: exporter.record(result, metrics, truth)
    return results, metrics
