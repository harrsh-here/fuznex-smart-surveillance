"""Configuration only; models and secrets remain outside Git."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class SFaceConfig:
    detector_model: Path
    recognizer_model: Path
    detector_threshold: float = 0.55
    detector_interval_frames: int = 5
    recognition_refresh_seconds: float = 1.0
    cosine_threshold: float = 0.363
    uncertain_band: float = 0.05
    min_face_size_px: float = 40.0
    track_ttl_updates: int = 2
    association_iou: float = 0.20
    model_version: str = "sface-2021dec-experimental"
