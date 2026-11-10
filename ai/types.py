"""Backend-neutral AI result contracts."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any

@dataclass(frozen=True)
class FaceResult:
    bbox_xyxy: tuple[float, float, float, float]
    bbox_xywh: tuple[float, float, float, float]
    track_id: int
    detection_score: float
    identity_id: str | None
    state: str  # known | unknown | uncertain | unevaluable
    match_score: float | None  # cosine similarity, never a probability
    score_age_ms: int | None
    threshold_source: str | None
    quality_reason: str | None
    def to_dict(self) -> dict[str, Any]: return asdict(self)

@dataclass(frozen=True)
class FrameResult:
    model_version: str
    faces: tuple[FaceResult, ...]
    def to_dict(self) -> dict[str, Any]: return {"model_version": self.model_version, "faces": [f.to_dict() for f in self.faces]}
