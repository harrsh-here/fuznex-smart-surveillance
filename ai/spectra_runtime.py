"""Authoritative SPECTRA execution planning and real-pipeline construction.

Both the desktop console and the headless runner use this module.  It contains
no camera or UI code and never substitutes a disabled model with a fake result.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, FrozenSet

from ai.combined import CombinedPipeline
from ai.config import SFaceConfig
from ai.detection import SCRFD500MDetector
from ai.local_gallery import parse_gallery, sha256_file, templates_from_gallery
from ai.object_detection import YoloConfig, YoloPersonLaptopDetector
from ai.pipeline import SFacePipeline
from ai.recognition import SFaceRecognizer


LAYER_NAMES = ("capture", "yolo", "scrfd", "sface", "consensus", "tracking")
RUNNABLE_LAYERS = frozenset({"yolo", "scrfd", "sface"})


@dataclass(frozen=True)
class LayerDecision:
    layer: str
    state: str  # enabled, disabled, blocked, future
    reason: str


@dataclass(frozen=True)
class ExecutionPlan:
    mode: str  # integrated or layer
    enabled: FrozenSet[str]
    decisions: tuple[LayerDecision, ...]

    @property
    def blocked(self) -> bool:
        return any(item.state == "blocked" for item in self.decisions)

    @property
    def active_layers(self) -> tuple[str, ...]:
        return tuple(name for name in LAYER_NAMES if name in self.enabled)

    def explanation(self) -> str:
        items = [f"{item.layer}: {item.reason}" for item in self.decisions if item.state in {"blocked", "future"}]
        return " · ".join(items) if items else "All selected layers are ready."


def _available(availability: dict[str, str], layer: str) -> bool:
    return availability.get(layer) == "available"


def make_plan(mode: str, requested: set[str] | FrozenSet[str] | None, availability: dict[str, str],
              gallery_ready: bool, private_recognition: bool = False) -> ExecutionPlan:
    """Resolve dependencies before any model is loaded.

    Integrated mode is locked to capture + YOLO + SCRFD. SFace is added only
    when a local operator explicitly enables private recognition and a valid
    gallery is available. Layer mode accepts real isolated/custom combinations.
    """
    if mode not in {"integrated", "layer"}:
        raise ValueError("mode must be integrated or layer")
    if mode == "integrated":
        selected = {"capture", "yolo", "scrfd"}
        if private_recognition:
            selected.add("sface")
    else:
        selected = {"capture"} | set(requested or set())
        unknown = selected - set(LAYER_NAMES)
        if unknown:
            raise ValueError("unknown layer selection: " + ", ".join(sorted(unknown)))

    decisions: list[LayerDecision] = [LayerDecision("capture", "enabled", "Camera input is required for every live test.")]
    for layer in ("yolo", "scrfd"):
        if layer not in selected:
            decisions.append(LayerDecision(layer, "disabled", "Not selected for this layer test."))
        elif _available(availability, layer):
            decisions.append(LayerDecision(layer, "enabled", "Validated local model is available."))
        else:
            decisions.append(LayerDecision(layer, "blocked", "Required local model is unavailable."))

    if "sface" not in selected:
        reason = "Private recognition requires explicit local opt-in." if mode == "integrated" else "Not selected for this layer test."
        decisions.append(LayerDecision("sface", "disabled", reason))
    elif "scrfd" not in selected or not _available(availability, "scrfd"):
        decisions.append(LayerDecision("sface", "blocked", "SFace needs the SCRFD face-detection layer."))
    elif not _available(availability, "sface"):
        decisions.append(LayerDecision("sface", "blocked", "The SFace recognition model is unavailable."))
    elif not gallery_ready:
        decisions.append(LayerDecision("sface", "blocked", "Private recognition needs a valid local consented gallery."))
    else:
        decisions.append(LayerDecision("sface", "enabled", "Private recognition is enabled locally for this test."))

    for future in ("consensus", "tracking"):
        decisions.append(LayerDecision(future, "future", "Reserved extension point; not implemented in this build."))
    enabled = frozenset(item.layer for item in decisions if item.state == "enabled")
    return ExecutionPlan(mode, enabled, tuple(decisions))


class DetectionOnlyFacePipeline:
    """A real SCRFD adapter for tests where SFace is not enabled."""
    def __init__(self, model_path: Path) -> None:
        self.detector = SCRFD500MDetector(model_path)
        self.last_timings: dict[str, float] = {"scrfd_ms": 0.0, "sface_ms": 0.0}

    def reset_tracking(self) -> None:
        return None

    def process_frame(self, frame: Any) -> Any:
        started = time.perf_counter()
        rows = [type("Face", (), {"bbox_xyxy": box[:4], "detection_score": float(box[4]),
                                    "state": "detection_only", "quality_reason": None,
                                    "score_age_ms": 0})()
                for box, _landmarks in self.detector.detect(frame)]
        self.last_timings = {"scrfd_ms": (time.perf_counter() - started) * 1000, "sface_ms": 0.0}
        return type("Faces", (), {"faces": tuple(rows)})()


def gallery_is_ready(gallery_path: Path, sface_path: str | Path | None) -> bool:
    if not sface_path or not gallery_path.is_file():
        return False
    try:
        return bool(parse_gallery(gallery_path, sha256_file(Path(sface_path))).get("participants"))
    except (OSError, ValueError):
        return False


def build_pipeline(plan: ExecutionPlan, config: dict[str, Any], gallery_path: Path | None = None) -> CombinedPipeline:
    """Build only layers enabled by a non-blocked plan from the production components."""
    if plan.blocked:
        raise ValueError("Test is blocked: " + plan.explanation())
    yolo: Any | None = None
    faces: Any | None = None
    if "yolo" in plan.enabled:
        yolo = YoloPersonLaptopDetector(YoloConfig(model_path=Path(config["yolo"])))
    if "sface" in plan.enabled:
        if gallery_path is None:
            raise ValueError("SFace needs a local gallery path")
        sface_path = Path(config["sface"])
        face_config = SFaceConfig(Path(config["scrfd"]), sface_path)
        templates = templates_from_gallery(gallery_path, sha256_file(sface_path))
        faces = SFacePipeline(face_config, SCRFD500MDetector(face_config.detector_model),
                              SFaceRecognizer(face_config.recognizer_model), templates)
    elif "scrfd" in plan.enabled:
        faces = DetectionOnlyFacePipeline(Path(config["scrfd"]))
    return CombinedPipeline(faces, yolo, enabled_layers=plan.enabled, layer_decisions=plan.decisions)
