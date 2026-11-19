"""Portable YOLO person/laptop adapter.

This adapter deliberately has no camera, UI, storage, or identity dependency.
It loads only an explicitly supplied, locally verified COCO YOLO weight file.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import os
import time
from typing import Any

COCO_CLASSES = {0: "person", 63: "laptop"}


@dataclass(frozen=True)
class YoloConfig:
    model_path: Path
    input_size: int = 320
    confidence: float = 0.25
    device: str = "cpu"
    threads: int = 2

    def __post_init__(self) -> None:
        if self.input_size <= 0 or self.input_size % 32:
            raise ValueError("input_size must be a positive multiple of 32")
        if not math.isfinite(self.confidence) or not 0 < self.confidence <= 1:
            raise ValueError("confidence must be in (0, 1]")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if self.threads < 1:
            raise ValueError("threads must be positive")


class YoloPersonLaptopDetector:
    """YOLO11 COCO adapter; import is lazy so contract tests need no ML stack."""

    def __init__(self, config: YoloConfig) -> None:
        self.config = config
        path = Path(config.model_path)
        if not path.is_file():
            raise FileNotFoundError("Model unavailable: use the manifest to obtain a trusted local weight")
        os.environ.setdefault("YOLO_OFFLINE", "true")
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install ai/requirements/yolo.txt before enabling YOLO inference") from exc
        if config.device == "cpu":
            torch.set_num_threads(config.threads)
        start = time.perf_counter()
        self.model = YOLO(str(path), task="detect")
        if any(self.model.names.get(index) != name for index, name in COCO_CLASSES.items()):
            raise ValueError("Model must use the COCO person=0 and laptop=63 class mapping")
        self.load_ms = (time.perf_counter() - start) * 1000

    def detect(self, frame: Any) -> list[dict[str, Any]]:
        if frame is None or getattr(frame, "size", 0) == 0:
            raise ValueError("Empty input frame")
        result = self.model.predict(
            frame, device=self.config.device, imgsz=self.config.input_size,
            conf=self.config.confidence, classes=list(COCO_CLASSES), rect=False,
            half=False, verbose=False,
        )[0]
        detections = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            if class_id in COCO_CLASSES:
                detections.append({
                    "kind": COCO_CLASSES[class_id],
                    "bbox_xyxy": [float(value) for value in box.xyxy[0].tolist()],
                    "detection_confidence": float(box.conf.item()),
                })
        return detections
