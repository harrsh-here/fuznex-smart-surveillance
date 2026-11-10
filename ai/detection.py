"""SCRFD-500M-KPS adapter; model files are supplied by deployment configuration."""
from __future__ import annotations
from pathlib import Path
import numpy as np

class SCRFD500MDetector:
    def __init__(self, model_path: str | Path, threshold: float=0.55):
        from ai.scrfd_reference import SCRFD
        self.threshold=threshold; self.backend=SCRFD(str(model_path)); self.backend.prepare(-1,input_size=(640,640))
    def detect(self, frame: np.ndarray) -> list[tuple[np.ndarray,np.ndarray]]:
        boxes, landmarks=self.backend.detect(frame,self.threshold,input_size=(640,640))
        if boxes is None or landmarks is None: return []
        out=[]
        for box,kps in zip(boxes,landmarks):
            box=np.asarray(box,dtype=np.float32); kps=np.asarray(kps,dtype=np.float32)
            if box.shape[0]>=5 and box[2]>box[0] and box[3]>box[1] and kps.shape==(5,2): out.append((box,kps))
        return out
