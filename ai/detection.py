"""SCRFD-500M-KPS adapter; model files are supplied by deployment configuration."""
from __future__ import annotations
from pathlib import Path
import numpy as np

class SCRFD500MDetector:
    def __init__(self, model_path: str | Path, threshold: float=0.55, input_size: tuple[int,int]=(640,640)):
        from ai.scrfd_reference import SCRFD
        if (not isinstance(input_size, tuple) or len(input_size)!=2
                or any(isinstance(value, bool) or not isinstance(value, int) or value<=0 for value in input_size)):
            raise ValueError("input_size must be a tuple of two positive integers")
        self.threshold=threshold; self.input_size=tuple(input_size); self.backend=SCRFD(str(model_path)); self.backend.prepare(-1,input_size=self.input_size)
    def detect(self, frame: np.ndarray) -> list[tuple[np.ndarray,np.ndarray]]:
        boxes, landmarks=self.backend.detect(frame,self.threshold,input_size=self.input_size)
        if boxes is None or landmarks is None: return []
        out=[]
        for box,kps in zip(boxes,landmarks):
            box=np.asarray(box,dtype=np.float32); kps=np.asarray(kps,dtype=np.float32)
            if box.shape[0]>=5 and box[2]>box[0] and box[3]>box[1] and kps.shape==(5,2): out.append((box,kps))
        return out
