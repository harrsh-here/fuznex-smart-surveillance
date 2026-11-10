"""Small multi-face IoU tracker. It stores no images or persistent identities."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1,y1=max(a[0],b[0]),max(a[1],b[1]); x2,y2=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0.,x2-x1)*max(0.,y2-y1); area=lambda z:max(0.,z[2]-z[0])*max(0.,z[3]-z[1])
    union=area(a)+area(b)-inter
    return float(inter/union) if union else 0.

@dataclass
class Track:
    track_id: int; box: np.ndarray; landmarks: np.ndarray; display_box: np.ndarray; last_seen_frame: int
    missed: int = 0; embedding: np.ndarray | None = None; last_measurement: float | None = None
    state: str = "unevaluable"; identity_id: str | None = None; score: float | None = None; quality_reason: str | None = "awaiting measurement"

class MultiFaceTracker:
    def __init__(self, iou_threshold: float=0.2, ttl_updates: int=2, smoothing: float=0.35):
        self.iou_threshold=iou_threshold; self.ttl_updates=ttl_updates; self.smoothing=smoothing; self._next=1; self.tracks: dict[int,Track]={}
    def update(self, detections: list[tuple[np.ndarray,np.ndarray]], frame_no: int) -> list[Track]:
        pairs=sorted(((iou(t.box,b),tid,idx) for tid,t in self.tracks.items() for idx,(b,_) in enumerate(detections) if iou(t.box,b)>=self.iou_threshold), reverse=True)
        used_t:set[int]=set(); used_d:set[int]=set()
        for _,tid,idx in pairs:
            if tid in used_t or idx in used_d: continue
            t=self.tracks[tid]; b,k=detections[idx]; was_missed=t.missed>0
            t.display_box=self.smoothing*t.display_box+(1-self.smoothing)*b; t.display_box[4]=b[4]; t.box=b; t.landmarks=k; t.last_seen_frame=frame_no; t.missed=0
            if was_missed: t.last_measurement=None
            used_t.add(tid); used_d.add(idx)
        for idx,(b,k) in enumerate(detections):
            if idx not in used_d:
                self.tracks[self._next]=Track(self._next,b,k,b.copy(),frame_no); self._next+=1
        for tid,t in list(self.tracks.items()):
            if tid not in used_t and t.last_seen_frame!=frame_no:
                t.missed+=1
                if t.missed>self.ttl_updates: del self.tracks[tid]
        return self.visible()
    def visible(self) -> list[Track]: return [self.tracks[k] for k in sorted(self.tracks) if self.tracks[k].missed==0]
