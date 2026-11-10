"""Reusable frame pipeline; no database, camera, alert, or UI coupling."""
from __future__ import annotations
import time
import numpy as np
from ai.config import SFaceConfig
from ai.recognition import SFACE_REFERENCE_SOURCE, decide
from ai.tracking import MultiFaceTracker, Track
from ai.types import FaceResult, FrameResult

class SFacePipeline:
    def __init__(self, config:SFaceConfig, detector, recognizer, templates:dict[str,np.ndarray]|None=None, clock=time.monotonic):
        self.config=config; self.detector=detector; self.recognizer=recognizer; self.templates=templates or {}; self.clock=clock; self.frame_no=0
        self.tracker=MultiFaceTracker(config.association_iou,config.track_ttl_updates,ambiguity_margin=config.association_ambiguity_margin)
    def set_templates(self, templates:dict[str,np.ndarray]) -> None: self.templates={k:np.asarray(v,dtype=np.float32) for k,v in templates.items()}
    def process_frame(self, frame:np.ndarray) -> FrameResult:
        self.frame_no+=1; detections=self.detector.detect(frame); visible=self.tracker.update(detections,self.frame_no); now=self.clock()
        known:dict[str,Track]={}
        for track in visible:
            due=track.last_measurement is None or now-track.last_measurement>=self.config.recognition_refresh_seconds
            if due:
                self._measure(track,frame,now)
            if track.state=="known" and track.identity_id:
                prior=known.get(track.identity_id)
                if prior is None or (track.score or -2)>(prior.score or -2):
                    if prior is not None:
                        prior.state="uncertain"; prior.identity_id=None; prior.quality_reason="duplicate identity claim"
                    known[track.identity_id]=track
                else:
                    track.state="uncertain"; track.identity_id=None; track.quality_reason="duplicate identity claim"
        return FrameResult(self.config.model_version,tuple(self._result(t,now) for t in visible))
    def _measure(self,t:Track,frame:np.ndarray,now:float)->None:
        w,h=t.box[2]-t.box[0],t.box[3]-t.box[1]
        if w<self.config.min_face_size_px or h<self.config.min_face_size_px:
            t.state="unevaluable"; t.quality_reason="face too small"; return
        try: embedding,crop=self.recognizer.embed(frame,t.box,t.landmarks)
        except Exception as exc: t.state="unevaluable"; t.quality_reason=f"alignment failed: {exc}"; return
        try:
            import cv2
            if cv2.Laplacian(crop,cv2.CV_64F).var()<40: t.state="unevaluable"; t.quality_reason="face crop too blurry"; return
        except ImportError: pass
        t.embedding=embedding; t.last_measurement=now; t.state,t.identity_id,t.score=decide(embedding,self.templates,self.config.cosine_threshold,self.config.uncertain_band); t.quality_reason=None
    def _result(self,t:Track,now:float)->FaceResult:
        b=t.display_box; age=None if t.last_measurement is None else int(max(0,now-t.last_measurement)*1000)
        return FaceResult((float(b[0]),float(b[1]),float(b[2]),float(b[3])),(float(b[0]),float(b[1]),float(b[2]-b[0]),float(b[3]-b[1])),t.track_id,float(t.box[4]),t.identity_id,t.state,t.score,age,SFACE_REFERENCE_SOURCE if t.score is not None else None,t.quality_reason)

RecognitionSystem=SFacePipeline
