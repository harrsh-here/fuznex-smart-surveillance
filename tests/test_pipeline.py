import unittest
import numpy as np
from ai.config import SFaceConfig
from ai.detection import SCRFD500MDetector
from ai.tracking import MultiFaceTracker
from ai.pipeline import SFacePipeline
from ai.recognition import decide

class Detector:
    def detect(self, frame): return [(np.array([0,0,80,80,.9],dtype=np.float32),np.array([[10,10],[40,10],[25,30],[12,55],[38,55]],dtype=np.float32))]
class Recognizer:
    def __init__(self): self.calls=0
    def embed(self, frame, box, landmarks):
        self.calls+=1
        crop=np.indices((20,20)).sum(axis=0).astype(np.uint8)%2*255
        return np.array([1.,0.],dtype=np.float32),np.repeat(crop[...,None],3,axis=2)

class PipelineTests(unittest.TestCase):
    def test_detector_input_size_is_explicit_and_defaults_to_live_setting(self):
        cfg=SFaceConfig(detector_model="d",recognizer_model="s")
        self.assertEqual(cfg.detector_input_size,(640,640))
        self.assertEqual(SFaceConfig(detector_model="d",recognizer_model="s",detector_input_size=(320,320)).detector_input_size,(320,320))
    def test_detector_rejects_invalid_input_sizes_before_model_load(self):
        for value in (None,(0,320),(-1,320),(320,),[320,320],("320",320)):
            with self.assertRaises(ValueError): SCRFD500MDetector("not-loaded.onnx",input_size=value)
    def test_ambiguous_association_starts_fresh_tracks(self):
        tracker=MultiFaceTracker(iou_threshold=.1,ambiguity_margin=.2)
        k=np.zeros((5,2),dtype=np.float32)
        tracker.update([(np.array([0,0,10,10,.9],dtype=np.float32),k),(np.array([8,0,18,10,.9],dtype=np.float32),k)],1)
        visible=tracker.update([(np.array([4,0,14,10,.9],dtype=np.float32),k)],2)
        self.assertEqual([track.track_id for track in visible],[3])
    def test_process_frame_returns_backend_neutral_contract_and_refreshes(self):
        clock=[0.]
        cfg=SFaceConfig(detector_model="det.onnx",recognizer_model="sface.onnx",recognition_refresh_seconds=1.0,min_face_size_px=10)
        rec=Recognizer(); pipe=SFacePipeline(cfg,Detector(),rec,{"harsh":np.array([1.,0.],dtype=np.float32)},clock=lambda:clock[0])
        first=pipe.process_frame(np.zeros((100,100,3),dtype=np.uint8)).to_dict()
        self.assertEqual(first["faces"][0]["state"],"known"); self.assertEqual(first["faces"][0]["track_id"],1); self.assertEqual(rec.calls,1)
        clock[0]=.5; pipe.process_frame(np.zeros((100,100,3),dtype=np.uint8)); self.assertEqual(rec.calls,1)
        clock[0]=1.1; pipe.process_frame(np.zeros((100,100,3),dtype=np.uint8)); self.assertEqual(rec.calls,2)
    def test_no_template_is_unknown_not_a_probability(self):
        cfg=SFaceConfig(detector_model="d",recognizer_model="s",min_face_size_px=10); pipe=SFacePipeline(cfg,Detector(),Recognizer())
        face=pipe.process_frame(np.zeros((100,100,3),dtype=np.uint8)).faces[0]
        self.assertEqual(face.state,"unknown"); self.assertIsNone(face.match_score)
    def test_non_known_nearest_template_does_not_leak_identity(self):
        state,identity,score=decide(np.array([0.,1.]),{"person-1":np.array([1.,0.])})
        self.assertEqual(state,"unknown"); self.assertIsNone(identity); self.assertIsNotNone(score)

if __name__ == "__main__": unittest.main()
