import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from ai.combined import CombinedPipeline
from ai.console import EmptyFacePipeline, load_local_gallery, make_face_pipeline, private_face_status
from ai.local_gallery import new_gallery, update_participant
from ai.spectra_diag import DiagnosticExport, GroundTruth, decision, list_sessions, replay_frames

class Faces:
    def process_frame(self, frame): return type("Result", (), {"faces": ()})()
class Objects:
    def detect(self, frame): return [{"kind":"laptop", "class_id":63, "bbox_xyxy":(1,2,30,40), "detection_confidence":.9}]

class SpectraTests(unittest.TestCase):
    def test_local_identity_diagnosis_is_opt_in(self):
        args = type("Args", (), {"local_diagnostic": False})()
        self.assertIsInstance(make_face_pipeline(args), EmptyFacePipeline)

    def test_gallery_is_validated_in_memory(self):
        with tempfile.TemporaryDirectory() as temp:
            gallery = Path(temp) / "gallery.json"
            payload = update_participant(new_gallery("a" * 64, 2), "participant_1", np.array([3,4]), 3, {})
            gallery.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_local_gallery(gallery)
            self.assertEqual(set(loaded), {"participant_1"})
            self.assertAlmostEqual(float(np.linalg.norm(loaded["participant_1"])), 1.0)
    def test_private_status_hides_non_known_candidates(self):
        known = type("Face", (), {"state":"known", "identity_id":"participant_1", "quality_reason":None})()
        unknown = type("Face", (), {"state":"unknown", "identity_id":None, "quality_reason":None})()
        pipeline = type("Pipeline", (), {"process_frame":lambda *_: type("Result", (), {"faces":(known, unknown)})()})()
        rows = private_face_status(pipeline, object())
        self.assertEqual(rows[0]["participant_id"], "participant_1")
        self.assertNotIn("participant_id", rows[1])
    def test_unknown_truth_is_unevaluable(self):
        self.assertEqual(decision("laptop", True, None), "unevaluable")
        self.assertEqual(decision("laptop", True, GroundTruth(frozenset({"laptop"}))), "true_positive")
    def test_participant_requires_consent(self):
        with self.assertRaises(ValueError): GroundTruth(participant="participant_1")
    def test_consented_participant_labels_are_not_capped(self):
        self.assertEqual(GroundTruth(participant="person-005", consented=True).participant, "person-005")
    def test_phone_ground_truth_is_supported(self):
        self.assertEqual(decision("phone", True, GroundTruth(frozenset({"phone"}))), "true_positive")
    def test_replay_exports_non_overwriting_trace(self):
        pipe = CombinedPipeline(Faces(), Objects())
        with tempfile.TemporaryDirectory() as temp:
            export = DiagnosticExport(temp, {"object":{"classes":{"63":"laptop"}}})
            results, metrics = replay_frames(pipe, [np.zeros((50,50,3), dtype=np.uint8)], "replay", GroundTruth(frozenset({"laptop"})), export)
            export.close()
            self.assertEqual(results[0]["detections"][0]["class_id"], 63)
            self.assertEqual(metrics.frames, 1)
            self.assertIsNotNone(metrics.snapshot()["serialization_ms_avg"])
            self.assertEqual(results[0]["detection_trace"]["raw_emitted_counts"]["laptop"], 1)
            rows = (Path(export.directory) / "detections.csv").read_text(encoding="utf-8")
            self.assertIn("true_positive", rows)
            self.assertEqual(len((Path(export.directory) / "events.jsonl").read_text().splitlines()), 1)
            export.finalize("completed", {"mode": "replay"})
            self.assertEqual(list_sessions(temp)[0]["session_id"], export.session_id)

if __name__ == "__main__": unittest.main()
