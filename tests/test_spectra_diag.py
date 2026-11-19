import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from ai.combined import CombinedPipeline
from ai.console import EmptyFacePipeline, load_local_gallery, make_face_pipeline
from ai.spectra_diag import DiagnosticExport, GroundTruth, decision, replay_frames

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
            gallery.write_text(json.dumps({"templates":[{"identity_id":"opaque-test-id", "embedding":[3,4]}]}), encoding="utf-8")
            loaded = load_local_gallery(gallery)
            self.assertEqual(set(loaded), {"opaque-test-id"})
            self.assertAlmostEqual(float(np.linalg.norm(loaded["opaque-test-id"])), 1.0)
    def test_unknown_truth_is_unevaluable(self):
        self.assertEqual(decision("laptop", True, None), "unevaluable")
        self.assertEqual(decision("laptop", True, GroundTruth(frozenset({"laptop"}))), "true_positive")
    def test_participant_requires_consent(self):
        with self.assertRaises(ValueError): GroundTruth(participant="participant_1")
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

if __name__ == "__main__": unittest.main()
