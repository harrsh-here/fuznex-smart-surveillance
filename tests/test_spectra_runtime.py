import unittest

import numpy as np

from ai.combined import CombinedPipeline
from ai.spectra_runtime import make_plan


class Face:
    def __init__(self): self.calls = 0
    def process_frame(self, _frame):
        self.calls += 1
        return type("Result", (), {"faces": ()})()


class Objects:
    def __init__(self): self.calls = 0
    def detect(self, _frame):
        self.calls += 1
        return []


class RuntimePlanTests(unittest.TestCase):
    def test_integrated_requires_yolo_and_scrfd(self):
        plan = make_plan("integrated", None, {"yolo": "available", "scrfd": "not_found", "sface": "available"}, False)
        self.assertTrue(plan.blocked)
        self.assertIn("scrfd", plan.explanation())

    def test_integrated_enables_required_configured_layers(self):
        plan = make_plan("integrated", None, {"yolo": "available", "scrfd": "available", "sface": "available"}, False)
        self.assertFalse(plan.blocked)
        self.assertEqual(set(plan.active_layers), {"capture", "yolo", "scrfd"})

    def test_sface_dependency_is_enforced(self):
        plan = make_plan("layer", {"sface"}, {"yolo": "available", "scrfd": "available", "sface": "available"}, True, True)
        self.assertTrue(plan.blocked)
        self.assertIn("SFace needs the SCRFD", plan.explanation())

    def test_isolated_yolo_skips_face_inference(self):
        face, objects = Face(), Objects()
        pipeline = CombinedPipeline(face, objects, enabled_layers={"capture", "yolo"})
        result = pipeline.process_frame(np.zeros((20, 20, 3), dtype=np.uint8), "camera", "epoch", 0)
        self.assertEqual(objects.calls, 1)
        self.assertEqual(face.calls, 0)
        self.assertEqual(result["processing"]["scrfd_ms"], 0.0)

    def test_isolated_scrfd_skips_object_inference(self):
        face, objects = Face(), Objects()
        pipeline = CombinedPipeline(face, objects, enabled_layers={"capture", "scrfd"})
        result = pipeline.process_frame(np.zeros((20, 20, 3), dtype=np.uint8), "camera", "epoch", 0)
        self.assertEqual(objects.calls, 0)
        self.assertEqual(face.calls, 1)
        self.assertEqual(result["processing"]["yolo_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
