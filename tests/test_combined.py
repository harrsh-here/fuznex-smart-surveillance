import unittest
import numpy as np
from types import SimpleNamespace

from ai.combined import CombinedPipeline
from ai.object_detection import YoloConfig


class Face:
    def process_frame(self, _frame):
        return SimpleNamespace(faces=(SimpleNamespace(bbox_xyxy=(1, 2, 20, 30), detection_score=.9,
            state="known", quality_reason=None, score_age_ms=0, identity_id="must-not-leak"),))


class Objects:
    def detect(self, _frame):
        return [{"kind": "person", "bbox_xyxy": (0, 0, 50, 60), "detection_confidence": .8}]


class PhoneObjects:
    def detect(self, _frame):
        return [{"kind": "phone", "bbox_xyxy": (5, 6, 25, 36), "detection_confidence": .8}]


class Broken:
    def detect(self, _frame):
        raise RuntimeError("test failure")


class CombinedTests(unittest.TestCase):
    def test_config_rejects_unsupported_values(self):
        with self.assertRaises(ValueError): YoloConfig("model.pt", input_size=31)
        with self.assertRaises(ValueError): YoloConfig("model.pt", device="metal")

    def test_identity_is_never_published_or_linked(self):
        pipeline = CombinedPipeline(Face(), Objects())
        result = pipeline.process_frame(np.zeros((80, 100, 3)), "cam", "epoch", 0)
        face = result["detections"][0]
        self.assertIsNone(face["verified_identity"])
        self.assertIsNone(face["track"])
        self.assertEqual(face["identity_state"], "identity_disabled")
        self.assertIn("IDENTITY_ASSOCIATION_DISABLED", face["uncertainty_reasons"])
        self.assertEqual(face["quality"]["pixel_width"], 19.0)
        self.assertEqual(face["quality"]["landmarks"], "not_published_by_combined_contract")
        self.assertEqual(len(pipeline.last_private_faces), 1)
        self.assertNotIn("must-not-leak", repr(result))

    def test_partial_failure_is_sanitized(self):
        result = CombinedPipeline(Face(), Broken()).process_frame(np.zeros((80, 100, 3)), "cam", "epoch", 0)
        self.assertEqual(result["processing_errors"][0]["stage"], "object")
        self.assertEqual(len(result["detections"]), 1)

    def test_phone_uses_the_coco_cell_phone_class_id(self):
        result = CombinedPipeline(Face(), PhoneObjects()).process_frame(np.zeros((80, 100, 3)), "cam", "epoch", 0)
        phone = next(item for item in result["detections"] if item["kind"] == "phone")
        self.assertEqual(phone["class_id"], 67)
        self.assertEqual(result["detection_trace"]["raw_emitted_counts"]["phone"], 1)


if __name__ == "__main__":
    unittest.main()
