import unittest

from ai.live_camera import face_label, source_value


class Face:
    def __init__(self, state, identity_id=None, quality_reason=None):
        self.state = state
        self.identity_id = identity_id
        self.quality_reason = quality_reason


class LiveCameraTests(unittest.TestCase):
    def test_source_accepts_webcam_index_or_url(self):
        self.assertEqual(source_value("0"), 0)
        self.assertEqual(source_value("rtsp://camera/live"), "rtsp://camera/live")

    def test_only_known_state_receives_a_private_name(self):
        self.assertEqual(face_label(Face("known", "local-1"), {"local-1": "Ada"}), "Known: Ada")
        self.assertEqual(face_label(Face("unknown", "local-1"), {"local-1": "Ada"}), "Unknown")
        self.assertEqual(face_label(Face("uncertain", "local-1"), {"local-1": "Ada"}), "Uncertain")
        self.assertEqual(face_label(Face("unevaluable", quality_reason="face too small"), {}), "Unevaluable: face too small")
