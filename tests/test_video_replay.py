import unittest

from ai.video_replay import _face_state


class VideoReplayTests(unittest.TestCase):
    def test_face_overlay_never_contains_an_identity(self):
        self.assertEqual(_face_state("known"), "Known")
        self.assertEqual(_face_state("unknown"), "Unknown")
        self.assertEqual(_face_state("uncertain"), "Uncertain")
        self.assertEqual(_face_state("unevaluable"), "Unevaluable")
        self.assertEqual(_face_state("detection_only"), "Face detected")
        self.assertEqual(_face_state("anything-else"), "Unevaluable")
