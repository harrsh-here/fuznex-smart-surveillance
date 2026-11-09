import unittest
from ai.pipeline import RecognitionSystem

class TestMockContract(unittest.TestCase):
    def test_mock_gallery(self):
        gallery = RecognitionSystem().load_templates()
        self.assertEqual(len(gallery), 2)
        self.assertTrue(all(t["active"] for t in gallery))
        self.assertTrue(all("person_id" in t and "model_version" in t for t in gallery))

    def test_no_fake_recognition(self):
        with self.assertRaises(NotImplementedError):
            RecognitionSystem().recognize_frame(None, "CAM01", "2026-11-07T15:58:00+05:30")

if __name__ == "__main__":
    unittest.main()
