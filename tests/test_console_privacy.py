"""Regression checks for the consolidated local console's storage guarantees."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ai.local_gallery import new_gallery, parse_gallery, update_participant, write_gallery_safely
from ai.spectra_diag import DiagnosticExport, RunMetrics


class ConsolePrivacyTests(unittest.TestCase):
    def test_fourth_and_fifth_people_are_not_capped(self):
        gallery = new_gallery("a" * 64, 2)
        for number in range(1, 6):
            gallery = update_participant(gallery, f"local-{number}", np.array([number, 1.0]), 5, {}, f"Person {number}")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "private-gallery.json"; write_gallery_safely(path, gallery)
            restored = parse_gallery(path)
        self.assertEqual(len(restored["participants"]), 5)
        self.assertEqual(restored["participants"][4]["display_name"], "Person 5")

    def test_standard_export_never_contains_private_gallery_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            export = DiagnosticExport(temp, {"privacy": {"identity_publication": "disabled"}})
            export.record({"captured_at": "now", "frame_seq": 0, "detections": []}, RunMetrics())
            export.close()
            contents = (export.directory / "events.jsonl").read_text(encoding="utf-8")
        self.assertNotIn("Person 1", contents)
        self.assertNotIn("participant_id", contents)
        self.assertEqual(json.loads(contents)["ground_truth"], None)


if __name__ == "__main__":
    unittest.main()
