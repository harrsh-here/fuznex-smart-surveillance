import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from ai.runtime_discovery import discover_models


class RuntimeDiscoveryTests(unittest.TestCase):
    def test_discovers_project_models_and_rejects_bad_known_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "ai").mkdir(); (root / "models").mkdir()
            good = b"verified-scrfd"; (root / "models" / "det_500m.onnx").write_bytes(good)
            (root / "models" / "yolo11n.pt").write_bytes(b"local-yolo")
            manifest = {"models": [{"id": "scrfd-500m-kps", "sha256": hashlib.sha256(good).hexdigest()}, {"id": "sface-2021dec", "sha256": "f" * 64}, {"id": "yolo11n-coco", "sha256": None}]}
            (root / "ai" / "model-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            result = discover_models(root, {})
            self.assertEqual(result.availability, {"yolo": "available", "scrfd": "available", "sface": "not_found"})
            self.assertTrue(result.config["yolo"].endswith("yolo11n.pt"))
            self.assertTrue(result.config["scrfd"].endswith("det_500m.onnx"))

    def test_stale_saved_path_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "ai").mkdir(); (root / "ai" / "model-manifest.json").write_text('{"models": []}', encoding="utf-8")
            # Discovery normally also checks the user's Downloads folder.  The
            # test must not depend on unrelated local model artifacts there.
            with patch("ai.runtime_discovery._roots", return_value=[]):
                result = discover_models(root, {"yolo": str(root / "missing.pt"), "hashes": {"yolo": "x"}})
            self.assertEqual(result.availability["yolo"], "not_found")
            self.assertNotIn("yolo", result.config)


if __name__ == "__main__":
    unittest.main()
