"""Independent consumer checks for the provisional observation contract."""
import json
from pathlib import Path


def test_observation_contract_keeps_identity_and_association_disabled():
    sample = json.loads((Path(__file__).parents[1] / "contracts" / "observation.example.json").read_text(encoding="utf-8"))
    assert sample["schema_version"] == "1.1.0-draft"
    assert sample["detection_trace"]["raw_emitted_counts"]["person"] == 1
    for detection in sample["detections"]:
        assert detection["verified_identity"] is None
        assert "track" not in detection
