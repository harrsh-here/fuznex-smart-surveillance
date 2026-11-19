import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ai.local_gallery import (delete_participant, new_gallery, parse_gallery,
                              templates_from_gallery, update_participant,
                              write_gallery_safely)


def test_private_gallery_updates_and_normalises_without_legacy_conversion():
    gallery = new_gallery("b" * 64, 2)
    updated = update_participant(gallery, "participant_2", np.array([3, 4]), 3, {"min_blur": 40})
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "private-gallery.json"
        write_gallery_safely(path, updated)
        templates = templates_from_gallery(path, "b" * 64)
    assert set(templates) == {"participant_2"}
    assert np.isclose(np.linalg.norm(templates["participant_2"]), 1.0)


def test_private_gallery_rejects_model_mismatch_and_duplicates():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "private-gallery.json"
        malformed = new_gallery("c" * 64, 2)
        malformed["participants"] = [{"participant_id":"participant_1", "embedding":[1, 0], "sample_count":3},
                                     {"participant_id":"participant_1", "embedding":[1, 0], "sample_count":3}]
        path.write_text(json.dumps(malformed), encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate"):
            parse_gallery(path)
        with pytest.raises(ValueError, match="does not match"):
            parse_gallery(path, "d" * 64)


def test_update_creates_backup_and_explicit_deletion_removes_only_requested_participant():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "private-gallery.json"
        gallery = update_participant(new_gallery("e" * 64, 2), "participant_1", np.array([1, 0]), 3, {})
        write_gallery_safely(path, gallery)
        changed = update_participant(gallery, "participant_3", np.array([0, 1]), 3, {})
        backup = write_gallery_safely(path, changed)
        assert backup is not None and backup.exists()
        write_gallery_safely(path, delete_participant(changed, "participant_1"))
        assert set(templates_from_gallery(path)) == {"participant_3"}
