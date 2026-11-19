"""Private, local-only SFace gallery storage.

This module deliberately has no exporter or backend dependency.  A gallery is
an ignored file chosen by the operator; it may contain biometric embeddings and
must never be copied into diagnostics, contracts, or version control.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
MODEL_ID = "sface-2021dec"
# IDs are generated locally by the diagnostic console.  This deliberately has
# no fixed population cap; legacy participant_1 style IDs remain valid.
PARTICIPANT_IDS: frozenset[str] = frozenset()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_embedding(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector)) if vector.size else 0.0
    if not np.isfinite(vector).all() or norm == 0.0:
        raise ValueError("gallery contains an invalid embedding")
    return vector / norm


def new_gallery(model_sha256: str, embedding_dimension: int) -> dict[str, Any]:
    if len(model_sha256) != 64 or any(char not in "0123456789abcdef" for char in model_sha256.lower()):
        raise ValueError("model SHA-256 must be a 64-character hexadecimal value")
    if embedding_dimension <= 0:
        raise ValueError("embedding dimension must be positive")
    return {"schema_version": SCHEMA_VERSION,
            "model": {"id": MODEL_ID, "sha256": model_sha256.lower(), "embedding_dimension": embedding_dimension},
            "participants": []}


def parse_gallery(path: Path, expected_model_sha256: str | None = None) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError("private gallery does not exist") from None
    except json.JSONDecodeError as exc:
        raise ValueError("private gallery is not valid JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported private gallery schema")
    model = payload.get("model")
    entries = payload.get("participants")
    if not isinstance(model, dict) or model.get("id") != MODEL_ID or not isinstance(entries, list):
        raise ValueError("private gallery is missing model metadata or participants")
    model_sha = model.get("sha256")
    dimension = model.get("embedding_dimension")
    if not isinstance(model_sha, str) or not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("private gallery has invalid model metadata")
    if expected_model_sha256 and model_sha.lower() != expected_model_sha256.lower():
        raise ValueError("SFace model SHA-256 does not match the private gallery")
    seen: set[str] = set()
    checked: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("private gallery contains an invalid participant record")
        participant_id = entry.get("participant_id")
        if not isinstance(participant_id, str) or not participant_id.strip() or participant_id in seen:
            raise ValueError("private gallery contains an invalid or duplicate participant ID")
        embedding = _normalise_embedding(entry.get("embedding"))
        if embedding.size != dimension:
            raise ValueError("private gallery embedding dimension does not match model metadata")
        sample_count = entry.get("sample_count")
        if not isinstance(sample_count, int) or sample_count < 3:
            raise ValueError("private gallery sample count is invalid")
        seen.add(participant_id)
        display_name = entry.get("display_name", participant_id)
        if not isinstance(display_name, str) or not display_name.strip():
            raise ValueError("private gallery has an invalid display name")
        checked.append({"participant_id": participant_id, "display_name": display_name.strip(), "embedding": embedding.tolist(),
                        "sample_count": sample_count, "enrolled_at": entry.get("enrolled_at"),
                        "quality": entry.get("quality", {})})
    return {"schema_version": SCHEMA_VERSION, "model": dict(model), "participants": checked}


def templates_from_gallery(path: Path, expected_model_sha256: str | None = None) -> dict[str, np.ndarray]:
    """Load local templates into memory.  This function never writes the gallery."""
    gallery = parse_gallery(path, expected_model_sha256)
    return {entry["participant_id"]: _normalise_embedding(entry["embedding"])
            for entry in gallery["participants"]}


def update_participant(gallery: dict[str, Any], participant_id: str, embedding: np.ndarray,
                       sample_count: int, quality: dict[str, Any], display_name: str | None = None) -> dict[str, Any]:
    if not isinstance(participant_id, str) or not participant_id.strip():
        raise ValueError("participant ID must be a non-empty local ID")
    if display_name is not None and (not isinstance(display_name, str) or not display_name.strip()):
        raise ValueError("display name must be non-empty")
    vector = _normalise_embedding(embedding)
    if vector.size != gallery["model"]["embedding_dimension"]:
        raise ValueError("embedding dimension does not match this private gallery")
    replacement = {"participant_id": participant_id, "display_name": (display_name or participant_id).strip(), "embedding": vector.tolist(),
                   "sample_count": sample_count,
                   "enrolled_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                   "quality": quality}
    gallery["participants"] = [entry for entry in gallery["participants"] if entry["participant_id"] != participant_id]
    gallery["participants"].append(replacement)
    return gallery


def delete_participant(gallery: dict[str, Any], participant_id: str) -> dict[str, Any]:
    if not isinstance(participant_id, str) or not participant_id.strip():
        raise ValueError("participant ID must be a non-empty local ID")
    gallery["participants"] = [entry for entry in gallery["participants"] if entry["participant_id"] != participant_id]
    return gallery


def write_gallery_safely(path: Path, gallery: dict[str, Any], backup: bool = True) -> Path | None:
    """Atomically replace a caller-owned gallery, preserving a timestamped backup."""
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_path: Path | None = None
    if backup and path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = path.with_name(f"{path.name}.{stamp}.bak")
        shutil.copy2(path, backup_path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(gallery, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return backup_path
