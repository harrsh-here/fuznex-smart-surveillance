"""Safe local discovery for the SPECTRA desktop console.

Only project-owned folders, an already saved choice, and explicitly saved model
directories are scanned.  This module never downloads, executes, moves, or
replaces model files.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


ROLE_FILENAMES = {
    "yolo": {"yolo11n.pt"},
    "scrfd": {"det_500m.onnx"},
    "sface": {"face_recognition_sface_2021dec.onnx"},
}


@dataclass(frozen=True)
class Discovery:
    config: dict[str, Any]
    availability: dict[str, str]
    changed: bool


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(project_root: Path) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads((project_root / "ai" / "model-manifest.json").read_text(encoding="utf-8"))
        return {item["id"]: item for item in data["models"]}
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return {}


def _roots(project_root: Path, config: dict[str, Any]) -> list[Path]:
    roots = [project_root / "models", project_root / "ai" / "models", project_root / "local_data" / "models", project_root.parent]
    # Downloaded weights are a conventional, user-owned local staging location.
    # We inspect only exact approved filenames; no media or arbitrary documents.
    try: roots.append(project_root.parents[3] / "Downloads")
    except IndexError: pass
    for value in config.get("model_dirs", []):
        if isinstance(value, str): roots.append(Path(value))
    # A saved file remains the first preference even when it is outside a scan root.
    for role in ROLE_FILENAMES:
        if isinstance(config.get(role), str) and config[role]: roots.insert(0, Path(config[role]).parent)
    unique: list[Path] = []
    for root in roots:
        try: root = root.resolve()
        except OSError: continue
        if root not in unique and root.is_dir(): unique.append(root)
    return unique


def _candidate_paths(project_root: Path, config: dict[str, Any], role: str) -> list[Path]:
    names = ROLE_FILENAMES[role]; found: list[Path] = []
    saved = config.get(role)
    if isinstance(saved, str) and saved:
        path = Path(saved)
        if path.is_file() and path.name.lower() in names: found.append(path)
    for root in _roots(project_root, config):
        try:
            for name in names:
                found.extend(root.rglob(name))
        except OSError:
            continue
    # Deterministic selection and no repeated hash work.
    unique: list[Path] = []
    for path in found:
        try: resolved = path.resolve()
        except OSError: continue
        if resolved.is_file() and resolved not in unique: unique.append(resolved)
    return unique


def _expected(manifest: dict[str, dict[str, Any]], role: str) -> str | None:
    key = {"yolo": "yolo11n-coco", "scrfd": "scrfd-500m-kps", "sface": "sface-2021dec"}[role]
    value = manifest.get(key, {}).get("sha256")
    return value.lower() if isinstance(value, str) else None


def discover_models(project_root: Path, saved_config: dict[str, Any]) -> Discovery:
    """Find compatible local files and reject a mismatched known checksum."""
    config = dict(saved_config); config.setdefault("hashes", {}); manifest = _manifest(project_root); availability: dict[str, str] = {}; changed = False
    for role in ROLE_FILENAMES:
        accepted: Path | None = None; digest: str | None = None; expected = _expected(manifest, role)
        for candidate in _candidate_paths(project_root, config, role):
            try: candidate_hash = file_sha256(candidate)
            except OSError: continue
            if expected and candidate_hash.lower() != expected:
                continue
            accepted, digest = candidate, candidate_hash; break
        if accepted:
            value = str(accepted)
            if config.get(role) != value or config["hashes"].get(role) != digest:
                config[role] = value; config["hashes"][role] = digest; changed = True
            availability[role] = "available"
        else:
            if config.pop(role, None) is not None: changed = True
            config["hashes"].pop(role, None)
            availability[role] = "not_found"
    return Discovery(config, availability, changed)


def discover_cameras(preferred: str | None, max_index: int = 4) -> list[str]:
    """Return working camera indices, preferring the last successful source."""
    import cv2
    choices: list[str] = []
    if preferred and preferred.isdecimal(): choices.append(preferred)
    choices.extend(str(index) for index in range(max_index + 1) if str(index) not in choices)
    working: list[str] = []
    for source in choices:
        capture = cv2.VideoCapture(int(source))
        try:
            if capture.isOpened():
                ok, _frame = capture.read()
                if ok: working.append(source)
        finally:
            capture.release()
    return working
