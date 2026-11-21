"""Shared helpers for the environment package and intake contracts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"
REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_SCHEMA = "env_package/v1"
SUPPORT_SCHEMA = "support_surfaces/v1"
INTAKE_SCHEMA = "intake/v1"
TASK_SCHEMA = "task_spec/v1"
ISAACLAB_SCHEMA = "isaaclab_env/v1"
MUJOCO_SCHEMA = "mujoco_env/v1"
_SCHEMA_FILES = {
    MUJOCO_SCHEMA: "mujoco_env_v1.schema.json",
    ISAACLAB_SCHEMA: "isaaclab_env_v1.schema.json",
    TASK_SCHEMA: "task_spec_v1.schema.json",
    PACKAGE_SCHEMA: "env_package_v1.schema.json",
    SUPPORT_SCHEMA: "support_surfaces_v1.schema.json",
    INTAKE_SCHEMA: "intake_v1.schema.json",
}


class ContractError(ValueError):
    """An input or package violates its contract."""


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def load_schema(name: str) -> dict[str, Any]:
    return json.loads((SCHEMA_DIR / _SCHEMA_FILES[name]).read_text())


def schema_errors(name: str, document: Any) -> list[str]:
    """Every JSON-schema violation of ``document`` against schema ``name`` (empty: valid)."""
    try:
        import jsonschema
    except ImportError as exc:  # fail loudly: an unvalidated package must never look valid
        raise ContractError("the `jsonschema` package is required (pip install jsonschema)") from exc
    validator_cls = jsonschema.validators.validator_for(load_schema(name))
    validator = validator_cls(load_schema(name), format_checker=validator_cls.FORMAT_CHECKER)
    errors = []
    for error in sorted(validator.iter_errors(document), key=lambda e: list(e.absolute_path)):
        where = "/".join(str(p) for p in error.absolute_path) or "(root)"
        errors.append(f"{name} {where}: {error.message}")
    return errors


def rigid_errors(matrix: Any, label: str, tol: float = 1e-5) -> list[str]:
    """Errors if ``matrix`` is not a 4x4 proper rigid transform."""
    m = np.asarray(matrix, dtype=float)
    if m.shape != (4, 4) or not np.all(np.isfinite(m)):
        return [f"{label}: not a finite 4x4 matrix"]
    errors = []
    rotation = m[:3, :3]
    if np.abs(rotation @ rotation.T - np.eye(3)).max() > tol or abs(np.linalg.det(rotation) - 1) > tol:
        errors.append(f"{label}: rotation block is not orthonormal with det +1")
    if np.abs(m[3] - [0, 0, 0, 1]).max() > tol:
        errors.append(f"{label}: last row is not [0, 0, 0, 1]")
    return errors


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(tmp, path)


def git_state(repo: Path = REPO_ROOT) -> tuple[str | None, bool | None]:
    """(HEAD revision, worktree dirty) of the repository that builds the package. In a Modal
    container the checkout is a snapshot: the launcher's V2D_SOURCE_REVISION ("<sha>[+dirty]")
    is the truth there."""
    source = os.environ.get("V2D_SOURCE_REVISION", "")
    if source and source != "unknown":
        return source.removesuffix("+dirty"), source.endswith("+dirty")
    try:
        revision = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
            check=True, capture_output=True, text=True,
        ).stdout
        return revision, bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None, None


def obj_references(obj_path: Path) -> tuple[list[str], list[str]]:
    """(material files, texture files) an OBJ needs, relative to its directory."""
    materials = []
    for line in obj_path.read_text(errors="replace").splitlines():
        if line.startswith("mtllib "):
            materials.append(line.split(None, 1)[1].strip())
    textures = []
    for material in materials:
        mtl = obj_path.parent / material
        if not mtl.is_file():
            continue
        for line in mtl.read_text(errors="replace").splitlines():
            parts = line.strip().split()
            if parts and parts[0].lower().startswith("map_"):
                textures.append(parts[-1])
    return materials, textures


def obj_vertices(obj_path: Path) -> np.ndarray:
    rows = [line.split()[1:4] for line in obj_path.read_text(errors="replace").splitlines() if line.startswith("v ")]
    return np.asarray(rows, dtype=float).reshape(-1, 3)
