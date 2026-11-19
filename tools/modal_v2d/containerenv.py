"""Environment and working directory for commands that stand in for a docker container.

A real `docker run` sees the image's ENV plus `-e` flags. Inside Modal, the process
environment additionally carries Modal runtime variables (including a PYTHONPATH that
puts Modal's client packages and sitecustomize first) and, in role containers, the
launcher's own variables. Commands get the image's recorded PYTHONPATH instead, and the
role-only variables are dropped. Stdlib only.
"""

from __future__ import annotations

import os

from tools.modal_v2d import config as C


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as stream:
            return stream.read().strip()
    except FileNotFoundError:
        return None


def image_workdir() -> str | None:
    """The image's WORKDIR, recorded at build time (None outside a built image)."""
    return _read(C.IMAGE_WORKDIR_FILE) or None


def container_env(extra: dict[str, str] | None = None, image: str = "", base: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        key: value
        for key, value in (os.environ if base is None else base).items()
        if not key.startswith(C.ROLE_ONLY_ENV_PREFIXES) and key not in C.ROLE_ONLY_ENV_KEYS
    }
    recorded = _read(C.IMAGE_PYTHONPATH_FILE)
    if recorded is not None:  # running inside a built image: restore its own PYTHONPATH
        if recorded:
            env["PYTHONPATH"] = recorded
        else:
            env.pop("PYTHONPATH", None)
    # Python's default text encoding follows the C locale, which libraries in these
    # images reset to "C" (ASCII) at import time; gsplat's JIT build then fails reading
    # its UTF-8 CUDA sources (UnicodeDecodeError, reconstruct SAM3D, 2026-11-17). UTF-8
    # mode makes open() independent of the locale; an explicit -e still wins.
    env.setdefault("PYTHONUTF8", "1")
    env.update(extra or {})
    prefix = C.IMAGE_PATH_PREFIX.get(image)
    if prefix:
        env["PATH"] = f"{prefix}:{env.get('PATH', '')}"
    return env
