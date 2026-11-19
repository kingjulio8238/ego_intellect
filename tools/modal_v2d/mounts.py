"""Emulate docker bind mounts with symlinks inside a Modal container (stdlib only)."""

from __future__ import annotations

import os

# Bind-mount destinations that only matter for a real docker daemon on a desktop host:
# per-user passwd/group shims, X11, ssh agent, and Isaac Kit cache dirs (Modal runs as
# root, so the image's own Kit dirs are writable).
SKIPPED_DESTINATIONS = (
    "/etc/passwd",
    "/etc/group",
    "/tmp/.X11-unix",
    "/tmp/.ssh",
    "/ssh-agent",
    "/isaac-sim/kit/data",
    "/isaac-sim/kit/cache",
    "/isaac-sim/kit/logs",
)
_ASIDE_SUFFIX = ".v2d-image"


def skipped(destination: str) -> bool:
    return destination.rstrip("/") in SKIPPED_DESTINATIONS


def apply(mounts: list[dict[str, str]]) -> list[dict[str, str]]:
    """Symlink each ``dst`` to ``src``; return undo records (apply order)."""
    undo: list[dict[str, str]] = []
    try:
        for mount in mounts:
            src, dst = mount["src"], mount["dst"].rstrip("/") or "/"
            if skipped(dst) or mount.get("mode") == "anonymous":
                continue
            if not os.path.lexists(src):
                raise FileNotFoundError(f"bind-mount source does not exist: {src} (for {dst})")
            if os.path.lexists(dst) and os.path.realpath(dst) == os.path.realpath(src):
                continue
            record = {"src": src, "dst": dst, "aside": ""}
            if os.path.islink(dst):
                record["previous_link"] = os.readlink(dst)
                os.unlink(dst)
            elif os.path.lexists(dst):
                aside = dst + _ASIDE_SUFFIX
                if os.path.lexists(aside):
                    raise FileExistsError(f"cannot shadow {dst}: {aside} already exists")
                os.rename(dst, aside)
                record["aside"] = aside
            os.makedirs(os.path.dirname(dst) or "/", exist_ok=True)
            os.symlink(src, dst)
            undo.append(record)
    except BaseException:
        revert(undo)
        raise
    return undo


def revert(undo: list[dict[str, str]]) -> None:
    """Undo ``apply`` in reverse order, restoring shadowed image paths."""
    for record in reversed(undo):
        dst = record["dst"]
        if os.path.islink(dst) and os.readlink(dst) == record["src"]:
            os.unlink(dst)
        if record.get("aside") and os.path.lexists(record["aside"]):
            if os.path.lexists(dst):
                # Something replaced our symlink; never delete unknown content.
                print(f"[v2d-mounts] warning: {dst} changed during the run; image copy left at {record['aside']}")
                continue
            os.rename(record["aside"], dst)
        elif record.get("previous_link") and not os.path.lexists(dst):
            os.symlink(record["previous_link"], dst)
