"""Small preflight checks used inside Modal role containers (stdlib only).

    python -m tools.modal_v2d.checks lfs-materialized
"""

from __future__ import annotations

import fnmatch
import os
import sys
from pathlib import Path

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def _files(repo: Path):
    """Regular files under repo, never following symlinks (.e2e and weights are links)."""
    for current, dirnames, filenames in os.walk(repo, followlinks=False):
        dirnames[:] = [name for name in dirnames if name != ".git"]
        for name in filenames:
            path = Path(current, name)
            if not path.is_symlink():
                yield path


def _lfs_patterns(repo: Path) -> list[tuple[Path, str]]:
    patterns = []
    for attributes in (path for path in _files(repo) if path.name == ".gitattributes"):
        for line in attributes.read_text(errors="replace").splitlines():
            fields = line.split()
            if len(fields) >= 2 and "filter=lfs" in fields[1:]:
                patterns.append((attributes.parent, fields[0]))
    return patterns


def lfs_materialized(repo: Path) -> int:
    """Fail if any file matched by a `filter=lfs` attribute is still an LFS pointer."""
    patterns = _lfs_patterns(repo)
    checked = pointers = 0
    for path in _files(repo):
        for base, pattern in patterns:
            try:
                relative = path.relative_to(base).as_posix()
            except ValueError:
                continue
            target = relative if "/" in pattern else path.name
            if fnmatch.fnmatch(target, pattern.lstrip("/")):
                checked += 1
                with path.open("rb") as stream:
                    if stream.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX:
                        pointers += 1
                        print(f"LFS pointer, not content: {path}")
                break
    print(f"checked {checked} LFS-tracked files; {pointers} unmaterialised pointers")
    return 1 if pointers else 0


def main(argv: list[str]) -> int:
    if argv == ["lfs-materialized"]:
        return lfs_materialized(Path.cwd())
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
