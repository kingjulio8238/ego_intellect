"""Delta mirroring between a container's local disk and the state volume (stdlib only).

Local path ``P`` is mirrored at ``mirror_root + P``. One index (absolute local path ->
``[size, mtime_ns]``) records the local stat at the moment the local file and its mirror
were known identical. It serves two purposes:

* pushes copy only files whose local stat differs from the index;
* checkouts restore each file's exact local ``mtime_ns`` from the index, so
  ``tools/e2e`` directory fingerprints (path + size + mtime_ns) stay identical across
  containers whether or not the volume itself preserves mtimes.

Only the long-lived parent process of a role container reads or writes the index.

Partitioned mode (``partition_dir``): one index file per synced root instead of one file
for everything, so several invocations can run at once on one state volume (batch mode)
without overwriting each other's entries. An invocation loads and saves only the roots it
syncs; a root without a partition file yet is seeded from the legacy single-file index.
"""

from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

# Volume file operations are latency-bound (~0.25 s per small file measured on Modal:
# 391 SLAM pose files took 94 s to pull serially), so copies run on a thread pool.
COPY_THREADS = int(os.environ.get("V2D_MODAL_COPY_THREADS", "32"))


def _parallel(function, items):
    """Run function over items on the copy pool; yield (item, result, error) in order."""
    items = list(items)
    if len(items) <= 1 or COPY_THREADS <= 1:
        for item in items:
            try:
                yield item, function(item), None
            except OSError as exc:
                yield item, None, exc
        return

    def guarded(item):
        try:
            return item, function(item), None
        except OSError as exc:
            return item, None, exc

    with ThreadPoolExecutor(max_workers=COPY_THREADS) as pool:
        yield from pool.map(guarded, items)


def mirror_path(mirror_root: str, local: str) -> str:
    local = os.path.abspath(local)
    return mirror_root.rstrip("/") + local


def local_path(mirror_root: str, mirrored: str) -> str:
    prefix = mirror_root.rstrip("/")
    if not (mirrored == prefix or mirrored.startswith(prefix + "/")):
        raise ValueError(f"{mirrored} is not under mirror root {prefix}")
    return mirrored[len(prefix):] or "/"


def _stat_key(path: str) -> list[int]:
    st = os.lstat(path)
    return [st.st_size, st.st_mtime_ns]


def _copy_file(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.islink(src):
        if os.path.lexists(dst):
            os.unlink(dst)
        os.symlink(os.readlink(src), dst)
        return
    tmp = f"{dst}.v2d-tmp"
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def walk(root: str) -> tuple[list[str], list[str]]:
    """Return (files, dirs) under root, not following symlinked directories."""
    files: list[str] = []
    dirs: list[str] = []
    if os.path.isfile(root) or os.path.islink(root):
        return [root], []
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        dirs.append(current)
        for name in dirnames:
            full = os.path.join(current, name)
            if os.path.islink(full):
                files.append(full)
        for name in filenames:
            files.append(os.path.join(current, name))
    return files, dirs


@dataclass
class SyncStats:
    copied: int = 0
    skipped: int = 0
    deleted: int = 0
    bytes: int = 0
    dirs: int = 0
    errors: list[str] = field(default_factory=list)

    def raise_errors(self, context: str) -> None:
        """Abort on any copy error: continuing after a partial checkout would let the next
        checkin delete the files that failed to arrive."""
        if self.errors:
            raise RuntimeError(f"{context}: {len(self.errors)} errors, first: {self.errors[0]}")

    def summary(self) -> str:
        text = (
            f"copied={self.copied} skipped={self.skipped} deleted={self.deleted} "
            f"dirs={self.dirs} bytes={self.bytes / 1e6:.1f}MB"
        )
        if self.errors:
            text += f" errors={len(self.errors)} (first: {self.errors[0]})"
        return text


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _read_index(path: str) -> dict[str, list[int]]:
    try:
        with open(path, encoding="utf-8") as stream:
            data = json.load(stream)
    except FileNotFoundError:
        return {}
    return {key: list(value) for key, value in data.get("files", {}).items()}


def _write_index(path: str, files: dict[str, list[int]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as stream:
        json.dump({"version": 1, "files": files}, stream)
    os.replace(tmp, path)


class Mirror:
    """Index-backed mirror of local directories onto a volume directory."""

    def __init__(self, mirror_root: str, index_path: str, partition_dir: str | None = None):
        self.mirror_root = mirror_root
        self.index_path = index_path  # legacy single-file index (partitioned mode: seed only)
        self.partition_dir = partition_dir
        self.index: dict[str, list[int]] = {}
        self.roots: dict[str, bool] = {}  # partitioned mode: root -> shared (merge on save)
        self.dirty: set[str] = set()  # partitioned mode: roots whose entries changed since save
        self._legacy: dict[str, list[int]] | None = None
        # Local paths whose mirror this process has seen or written (checkout, push, pull).
        # A push skips the volume existence check for them: it is a network round trip per
        # file, and a payload pull (no volume reload) would otherwise see a stale view.
        self.verified: set[str] = set()

    # -- index persistence ------------------------------------------------------------
    def partition_file(self, root: str) -> str:
        assert self.partition_dir is not None
        return os.path.join(self.partition_dir, os.path.abspath(root).strip("/").replace("/", "__") + ".json")

    def load(self, roots: dict[str, bool] | None = None) -> None:
        """Legacy mode: read the single index. Partitioned: read one file per root
        (``roots``: path -> shared, see register)."""
        if self.partition_dir is None:
            self.index = _read_index(self.index_path)
            return
        self.index, self.roots = {}, {}
        for root, shared in (roots or {}).items():
            self.register(root, shared=shared)

    def register(self, root: str, *, shared: bool = False) -> None:
        """Partitioned mode: track ``root`` in its own index file. Shared roots (used by
        every invocation, e.g. inputs) merge with the file's current content on save."""
        if self.partition_dir is None:
            return
        root = os.path.abspath(root)
        if root in self.roots:
            return
        self.roots[root] = shared
        path = self.partition_file(root)
        if os.path.exists(path):
            entries = _read_index(path)
        else:
            if self._legacy is None:
                self._legacy = _read_index(self.index_path)
            entries = {key: value for key, value in self._legacy.items() if _under(key, root)}
        for key, value in entries.items():
            self.index.setdefault(key, value)

    def _mark(self, path: str) -> None:
        for root in self.roots:
            if _under(path, root):
                self.dirty.add(root)
                return

    def save(self) -> None:
        if self.partition_dir is None:
            _write_index(self.index_path, self.index)
            return
        # Only partitions that changed: a batch holds many large run roots, and rewriting
        # (then committing) all of them before every module call cost seconds per call.
        # Paths under no registered root (temporary bind-mount sources) are not persisted:
        # only this invocation ever pushes them.
        dirty, self.dirty = self.dirty, set()
        for root, shared in self.roots.items():
            if root not in dirty:
                continue
            entries = {key: value for key, value in self.index.items() if _under(key, root)}
            path = self.partition_file(root)
            if not entries and not shared and os.path.exists(path):
                continue  # never blank another invocation's index (root not synced here)
            if shared:
                merged = _read_index(path)
                merged.update(entries)
                entries = merged
            _write_index(path, entries)

    # -- local -> volume --------------------------------------------------------------
    def push(self, local_root: str, *, delete: bool = True) -> SyncStats:
        """Make the mirror of ``local_root`` equal to it (deleting mirror-only files that
        this index knows about when ``delete``)."""
        stats = SyncStats()
        local_root = os.path.abspath(local_root)
        if not os.path.lexists(local_root):
            raise FileNotFoundError(local_root)
        files, dirs = walk(local_root)
        for directory in dirs:
            os.makedirs(mirror_path(self.mirror_root, directory), exist_ok=True)
            stats.dirs += 1
        present = set(files)

        def push_one(path: str):
            key = _stat_key(path)
            if self.index.get(path) == key and (
                path in self.verified or os.path.lexists(mirror_path(self.mirror_root, path))
            ):
                return None  # unchanged
            _copy_file(path, mirror_path(self.mirror_root, path))
            return key

        for path, key, error in _parallel(push_one, files):
            if isinstance(error, FileNotFoundError):
                continue  # vanished while walking (temporary file)
            if error is not None:
                stats.errors.append(f"{path}: {error}")
            elif key is None:
                stats.skipped += 1
                self.verified.add(path)
            else:
                self.index[path] = key
                self.verified.add(path)
                self._mark(path)
                stats.copied += 1
                stats.bytes += key[0]
        if delete:
            prefix = local_root.rstrip("/") + "/"
            for path in [p for p in self.index if (p == local_root or p.startswith(prefix)) and p not in present]:
                target = mirror_path(self.mirror_root, path)
                try:
                    if os.path.lexists(target):
                        os.unlink(target)
                    stats.deleted += 1
                except OSError as exc:
                    stats.errors.append(f"{target}: {exc}")
                self.index.pop(path, None)
                self.verified.discard(path)
                self._mark(path)
        return stats

    # -- volume -> local --------------------------------------------------------------
    def checkout(self, local_root: str) -> SyncStats:
        """Materialise the mirror of ``local_root`` locally, restoring indexed mtimes."""
        stats = SyncStats()
        local_root = os.path.abspath(local_root)
        source_root = mirror_path(self.mirror_root, local_root)
        os.makedirs(local_root, exist_ok=True)
        if not os.path.isdir(source_root):
            return stats
        files, dirs = walk(source_root)
        for directory in dirs:
            os.makedirs(local_path(self.mirror_root, directory), exist_ok=True)
            stats.dirs += 1
        def checkout_one(mirrored: str):
            path = local_path(self.mirror_root, mirrored)
            _copy_file(mirrored, path)
            recorded = self.index.get(path)
            if recorded and not os.path.islink(path) and os.path.getsize(path) == recorded[0]:
                os.utime(path, ns=(recorded[1], recorded[1]))
            return path, _stat_key(path)

        wanted = [m for m in files if not m.endswith(".v2d-tmp")]
        for mirrored, result, error in _parallel(checkout_one, wanted):
            if error is not None:
                stats.errors.append(f"{mirrored}: {error}")
                continue
            path, key = result
            self.index[path] = key
            self.verified.add(path)
            self._mark(path)
            stats.copied += 1
            stats.bytes += key[0]
        return stats

    def pull_changes(
        self,
        changed: list[str],
        deleted: list[str],
        dirs: list[str],
        payload: dict[str, tuple[bytes, int]] | None = None,
    ) -> SyncStats:
        """Apply a worker's reported mirror changes to local disk.

        ``payload`` (mirrored path -> (content, mtime_ns)): files the worker returned inline.
        They are written from memory with the worker's mtime instead of being read back from
        the volume, where each freshly committed small file costs a round trip (~0.25 s; 1561
        gsplat outputs took 12 s). Same bytes, same mtime as a volume copy (copy2).
        """
        stats = SyncStats()
        payload = payload or {}
        for mirrored in dirs:
            os.makedirs(local_path(self.mirror_root, mirrored), exist_ok=True)
            stats.dirs += 1
        def pull_one(mirrored: str):
            path = local_path(self.mirror_root, mirrored)
            inline = payload.get(mirrored)
            if inline is None:
                _copy_file(mirrored, path)
            else:
                content, mtime_ns = inline
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp = f"{path}.v2d-tmp"
                with open(tmp, "wb") as stream:
                    stream.write(content)
                os.utime(tmp, ns=(mtime_ns, mtime_ns))
                os.replace(tmp, path)
            return path, _stat_key(path)

        for mirrored, result, error in _parallel(pull_one, changed):
            if error is not None:
                stats.errors.append(f"{mirrored}: {error}")
                continue
            path, key = result
            self.index[path] = key
            self.verified.add(path)
            self._mark(path)
            stats.copied += 1
            stats.bytes += key[0]
        for mirrored in deleted:
            path = local_path(self.mirror_root, mirrored)
            try:
                if os.path.lexists(path) and not os.path.isdir(path):
                    os.unlink(path)
                stats.deleted += 1
            except OSError as exc:
                stats.errors.append(f"{path}: {exc}")
            self.index.pop(path, None)
            self.verified.discard(path)
            self._mark(path)
        return stats


def read_payload(changed: list[str], max_bytes: int, max_files: int = 20000) -> dict[str, tuple[bytes, int]]:
    """Worker side: the changed regular files as {path: (content, mtime_ns)}, or {} when they
    exceed the limits (the caller then copies them from the volume as before)."""
    if not changed or len(changed) > max_files:
        return {}
    sizes = {}
    for path in changed:
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            return {}
        if not os.path.isfile(path) or os.path.islink(path):
            return {}
        sizes[path] = (st.st_size, st.st_mtime_ns)
    if sum(size for size, _ in sizes.values()) > max_bytes:
        return {}

    def read(path: str):
        with open(path, "rb") as stream:
            return path, stream.read()

    out: dict[str, tuple[bytes, int]] = {}
    for path, result, error in _parallel(read, changed):
        if error is not None:
            return {}
        out[path] = (result[1], sizes[path][1])
    return out


def snapshot(roots: list[str]) -> tuple[dict[str, tuple[int, int]], set[str]]:
    """Stat every file and directory under ``roots`` (used by workers before/after a run)."""
    files: dict[str, tuple[int, int]] = {}
    directories: set[str] = set()
    for root in roots:
        if not os.path.lexists(root):
            continue
        found, dirs = walk(root)
        directories.update(dirs)
        for path in found:
            try:
                st = os.lstat(path)
            except FileNotFoundError:
                continue
            files[path] = (st.st_size, st.st_mtime_ns)
    return files, directories


def diff(
    before: tuple[dict[str, tuple[int, int]], set[str]],
    after: tuple[dict[str, tuple[int, int]], set[str]],
) -> tuple[list[str], list[str], list[str]]:
    """Return (changed_or_new_files, deleted_files, new_dirs)."""
    old_files, old_dirs = before
    new_files, new_dirs = after
    changed = sorted(path for path, key in new_files.items() if old_files.get(path) != key)
    deleted = sorted(path for path in old_files if path not in new_files)
    created_dirs = sorted(new_dirs - old_dirs)
    return changed, deleted, created_dirs
