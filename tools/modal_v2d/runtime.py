"""Main process of a role container (control or rg): bootstrap, state sync, RPC, launcher.

Lifecycle of one `run_e2e.sh` invocation on Modal:

1. bootstrap: writable repo copy at C.REPO_ROOT (content-derived mtimes), synthetic git
   metadata (tools/e2e fingerprints git-listed runtime files), `.e2e` pointer and
   reconstruction weights symlinked onto persistent storage, host wrapper packages
   installed, docker shim on PATH;
2. checkout: the active-run pointer, /e2e/inputs, and the run roots this invocation
   touches are copied from the state volume to local disk;
3. serve: an RPC socket turns shimmed `docker run` calls for other images into Modal
   function calls, syncing each bind-mounted directory through the state volume;
4. run `run_e2e.sh <argv>` with periodic checkpoints of those roots to the volume;
5. checkin: final push of the same roots, index save, volume commits.

All volume operations happen in this process under one lock. Modal volumes need explicit
commit/reload, and both fail while this container holds a volume file open, so the
launcher's working tree lives on local disk and volume files are only touched here.
Any sync error aborts: a partial checkout followed by a deleting checkin would otherwise
remove data from the volume.
"""

from __future__ import annotations

import json
import os
import shutil
import socketserver
import tempfile
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.modal_v2d import config as C
from tools.modal_v2d import containerenv, ledger, mounts, progress, sync

# Host-side wrapper packages per role (installed --no-deps; deps are baked into images).
CONTROL_HOST_PACKAGES = (
    "modules/v2d_common",
    "modules/v2d_docker",
    "modules/v2d_depth",
    "modules/v2d_anycalib/docker",
    "modules/v2d_geocalib/docker",
    "modules/v2d_droid_slam/docker",
    "modules/v2d_ego_hand_reconstruction/docker",
    "modules/v2d_foundation_pose/docker",
    "modules/v2d_grounding_dino/docker",
    "modules/v2d_gsplat_refinement/docker",
    "modules/v2d_hamer/docker",
    "modules/v2d_hand_alignment/docker",
    "modules/v2d_hawor/docker",
    "modules/v2d_mediapipe/docker",
    "modules/v2d_moge/docker",
    "modules/v2d_sam2/docker",
    "modules/v2d_sam3d/docker",
    "modules/v2d_wilor/docker",
    "modules/v2d_pipelines",
    "modules/v2d_task_library_loader/docker",
)
RG_HOST_PACKAGES = ("modules/v2d_docker", "modules/v2d_task_library_loader/docker")
RUNS_ROOT = os.path.join(C.LOCAL_STATE_ROOT, "runs")
INPUTS_ROOT = os.path.join(C.LOCAL_STATE_ROOT, "inputs")
POINTER_FILE = os.path.join(C.LOCAL_POINTER_DIR, "current")
# Synced by every invocation; all other synced roots are run roots, owned by one invocation.
SHARED_ROOTS = (C.LOCAL_POINTER_DIR, INPUTS_ROOT)
# Written by tools/modal_v2d/bake.py when the image already holds the prepared repo copy.
BAKED_MARKER = "/etc/v2d_repo_baked"  # outside the repo: never part of its git status
# Fixed epoch for uploaded repo files (Modal mounts do not carry local mtimes).
_REPO_MTIME_BASE_S = 946_684_800  # 2000-01-01


def log(message: str) -> None:
    print(f"[v2d {progress.utc()}] {message}", flush=True)


# --------------------------------------------------------------------------- bootstrap
def _copy_repo() -> None:
    """Writable repo copy whose mtimes depend only on file content.

    tools/e2e fingerprints directory inputs by (path, size, mtime_ns); repo directories
    can be stage inputs (e.g. `evaluate --motion-source repository`), so mtimes must be
    identical in every container that sees the same content.
    """
    started = time.time()
    shutil.copytree(C.REPO_UPLOAD, C.REPO_ROOT, symlinks=True, dirs_exist_ok=True)
    for current, _, filenames in os.walk(C.REPO_UPLOAD, followlinks=False):
        relative = os.path.relpath(current, C.REPO_UPLOAD)
        for name in filenames:
            source = os.path.join(current, name)
            if os.path.islink(source):
                continue
            checksum = 0
            with open(source, "rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    checksum = zlib.crc32(chunk, checksum)
            stamp = (_REPO_MTIME_BASE_S + checksum % 100_000_000) * 1_000_000_000
            os.utime(os.path.join(C.REPO_ROOT, relative, name), ns=(stamp, stamp))
    log(f"repo copied to {C.REPO_ROOT} in {time.time() - started:.0f}s")


def _symlink(link: str, target: str) -> None:
    os.makedirs(target, exist_ok=True)
    if os.path.islink(link):
        os.unlink(link)
    elif os.path.exists(link):
        raise FileExistsError(f"{link} exists in the uploaded repo; it must not be uploaded")
    os.makedirs(os.path.dirname(link), exist_ok=True)
    os.symlink(target, link)


def _synthetic_git(source_revision: str) -> None:
    """Give tools/e2e a git index of the uploaded tree (the real .git is not uploaded).

    Stage signatures use only the runtime-file tree hash, so a synthetic commit is
    resume-safe. The real revision is in the commit message and in
    /e2e/.e2e/source_revision.
    """
    repo = C.REPO_ROOT
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "v2d-modal",
        "GIT_AUTHOR_EMAIL": "v2d-modal@localhost",
        "GIT_COMMITTER_NAME": "v2d-modal",
        "GIT_COMMITTER_EMAIL": "v2d-modal@localhost",
        "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z",
        "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z",
    }
    no_lfs = [
        "-c", "filter.lfs.required=false", "-c", "filter.lfs.clean=cat",
        "-c", "filter.lfs.smudge=cat", "-c", "filter.lfs.process=",
    ]
    started = time.time()
    if not os.path.isdir(os.path.join(repo, ".git")):
        subprocess.run(["git", "init", "-q", repo], check=True, env=env)
    # tools/e2e signs every completed stage with a hash of the git-listed runtime files and
    # refuses to resume after any of them changes. This package is execution plumbing, not
    # pipeline logic: editing it must not invalidate completed stages (it did: "stage
    # 'retarget.prepare' inputs or command changed since completion", 2026-11-17).
    exclude = os.path.join(repo, ".git", "info", "exclude")
    os.makedirs(os.path.dirname(exclude), exist_ok=True)
    with open(exclude, "a", encoding="utf-8") as stream:
        stream.write("\n/tools/modal_v2d/\n")
    subprocess.run(["git", *no_lfs, "add", "-A"], cwd=repo, check=True, env=env)
    subprocess.run(
        ["git", *no_lfs, "commit", "-q", "--allow-empty", "-m", f"v2d-modal snapshot of {source_revision}"],
        cwd=repo,
        check=True,
        env=env,
    )
    log(f"synthetic git snapshot of {source_revision} in {time.time() - started:.0f}s")


def _install_host_packages(role: str) -> None:
    packages = CONTROL_HOST_PACKAGES if role == "control" else RG_HOST_PACKAGES
    args = [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--no-build-isolation"]
    for package in packages:
        args += ["-e", package]
    subprocess.run(args, cwd=os.path.join(C.REPO_ROOT, "reconstruction"), check=True)


def _install_shim() -> None:
    os.makedirs(C.SHIM_BIN, exist_ok=True)
    shim = Path(C.SHIM_BIN) / "docker"
    shim.write_text(
        "#!/bin/sh\n"
        f'exec "{sys.executable}" "{C.REPO_ROOT}/tools/modal_v2d/docker_shim.py" "$@"\n'
    )
    shim.chmod(0o755)


def _place_mano() -> None:
    """Copy the user's MANO files to every place the HaMeR route's readers look.

    MANO is licensed, so it is never in an image or downloaded; the user uploads it once
    to C.MANO_DIR/models/. Readers on this route (measured, not assumed):
      * HaMeR, manotorch and the three.js exporter, with assets root
        `<hamer>/_DATA/data`: they expect `<root>/`, `<root>/models/` or `<root>/mano/`;
      * WiLoR prompt masks and hand tracks, root `<wilor>/pretrained_models`: manotorch
        asserts `<root>/models/MANO_RIGHT.pkl`. WiLoR's own downloader leaves the file
        flat in `<root>/`, so `reconstruct` failed there (2026-11-17, "Can not find MANO
        assets .../pretrained_models/models/MANO_RIGHT.pkl").
    """
    source_dir = Path(C.MANO_DIR) / "models"
    if not (source_dir / "MANO_RIGHT.pkl").is_file():
        log(f"warning: {source_dir}/MANO_RIGHT.pkl missing; upload MANO before `setup` (see README)")
        return
    hamer = Path(C.RECON_WEIGHTS) / "hamer" / "_DATA" / "data"
    wilor = Path(C.RECON_WEIGHTS) / "wilor" / "pretrained_models"
    targets = [hamer, hamer / "models", hamer / "mano", wilor / "models"]
    for directory in targets:
        for name in ("MANO_RIGHT.pkl", "MANO_LEFT.pkl"):
            source, target = source_dir / name, directory / name
            if source.is_file() and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                log(f"placed {target}")


def baked() -> bool:
    return os.path.isfile(BAKED_MARKER)


def ensure_repo() -> None:
    """Repo copy for functions that only need the files (e.g. img_robotic_grounding)."""
    if not baked():
        _copy_repo()


def bootstrap(role: str) -> None:
    prebaked = baked()
    if not prebaked:  # images built before bake.py existed
        _copy_repo()
    _symlink(os.path.join(C.REPO_ROOT, ".e2e"), C.LOCAL_POINTER_DIR)
    _symlink(os.path.join(C.REPO_ROOT, "reconstruction", "data", "weights"), C.RECON_WEIGHTS)
    if not prebaked:
        _install_host_packages(role)
    _install_shim()
    log("bootstrap: repo copy, git index and host packages baked into the image" if prebaked else "bootstrap: prepared repo at runtime (image not baked)")
    if role == "control":
        _place_mano()


def launcher_env(role: str, source_revision: str) -> dict[str, str]:
    """Environment for run_e2e.sh: the image's own PYTHONPATH (not Modal's) plus the
    variables the shim and Modal-aware stages need."""
    env = containerenv.container_env()
    env.update(
        {
            "V2D_EXECUTOR": "modal",
            "V2D_MODAL_ROLE": role,
            "V2D_MODAL_IMAGES": ",".join(C.ALL_IMAGES),
            "V2D_MODAL_LOCAL_IMAGE": C.RG_IMAGE if role == "rg" else "",
            "V2D_MODAL_RPC": C.RPC_SOCKET,
            "V2D_SOURCE_REVISION": source_revision,
            "PATH": f"{C.SHIM_BIN}:{env.get('PATH', '')}",
            "PYTHON": sys.executable,
            "HF_HOME": C.HF_HOME,
            "ACCEPT_EULA": "Y",
            "PRIVACY_CONSENT": "Y",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


# --------------------------------------------------------------------------- argv
def _option_values(argv: list[str], name: str) -> list[str]:
    values = []
    for index, token in enumerate(argv):
        if token == name:
            if index + 1 >= len(argv):
                raise ValueError(f"{name} requires a value")
            values.append(argv[index + 1])
        elif token.startswith(name + "="):
            values.append(token.split("=", 1)[1])
    return values


def _resolve(path: str) -> str:
    # run_e2e.sh runs with cwd C.REPO_ROOT, so relative paths resolve there.
    return os.path.normpath(os.path.join(C.REPO_ROOT, path))


def _run_root_of(config_path: str) -> str:
    return os.path.dirname(_resolve(config_path))


def runs_touched(argv: list[str]) -> set[str]:
    """Run roots this invocation may read or write, validated to persist under RUNS_ROOT."""
    roots = {_resolve(value) for value in _option_values(argv, "--run-root")}
    roots |= {_run_root_of(value) for value in _option_values(argv, "--config")}
    if C.subcommand(argv) == "use":
        positional = [token for token in argv[argv.index("use") + 1:] if not token.startswith("-")]
        roots |= {_run_root_of(value) for value in positional[:1]}
    for root in roots:
        if not C.under(root, RUNS_ROOT) or root == RUNS_ROOT:
            raise ValueError(
                f"run roots must be inside {RUNS_ROOT}/ on Modal (got {root}); only that tree "
                "is persisted to the state volume"
            )
    return roots


def _active_run_from_volume() -> str | None:
    pointer = sync.mirror_path(C.MIRROR_ROOT, POINTER_FILE)
    try:
        with open(pointer, encoding="utf-8") as stream:
            value = stream.read().strip()
    except FileNotFoundError:
        return None
    return _run_root_of(value) if value else None


# --------------------------------------------------------------------------- state sync
class StateSync:
    """Owns the mirror index and serialises every volume operation in this container."""

    def __init__(self, state_volume, weights_volume):
        self.state_volume = state_volume
        self.weights_volume = weights_volume
        # One index file per synced root, so concurrent invocations (batch mode, one per
        # clip) never overwrite each other's entries.
        self.mirror = sync.Mirror(C.MIRROR_ROOT, C.INDEX_PATH, C.INDEX_PARTITION_DIR)
        self.lock = threading.RLock()
        self.completed: set[str] = set()  # signatures of CPU-only specs that already succeeded
        # (image, module) -> (FunctionCall, queue partition key): preloaded containers waiting
        # for their call (reconstruct --fast; see worker.serve_resident).
        # (caller, image, module): caller = the run_e2e.sh invocation that asked (batch mode runs
        # several in one container).
        self.residents: dict[tuple, tuple[object, str]] = {}
        self.resident_entries: dict[tuple, ledger.Entry] = {}  # their GPU budget entries
        self.trackers: dict[str, "progress.Tracker"] = {}  # caller -> its journal (batch mode)
        self.resident_lock = threading.Lock()
        self.ledger = ledger.GpuLedger(C.GPU_BUDGET, C.GPU_BUDGET_RESERVE, C.MODULE_SCALEDOWN_S) if C.GPU_BUDGET > 0 else None
        self.roots: list[str] = list(SHARED_ROOTS)
        # Volumes may surface at a different physical path; map back to logical mounts.
        self.logical_roots = [(os.path.realpath(root), root) for root in (C.STATE_MOUNT, C.WEIGHTS_MOUNT)]

    def logical(self, path: str) -> str:
        for physical, logical in self.logical_roots:
            if physical != logical and (path == physical or path.startswith(physical + "/")):
                return logical + path[len(physical):]
        return path

    @staticmethod
    def _volume_op(operation, what: str) -> None:
        for attempt in range(1, 4):
            try:
                operation()
                return
            except Exception as exc:  # e.g. "there are open files preventing the operation"
                if attempt == 3:
                    raise RuntimeError(
                        f"volume {what} failed after {attempt} attempts: {exc}. A process in this "
                        "container is probably holding a file under /vol or /weights open; keep "
                        "inputs under /e2e (see README)."
                    ) from exc
                log(f"volume {what} attempt {attempt} failed ({exc}); retrying")
                time.sleep(10 * attempt)

    def commit(self, *, weights: bool = False) -> None:
        self._volume_op(self.state_volume.commit, "commit")
        if weights:
            self._volume_op(self.weights_volume.commit, "weights commit")

    def reload(self, *, weights: bool = False) -> None:
        self._volume_op(self.state_volume.reload, "reload")
        if weights:
            self._volume_op(self.weights_volume.reload, "weights reload")

    def checkout(self, argv: list[str]) -> None:
        with self.lock:
            self.reload()
            runs = runs_touched(argv)
            if not runs:  # no explicit run root: the command works on the active run
                active = _active_run_from_volume()
                if active and C.under(active, RUNS_ROOT):
                    runs.add(active)
            self.roots += sorted(runs)
            self.mirror.load({root: root in SHARED_ROOTS for root in self.roots})
            for root in self.roots:
                stats = self.mirror.checkout(root)
                stats.raise_errors(f"checkout of {root}")
                log(f"checkout {root}: {stats.summary()}")

    def checkin(self, reason: str) -> None:
        with self.lock:
            for root in self.roots:
                if not os.path.lexists(root):
                    continue
                # Shared roots never delete: another invocation may have added files
                # this one has not seen.
                stats = self.mirror.push(root, delete=root not in SHARED_ROOTS)
                stats.raise_errors(f"checkin of {root}")
                log(f"checkin ({reason}) {root}: {stats.summary()}")
            self.mirror.save()
            # Final checkin also commits the weights volume (e.g. the GR00T base model
            # cached under HF_HOME during finetune).
            self.commit(weights=reason == "final")

    def add_root(self, root: str) -> None:
        """Track a run root created during this invocation (e.g. by `init`)."""
        with self.lock:
            if root not in self.roots:
                self.roots.append(root)
                self.mirror.register(root)

    tracker: "progress.Tracker | None" = None

    cpu_functions: dict = {}
    hedge_functions: dict = {}
    weights_dirty = True  # bootstrap may have placed MANO under /weights
    # Per-invocation execution options (batch mode): warm (bool), residents (bool),
    # hedge_after_s (float), gpu_slots_key (str: GPU_SLOTS_QUEUE partition).
    options: dict = {}

    @staticmethod
    def _signature(spec: dict) -> str:
        keys = ("image", "argv", "env", "workdir", "mounts")
        return json.dumps({key: spec.get(key) for key in keys}, sort_keys=True)

    def run_remote(self, spec: dict, functions: dict, emit, caller: str | None = None) -> int:
        """Execute one shimmed container spec in its Modal function; return exit code.

        The sync lock covers pushing inputs, the mirror index and pulling outputs, but not
        the remote call itself, so several calls can be in flight at once (prefetch_weights,
        reconstruct --fast). Workers never reload mid-call, so a concurrent checkin or
        another call's commit cannot change the files a running worker sees.
        """
        image = spec["image"]
        tracker = self.trackers.get(caller, self.tracker) if caller else self.tracker
        on_cpu = image in self.cpu_functions and any(
            marker in arg for arg in spec["argv"] for marker in C.CPU_ONLY_MARKERS
        )
        if on_cpu and self._signature(spec) in self.completed:
            emit(f"[v2d-modal] {image}: {' '.join(spec['argv'])} already completed in this invocation (prefetch); skipping\n")
            return 0
        function = self.cpu_functions[image] if on_cpu else functions.get(image)
        if function is None:
            raise RuntimeError(f"no Modal function for image {image!r}")
        repo_root = os.path.realpath(C.REPO_ROOT)
        remote_mounts = []
        push_bytes = push_files = 0
        call_t0 = time.time()
        self.lock.acquire()
        held = True
        try:
            for mount in spec["mounts"]:
                if mount.get("mode") == "anonymous" or mounts.skipped(mount["dst"]):
                    continue  # anonymous volumes; desktop-only mounts (passwd, X11, ssh, Kit caches)
                src = self.logical(os.path.realpath(mount["src"]))
                entry = {"dst": mount["dst"], "src": src, "mirrored": False}
                if image in C.MODULE_IMAGES and self._is_dev_mount(mount["dst"], src, repo_root):
                    emit(f"[v2d-modal] {image}: --dev mount of {src} not needed (image built from this checkout)\n")
                    continue
                if C.under(src, C.STATE_MOUNT) or C.under(src, C.WEIGHTS_MOUNT):
                    pass  # already shared storage
                elif image in C.IMAGES_WITH_REPO and C.under(src, repo_root):
                    entry["src"] = os.path.normpath(os.path.join(C.REPO_ROOT, os.path.relpath(src, repo_root)))
                elif os.path.lexists(src):
                    stats = self.mirror.push(src)
                    stats.raise_errors(f"sync of {src}")
                    push_bytes += stats.bytes
                    push_files += stats.copied
                    entry["src"] = sync.mirror_path(C.MIRROR_ROOT, src)
                    entry["mirrored"] = True
                    emit(f"[v2d-modal] synced {src} -> volume: {stats.summary()}\n")
                else:
                    raise FileNotFoundError(f"bind-mount source does not exist: {src}")
                remote_mounts.append(entry)
            self.mirror.save()
            # The control container writes /weights only when bootstrap placed MANO; workers
            # reload /weights themselves, so commit it only then.
            self.commit(weights=self.weights_dirty)
            self.weights_dirty = False
            mirrored = any(entry["mirrored"] for entry in remote_mounts)

            remote_spec = {**spec, "mounts": remote_mounts, "call_id": uuid.uuid4().hex[:12]}
            if not on_cpu and self.options.get("keep_models"):
                remote_spec["keep_warm"] = True  # worker preloads this module for the next call
            if not on_cpu and self.options.get("exit_after_call"):
                # Batch mode: the container exits after its call, so the GPUs held equal the
                # calls in flight (GPU slots) instead of including idle scaledown containers.
                remote_spec["exit_after"] = True
            call_id = remote_spec["call_id"]
            emit(f"[v2d-modal] -> Modal function for {image}{' (CPU)' if on_cpu else ''}: {' '.join(spec['argv'])}\n")
            started = time.time()
            push_s = started - call_t0
            if tracker:
                gpu = None if on_cpu else (C.RG_GPU if image == C.RG_IMAGE else C.gpu_label(C.MODULE_GPU))
                if on_cpu:
                    cpu, memory = C.DOWNLOAD_CPU, C.DOWNLOAD_MEM_GIB
                elif image == C.RG_IMAGE:
                    cpu, memory = C.RG_CPU, C.RG_MEM_GIB
                else:
                    cpu, memory = C.MODULE_CPU, C.MODULE_MEM_GIB
                tracker.call_start(image, call_id, spec["argv"], gpu, cpu, memory)
                # Full spec (volume-side mount sources) so tools/fastrecon can replay
                # this exact call in isolation later.
                tracker.event(
                    "call_spec",
                    call_id=call_id,
                    on_cpu=on_cpu,
                    spec={key: remote_spec.get(key) for key in ("image", "argv", "env", "workdir", "mounts", "gpu")},
                )
            # Release the sync lock while the call runs, so an orchestrator that issues
            # independent calls concurrently (reconstruct --fast) gets real overlap. Safe for
            # mirrored mounts too: each worker reads its own volume snapshot (no reload
            # mid-call), commits only the files it wrote, and every push/pull/index update
            # below still happens under the lock. A serial orchestrator behaves as before.
            self.lock.release()
            held = False
            slot = None if on_cpu else self._acquire_slot(image, emit)
            slot_wait_s = round(time.time() - started, 3)
            # Single-clip mode: budget this invocation's GPU containers (ledger.py).
            budget = self.ledger if (slot is None and not on_cpu and not self.options.get("exit_after_call")) else None
            argv = remote_spec["argv"]
            module = argv[2] if len(argv) > 2 and argv[1] == "-m" else None
            with self.resident_lock:
                held_entry = self.resident_entries.pop((caller, image, module), None)
            try:
                result = self._run_on_resident(image, remote_spec, emit, caller)
                if result is None:
                    if budget is not None:
                        budget.drop(held_entry)  # resident (if any) not used: its container is gone
                        held_entry, waited = budget.acquire(image)
                        slot_wait_s = round(slot_wait_s + waited, 3)
                        if waited > 5:
                            emit(f"[v2d-modal] {image}: waited {waited:.0f}s for GPU budget "
                                 f"({budget.held()}/{budget.capacity} containers held)\n")
                    result = self._call_hedged(image, function, remote_spec, emit, on_cpu, tracker, budget)
            except BaseException:
                if tracker:
                    tracker.call_end(-1, time.time() - started, call_id=call_id)
                raise
            finally:
                self._release_slot(slot)
                if budget is not None:
                    budget.done(held_entry)
            if slot is not None or budget is not None:
                result = {**result, "timing": {**(result.get("timing") or {}), "slot_wait_s": slot_wait_s}}
            remote_s = time.time() - started
            if not held:
                self.lock.acquire()
                held = True
            pull_t0 = time.time()
            self.reload()  # this container never reads /weights after bootstrap
            stats = self.mirror.pull_changes(result["changed"], result["deleted"], result["dirs"])
            self.mirror.save()
            pull_s = time.time() - pull_t0
            if tracker:
                timing = dict(result.get("timing") or {})
                worker_s = timing.get("worker_s", result.get("elapsed_s", 0.0))
                tracker.call_end(
                    int(result["returncode"]),
                    time.time() - call_t0,
                    {
                        **timing,
                        "push_s": round(push_s, 3),
                        "push_bytes": push_bytes,
                        "push_files": push_files,
                        "remote_s": round(remote_s, 3),
                        # scheduling + cold start + image load: the round trip minus the worker's own time
                        "startup_s": round(max(remote_s - worker_s - timing.get("slot_wait_s", 0.0), 0.0), 3),
                        "pull_s": round(pull_s, 3),
                        "pull_bytes": stats.bytes,
                        "overhead_s": round(time.time() - call_t0 - timing.get("exec_s", worker_s), 3),
                        "on_cpu": on_cpu,
                    },
                    call_id=call_id,
                )
                # Volume-side paths this call wrote: the replay harness's ground truth for
                # what a call produces (tools/fastrecon).
                tracker.event(
                    "call_outputs",
                    call_id=call_id,
                    changed=list(result["changed"]),
                    deleted=list(result["deleted"]),
                    dirs=list(result["dirs"]),
                )
            stats.raise_errors(f"pull of {image} outputs")
            try:
                with open(result["log"], encoding="utf-8", errors="replace") as stream:
                    for line in stream:
                        emit(line)
            except FileNotFoundError:
                emit(f"[v2d-modal] remote log missing: {result['log']}\n")
        finally:
            if held:
                self.lock.release()
        if on_cpu and int(result["returncode"]) == 0:
            self.completed.add(self._signature(spec))
        if not on_cpu and int(result["returncode"]) == 0:
            record_resident_hint(remote_spec)
        emit(
            f"[v2d-modal] <- {image} exit={result['returncode']} in {time.time() - started:.0f}s; "
            f"pulled {stats.summary()}\n"
        )
        return int(result["returncode"])

    # ------------------------------------------------------------------ GPU slots
    def _acquire_slot(self, image: str, emit):
        key = self.options.get("gpu_slots_key")
        if not key:
            return None
        import modal  # noqa: PLC0415

        queue = modal.Queue.from_name(C.GPU_SLOTS_QUEUE, create_if_missing=True)
        t0 = time.time()
        token = None
        while token is None:  # blocking get, in bounded waits so a dead batch cannot hang us forever
            try:
                token = queue.get(partition=key, timeout=600)
            except Exception:  # noqa: BLE001  - queue.Empty on timeout
                token = None
            if token is None and time.time() - t0 > C.MODULE_TIMEOUT:
                raise RuntimeError(f"no GPU slot for {image} after {time.time() - t0:.0f}s")
        waited = time.time() - t0
        if waited > 5:
            emit(f"[v2d-modal] {image}: waited {waited:.0f}s for a GPU slot (batch at its GPU budget)\n")
        return (queue, key, token)

    @staticmethod
    def _release_slot(slot) -> None:
        if slot is None:
            return
        queue, key, token = slot
        for _ in range(3):
            try:
                queue.put(token, partition=key)
                return
            except Exception:  # noqa: BLE001
                time.sleep(2)

    # ------------------------------------------------------------------ hedged calls
    def _call_hedged(self, image: str, function, spec: dict, emit, on_cpu: bool, tracker=None, budget=None) -> dict:
        """Run ``spec`` on ``function``; if no container has started it after
        C.HEDGE_AFTER_S, also send it to the image's hedge twin. Exactly one runs it: the
        first container to claim it (worker.claim); the other is cancelled, or exits at once.
        """
        hedge_after = float(self.options.get("hedge_after_s", C.HEDGE_AFTER_S))
        hedge = None if on_cpu else self.hedge_functions.get(image)
        if hedge is None or hedge_after <= 0:
            return function.remote(spec)
        import modal  # noqa: PLC0415

        claims = modal.Dict.from_name(C.CLAIM_DICT, create_if_missing=True)
        key = f"{spec['call_id']}-{uuid.uuid4().hex[:8]}"
        spec = {**spec, "claim": key}
        t0 = time.time()
        calls = {"primary": function.spawn({**spec, "lane": "primary"})}
        owner = None
        hedge_entry = None
        hedge_blocked = False
        try:
            while owner is None:
                owner = claims.get(key)
                if owner is not None:
                    break
                for lane, call in calls.items():
                    try:
                        call.get(timeout=0)  # returned without a claim: it failed at start
                    except (TimeoutError, modal.exception.TimeoutError):
                        continue
                    except Exception:  # noqa: BLE001
                        if len(calls) == 1:
                            raise
                        emit(f"[v2d-modal] {image}: {lane} container failed before starting the call\n")
                if "hedge" not in calls and not hedge_blocked and time.time() - t0 > hedge_after:
                    # A hedge is one more GPU container: only with budget left, or it pushes
                    # the workspace past its cap and delays everything (batch b2: SAM3D
                    # hedges waited 96-200 s behind each other).
                    if budget is not None:
                        hedge_entry = budget.speculate(image, busy=True, reserve=0)
                        if hedge_entry is None:
                            hedge_blocked = True
                            emit(f"[v2d-modal] {image}: no container after {hedge_after:.0f}s; no GPU budget "
                                 "left for a hedge, waiting\n")
                            continue
                    emit(f"[v2d-modal] {image}: no container after {hedge_after:.0f}s; hedging on a "
                         "second container pool (first to start runs it)\n")
                    calls["hedge"] = hedge.spawn({**spec, "lane": "hedge"})
                    if tracker:
                        tracker.event("hedge", call_id=spec["call_id"], image=image, after_s=round(time.time() - t0, 1))
                time.sleep(1.0)
            lane = owner["lane"]
            claim_s = round(owner["t"] - t0, 3)
            for other, call in calls.items():
                if other != lane:
                    try:
                        call.cancel()
                    except Exception:  # noqa: BLE001
                        pass
            if "hedge" in calls:
                emit(f"[v2d-modal] {image}: {lane} container started the call after {claim_s:.0f}s\n")
            result = calls[lane].get()
        finally:
            if budget is not None:
                budget.drop(hedge_entry)  # the loser exits; the winner is the call's own entry
            # The claim stays (a tombstone; Modal expires idle Dict entries). Cancelling the
            # losing call does not always stop it: batch b5's cancelled primary started 31 s
            # after its hedge had finished and the claim was gone, ran the call a second time,
            # and its orphaned encode corrupted another clip's video.
        timing = dict(result.get("timing") or {})
        timing.update({"lane": lane, "hedged": "hedge" in calls, "claim_s": claim_s})
        return {**result, "timing": timing}

    # ------------------------------------------------------------------ resident calls
    def spawn_residents(self, image: str, function, caller: str | None = None) -> int:
        """Start a preloading container for each call of `image` seen in earlier runs."""
        hints = resident_hints(image)
        started = 0
        for module, hint in hints.items():
            if module not in C.RESIDENT_MODULES:
                continue
            with self.resident_lock:
                if (caller, image, module) in self.residents:
                    continue
                entry = self.ledger.speculate(image, busy=True) if self.ledger is not None else None
                if self.ledger is not None and entry is None:
                    log(f"resident {image} {module}: skipped (GPU budget: {self.ledger.held()}/{self.ledger.capacity} held)")
                    continue
                if entry is not None:
                    self.resident_entries[(caller, image, module)] = entry
                key = "r" + uuid.uuid4().hex[:16]
                call = function.spawn({"image": image, "resident": {**hint, "key": key},
                                       "call_id": "resident-" + key[1:9]})
                self.residents[(caller, image, module)] = (call, key)
            started += 1
            log(f"resident {image} {module}: preloading (call {call.object_id})")
        return started

    def _run_on_resident(self, image: str, remote_spec: dict, emit, caller: str | None = None) -> dict | None:
        """Hand the spec to a preloaded container; None if there is none (or it is gone)."""
        argv = remote_spec["argv"]
        module = argv[2] if len(argv) > 2 and argv[1] == "-m" else None
        with self.resident_lock:
            entry = self.residents.pop((caller, image, module), None) if module else None
        if entry is None:
            return None
        call, key = entry
        import modal  # noqa: PLC0415

        queue = modal.Queue.from_name(C.RESIDENT_QUEUE, create_if_missing=True)
        queue.put(remote_spec, partition=key)
        if not self._claimed(queue, key, call):
            emit(f"[v2d-modal] {image}: resident container did not pick up the call within "
                 f"{C.RESIDENT_CLAIM_S}s (not scheduled yet?); running normally\n")
            return None
        emit(f"[v2d-modal] {image}: running on the preloaded container ({module})\n")
        try:
            result = call.get()
        except Exception as exc:  # noqa: BLE001  - crashed before/while running: run it normally
            emit(f"[v2d-modal] {image}: resident container failed ({type(exc).__name__}: {exc}); running normally\n")
            return None
        if result.get("unclaimed"):
            emit(f"[v2d-modal] {image}: resident container gave up waiting; running normally\n")
            return None
        return result

    @staticmethod
    def _claimed(queue, key: str, call) -> bool:
        """True once the resident has taken the spec off its queue partition.

        A resident container that was never scheduled would hold the call forever
        (fast4b: Grounding DINO's stayed pending 5 min while fresh calls started at once).
        After RESIDENT_CLAIM_S the spec is taken back; queue.get is atomic, so exactly one
        side owns it: if it is still there, the resident is cancelled and the call runs
        normally; if it is gone, the resident has it.
        """
        deadline = time.time() + C.RESIDENT_CLAIM_S
        while time.time() < deadline:
            if queue.len(partition=key) == 0:
                return True
            time.sleep(0.5)
        if queue.get(partition=key, block=False) is None:
            return True  # claimed between the last check and the take-back
        try:
            call.cancel()
        except Exception:  # noqa: BLE001
            pass
        return False

    def cancel_residents(self) -> None:
        with self.resident_lock:
            pending, self.residents = list(self.residents.items()), {}
            entries, self.resident_entries = list(self.resident_entries.values()), {}
        if self.ledger is not None:
            for entry in entries:
                self.ledger.drop(entry)
        for (_, image, module), (call, _) in pending:
            try:
                call.cancel()
                log(f"resident {image} {module}: not used; cancelled")
            except Exception as exc:  # noqa: BLE001
                log(f"resident {image} {module}: cancel failed: {exc}")

    @staticmethod
    def _is_dev_mount(dst: str, src: str, repo_root: str) -> bool:
        subtree, target = C.DEV_MOUNT
        return dst.rstrip("/") == target and os.path.normpath(src) == os.path.join(repo_root, subtree)


# --------------------------------------------------------------------------- resident hints
def _hints_dict():
    import modal  # noqa: PLC0415

    return modal.Dict.from_name(C.RESIDENT_HINTS_DICT, create_if_missing=True)


def resident_hints(image: str) -> dict:
    """{module: {argv, env, workdir}} of `image`'s calls in earlier runs ({} if disabled)."""
    try:
        store = _hints_dict()
        if store.get("disabled"):
            return {}
        return dict(store.get(image) or {})
    except Exception as exc:  # noqa: BLE001  - no hints = plain warm-up
        log(f"resident hints unavailable: {exc}")
        return {}


def record_resident_hint(spec: dict) -> None:
    """Remember a successful `python -m` call so the next run can preload it."""
    argv = spec.get("argv") or []
    if len(argv) < 3 or argv[1] != "-m":
        return
    try:
        store = _hints_dict()
        hints = dict(store.get(spec["image"]) or {})
        hints[argv[2]] = {"argv": list(argv), "env": dict(spec.get("env") or {}), "workdir": spec.get("workdir"),
                          "mounts": [m for m in spec.get("mounts") or [] if not m.get("mirrored")]}
        store[spec["image"]] = hints
    except Exception as exc:  # noqa: BLE001
        log(f"could not record resident hint for {spec['image']}: {exc}")


# --------------------------------------------------------------------------- RPC server
def serve(state: StateSync, functions: dict) -> socketserver.BaseServer:
    if os.path.exists(C.RPC_SOCKET):
        os.unlink(C.RPC_SOCKET)

    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            def emit(text: str) -> None:
                self.wfile.write((json.dumps({"log": text}) + "\n").encode())
                self.wfile.flush()

            try:
                request = json.loads(self.rfile.readline())
                if request.get("op") == "warm":
                    function = functions.get(request["image"])
                    if not state.options.get("warm", True):
                        function = None  # batch throughput mode: no speculative containers
                    if function is not None and state.options.get("residents", True) and state.spawn_residents(request["image"], function, request.get("caller")):
                        pass  # preloading containers started; they also serve as the warm-up
                    elif function is not None and state.ledger is not None and state.ledger.speculate(request["image"], busy=False) is None:
                        log(f"warm-up of {request['image']} skipped (GPU budget: {state.ledger.held()}/{state.ledger.capacity} held)")
                    elif function is not None:
                        # A no-op call: Modal starts (or reuses) a container of this image,
                        # which then stays up for the function's scaledown window.
                        function.spawn({"image": request["image"], "argv": ["true"], "env": {},
                                        "workdir": None, "mounts": [], "call_id": "warm-" + uuid.uuid4().hex[:8]})
                        log(f"warming a {request['image']} container")
                    self.wfile.write((json.dumps({"exit": 0}) + "\n").encode())
                    self.wfile.flush()
                    return
                if request.get("op") != "run":
                    raise ValueError(f"unknown RPC op {request.get('op')!r}")
                code = state.run_remote(request["spec"], functions, emit, request.get("caller"))
                self.wfile.write((json.dumps({"exit": code}) + "\n").encode())
            except Exception as exc:  # report to the shim; it exits non-zero
                traceback.print_exc()
                self.wfile.write((json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n").encode())
            self.wfile.flush()

    server = socketserver.ThreadingUnixStreamServer(C.RPC_SOCKET, Handler)
    threading.Thread(target=server.serve_forever, name="v2d-rpc", daemon=True).start()
    return server


def _periodic_checkin(state: StateSync, stop: threading.Event) -> None:
    while not stop.wait(C.SYNC_INTERVAL_S):
        try:
            state.checkin("periodic")
        except Exception as exc:  # retried next tick; the final checkin must succeed
            log(f"periodic checkin failed: {exc}")


# --------------------------------------------------------------------------- entry
def planned_stages(argv: list[str], env: dict[str, str]) -> list[str]:
    """Stage names this invocation will run, from the launcher's own --dry-run."""
    if "--dry-run" in argv or C.subcommand(argv) in {"init", "use", "status"}:
        return []
    try:
        result = subprocess.run(
            ["bash", os.path.join(C.REPO_ROOT, "run_e2e.sh"), "--dry-run", *argv],
            cwd=C.REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        log("stage plan: --dry-run timed out; progress will show stage N/?")
        return []
    names = [m.group(1) for line in result.stdout.splitlines() if (m := progress.STAGE_PLAN.match(line))]
    if result.returncode != 0 or not names:
        tail = (result.stdout + result.stderr).strip().splitlines()[-3:]
        log(f"stage plan unavailable (exit {result.returncode}): {' / '.join(tail)}")
    return names


WEIGHTS_SCRIPT = "scripts/download_ego_reconstruction_weights.sh"
WEIGHTS_MODE = "hamer_prompt"  # what tools/e2e/stages.py setup_stages runs


def prefetch_weights(argv: list[str], env: dict[str, str], tracker: progress.Tracker, state: StateSync) -> None:
    """Run setup's weight downloads concurrently on CPU containers before setup itself.

    The repository's download script runs eleven wrappers one after another, each a
    separate container start. Here the script is first run with PYTHON pointing at a
    collector, which records each wrapper invocation without executing it; the
    invocations then run in parallel (each `docker run` lands in run_remote, which lets
    unmirrored calls overlap). Specs that succeed are remembered, so when setup's own
    sequential pass makes the identical `docker run`, it returns at once. A prefetch
    failure is reported and left for that sequential pass, which retries it with the
    launcher's normal logging and failure handling.
    """
    if C.subcommand(argv) != "setup" or "--dry-run" in argv or "--skip-reconstruction" in argv or not C.PREFETCH_WEIGHTS:
        return
    recon = os.path.join(C.REPO_ROOT, "reconstruction")
    python = env.get("PYTHON", sys.executable)
    with tempfile.TemporaryDirectory() as tmp:
        collector, listing = os.path.join(tmp, "collect-python"), os.path.join(tmp, "calls.jsonl")
        Path(collector).write_text(
            "#!/bin/sh\n"
            f'exec "{python}" -c \'import json, sys; open(sys.argv[1], "a").write(json.dumps(sys.argv[2:]) + "\\n")\' "{listing}" "$@"\n'
        )
        os.chmod(collector, 0o755)
        script_args = ["bash", WEIGHTS_SCRIPT, "--mode", WEIGHTS_MODE]
        if "--accept-nvidia-model-eula" in argv:
            script_args.append("--accept-nvidia-model-eula")
        listed = subprocess.run(script_args, cwd=recon, env={**env, "PYTHON": collector}, capture_output=True, text=True)
        if listed.returncode != 0 or not os.path.exists(listing):
            log(f"prefetch skipped: could not list downloads (exit {listed.returncode}): {(listed.stderr or listed.stdout)[-300:]}")
            return
        calls = [json.loads(line) for line in Path(listing).read_text().splitlines() if line.strip()]
    names = [next((a.split(".")[1] for a in call if a.startswith("v2d.")), "?") for call in calls]
    tracker.side_stage_start("prefetch.weights", f"▶ prefetch: {len(calls)} weight downloads in parallel on CPU ({', '.join(names)})")

    def run_one(call: list[str]) -> tuple[list[str], int, str]:
        result = subprocess.run([python, *call], cwd=recon, env=env, capture_output=True, text=True)
        return call, result.returncode, result.stdout + result.stderr

    failures = []
    with ThreadPoolExecutor(max_workers=max(1, min(len(calls), C.PREFETCH_PARALLEL))) as pool:
        for call, code, output in pool.map(run_one, calls):
            name = next((a for a in call if a.startswith("v2d.")), " ".join(call))
            if code != 0:
                failures.append(name)
                tail = "\n".join(output.strip().splitlines()[-12:])
                log(f"prefetch ✘ {name} exit={code}; setup will retry it. Last lines:\n{tail}")
    tracker.side_stage_end("prefetch.weights", ok=not failures)
    log(
        f"prefetch: {len(calls) - len(failures)}/{len(calls)} downloads done"
        + (f"; failed: {', '.join(failures)}" if failures else "")
    )


def run_role(
    role: str,
    argv: list[str],
    *,
    source_revision: str,
    state_volume,
    weights_volume,
    functions,
    progress_store=None,
    invocation: str | None = None,
    app_id: str = "",
    cpu_functions: dict | None = None,
    hedge_functions: dict | None = None,
    options: dict | None = None,
) -> int:
    expected = C.role_for(argv)
    if role != expected:
        raise ValueError(f"`{C.subcommand(argv)}` must run in the {expected} role, not {role}")
    runs_touched(argv)  # validate before paying for bootstrap
    invocation = invocation or uuid.uuid4().hex[:8]
    gpu = C.RG_GPU if role == "rg" else None
    cpu, memory = (C.RG_CPU, C.RG_MEM_GIB) if role == "rg" else (C.CONTROL_CPU, C.CONTROL_MEM_GIB)
    tracker = progress.Tracker(
        invocation=invocation,
        role=role,
        argv=argv,
        timeout_s=C.RG_TIMEOUT if role == "rg" else C.CONTROL_TIMEOUT,
        gpu=gpu,
        cpu=cpu,
        memory_gib=memory,
        store=progress_store,
        journal_dir=C.PROGRESS_ROOT,
        meta={
            "app_id": app_id,
            "source_revision": source_revision,
            "run_roots": sorted(runs_touched(argv)),
        },
        gpu_sampler=progress.GpuSampler() if gpu else None,
    )
    tracker.publish()
    returncode = 1
    state = None
    server = None
    stop = threading.Event()

    def remote_lookup(call_id: str):
        return progress_store.get(f"call:{call_id}") if progress_store is not None else None

    try:
        bootstrap(role)
        checked_out = StateSync(state_volume, weights_volume)
        checked_out.tracker = tracker
        checked_out.cpu_functions = cpu_functions or {}
        checked_out.hedge_functions = hedge_functions or {}
        checked_out.options = dict(options or {})
        if options:
            log(f"execution options: {options}")
        checked_out.checkout(argv)
        state = checked_out  # only a complete checkout may be checked in (it deletes)
        if not baked():
            _synthetic_git(source_revision)
        Path(C.LOCAL_POINTER_DIR, "source_revision").write_text(source_revision + "\n")
        server = serve(state, functions)
        threading.Thread(target=_periodic_checkin, args=(state, stop), daemon=True).start()
        env = launcher_env(role, source_revision)
        tracker.begin(planned_stages(argv, env))
        tracker.start_heartbeat(remote_lookup)
        prefetch_weights(argv, env, tracker, state)
        command = ["bash", os.path.join(C.REPO_ROOT, "run_e2e.sh"), *argv]
        log(f"role={role} running: {' '.join(command)}")
        process = subprocess.Popen(
            command,
            cwd=C.REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            tracker.line(line)
        returncode = process.wait()
    finally:
        stop.set()
        if server is not None:
            server.shutdown()
        if state is not None:
            state.cancel_residents()
        active = _local_active_run()
        tracker.meta["active_run"] = active
        if state is not None:
            if active and os.path.isdir(active):
                state.add_root(active)  # e.g. a run root created by `init`
            state.checkin("final")
        tracker.finish(returncode)
        if state is not None:
            state.commit()  # the journal's last lines
    log(f"run_e2e.sh exited {returncode}")
    return returncode


def _local_active_run() -> str | None:
    try:
        value = Path(POINTER_FILE).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    root = _run_root_of(value) if value else None
    return root if root and C.under(root, RUNS_ROOT) and root != RUNS_ROOT else None


# --------------------------------------------------------------------------- batch (shared)
def _clip_roots(clips: list[dict]) -> list[str]:
    roots: set[str] = set()
    for clip in clips:
        for step in clip["steps"]:
            if C.role_for(step) != "control":
                raise ValueError(f"batch steps must run in the control role: {step}")
            roots |= runs_touched(step)
    return sorted(roots)


def run_batch_role(
    clips: list[dict],
    *,
    source_revision: str,
    state_volume,
    weights_volume,
    functions,
    progress_store=None,
    batch_id: str,
    app_id: str = "",
    cpu_functions: dict | None = None,
    hedge_functions: dict | None = None,
    options: dict | None = None,
    max_parallel: int = 4,
) -> list[dict]:
    """Run many clips' run_e2e.sh steps in this one control container.

    One StateSync, one RPC server and one GPU ledger serve every clip, so the ledger sees
    all GPU containers the batch holds: calls reuse idle containers of their image across
    clips, and new containers start only while the workspace is below its GPU cap. Each step
    still gets its own journal (invocation `<batch>-<clip>-s<n>`); the shim tags each request
    with V2D_MODAL_CALLER so module calls land in the right one.
    """
    roots = _clip_roots(clips)
    options = dict(options or {})
    state = None
    server = None
    stop = threading.Event()
    results: list[dict] = []

    def remote_lookup(call_id: str):
        return progress_store.get(f"call:{call_id}") if progress_store is not None else None

    try:
        bootstrap("control")
        checked_out = StateSync(state_volume, weights_volume)
        checked_out.cpu_functions = cpu_functions or {}
        checked_out.hedge_functions = hedge_functions or {}
        checked_out.options = options
        if checked_out.ledger is not None and options.get("scaledown_s"):
            checked_out.ledger.scaledown_s = float(options["scaledown_s"])  # the app's real window
        if checked_out.ledger is not None and options.get("scaledown_per_image"):
            checked_out.ledger.scaledown_per_image = {k: float(v) for k, v in options["scaledown_per_image"].items()}
        if checked_out.ledger is not None and options.get("gpu_budget"):
            checked_out.ledger.capacity = int(options["gpu_budget"])
        log(f"batch {batch_id}: {len(clips)} clips, {max_parallel} at a time, options {options}")
        argv = [token for root in roots for token in ("--run-root", root)]
        checked_out.checkout(["init", *argv])  # every clip's run root (runs_touched reads --run-root)
        state = checked_out
        if not baked():
            _synthetic_git(source_revision)
        Path(C.LOCAL_POINTER_DIR, "source_revision").write_text(source_revision + "\n")
        server = serve(state, functions)
        threading.Thread(target=_periodic_checkin, args=(state, stop), daemon=True).start()
        env = launcher_env("control", source_revision)
        # `setup` pip-installs host wrappers into this container's site-packages: concurrent
        # setups race on the same files (bshared smoke: "No such file ... __editable__...pth").
        setup_lock = threading.Lock()

        def run_clip(clip: dict) -> dict:
            start, code = time.time(), 0
            for index, step in enumerate(clip["steps"]):
                caller = f"{batch_id}-{clip['name']}-s{index + 1}"
                if init_done(step):
                    log(f"batch {batch_id}: {clip['name']} already initialized (restart after preemption?); skipping init")
                    continue
                tracker = progress.Tracker(
                    invocation=caller, role="control", argv=step, timeout_s=C.CONTROL_TIMEOUT,
                    gpu=None, cpu=0, memory_gib=0,  # the shared container is costed once, by the batch
                    store=progress_store, journal_dir=C.PROGRESS_ROOT,
                    meta={"app_id": app_id, "source_revision": source_revision, "batch": batch_id,
                          "run_roots": sorted(runs_touched(step))},
                )
                tracker.publish()
                state.trackers[caller] = tracker
                code = 1
                serial = C.subcommand(step) == "setup"
                if serial:
                    setup_lock.acquire()
                try:
                    tracker.begin(planned_stages(step, env))
                    tracker.start_heartbeat(remote_lookup)
                    process = subprocess.Popen(
                        ["bash", os.path.join(C.REPO_ROOT, "run_e2e.sh"), *step], cwd=C.REPO_ROOT,
                        env={**env, "V2D_MODAL_CALLER": caller}, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace",
                    )
                    assert process.stdout is not None
                    for line in process.stdout:
                        tracker.line(line)
                    code = process.wait()
                finally:
                    if serial:
                        setup_lock.release()
                    tracker.finish(code)
                    state.trackers.pop(caller, None)
                if code != 0:
                    break
            result = {"name": clip["name"], "code": code, "start": start, "end": time.time()}
            log(f"batch {batch_id}: {clip['name']} {'ok' if code == 0 else f'FAILED (exit {code})'} "
                f"in {result['end'] - start:.0f}s")
            if progress_store is not None:
                progress_store.put(f"batch:{batch_id}:{clip['name']}", result)
            return result

        with ThreadPoolExecutor(max_workers=max(1, max_parallel)) as pool:
            results = list(pool.map(run_clip, clips))
    finally:
        stop.set()
        if server is not None:
            server.shutdown()
        if state is not None:
            state.cancel_residents()
            state.checkin("final")
            state.commit()
        if state is not None and state.ledger is not None:
            log(f"batch {batch_id}: GPU budget skipped {state.ledger.skipped} speculative containers")
        log(f"batch {batch_id}: control container peak memory {_peak_memory_gib()} GiB (sizes control_batch)")
    return results


def _peak_memory_gib() -> str:
    for path in ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        try:
            return f"{int(Path(path).read_text().split()[0]) / 2**30:.1f}"
        except (OSError, ValueError, IndexError):
            continue
    return "unknown"


def init_done(step: list[str], *, mirrored: bool = False) -> bool:
    """True for an `init` step whose run root already has its configuration.

    Modal restarts a preempted function with the same input; a batch must then resume, not
    fail on "configuration already exists" (batch b4: preempted at 10 min, all 6 clips
    failed on the restart's init). Later steps resume through tools/e2e and the pipeline's
    own skip-if-done steps.
    """
    if C.subcommand(step) != "init":
        return False
    roots = runs_touched(step)
    if len(roots) != 1:
        return False
    config = os.path.join(next(iter(roots)), "e2e_config.json")
    return os.path.exists(sync.mirror_path(C.MIRROR_ROOT, config) if mirrored else config)
