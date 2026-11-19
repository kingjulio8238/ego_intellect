"""Remote side of a shimmed `docker run`: execute one container spec in this image.

Runs inside a module-image (or robotic-grounding-image) Modal function. The spec's
mounts already point at volume paths (or at this image's own repo copy); this module
symlinks them into place, runs the command exactly as docker would have, reports which
mirrored files changed, and commits the volumes. Stdlib only, plus the volume handles
passed in by app.py.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import uuid

from tools.modal_v2d import config as C
from tools.modal_v2d import containerenv, mounts, progress, sync


def _tee(process: subprocess.Popen, log_path: str, reporter: progress.CallReporter) -> None:
    """Raw output to the call log; timestamped, image-tagged output to the console."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            reporter.line(line)
        process.stdout.close()
    sys.stdout.flush()


def _inductor_cache_in(image: str, env: dict) -> str | None:
    """Seed TORCHINDUCTOR_CACHE_DIR from the weights volume (compiled graphs are reused,
    byte-identical to a fresh compile; only the compile time goes away)."""
    if "TORCHINDUCTOR_CACHE_DIR" in env:
        return None
    cache = os.path.join(C.INDUCTOR_CACHE_ROOT, image)
    if os.path.isdir(cache):
        shutil.copytree(cache, C.LOCAL_INDUCTOR_DIR, dirs_exist_ok=True)
    os.makedirs(C.LOCAL_INDUCTOR_DIR, exist_ok=True)
    env["TORCHINDUCTOR_CACHE_DIR"] = C.LOCAL_INDUCTOR_DIR
    return cache


def _inductor_cache_out(cache: str | None) -> None:
    if not cache or not os.path.isdir(C.LOCAL_INDUCTOR_DIR) or not os.listdir(C.LOCAL_INDUCTOR_DIR):
        return
    try:
        shutil.copytree(C.LOCAL_INDUCTOR_DIR, cache, dirs_exist_ok=True)
    except OSError as exc:
        print(f"[v2d-worker] could not save inductor cache: {exc}", flush=True)


def _ext_cache_in(image: str, env: dict) -> str | None:
    """Seed TORCH_EXTENSIONS_DIR from the per-image cache on the weights volume.

    Images that build CUDA extensions at first use (gsplat in SAM3D and gsplat_refinement)
    would otherwise recompile them in every fresh container (HOME=/tmp is ephemeral).
    """
    if "TORCH_EXTENSIONS_DIR" in env:
        return None
    cache = os.path.join(C.EXT_CACHE_ROOT, image)
    local = C.LOCAL_EXT_DIR
    if os.path.isdir(cache):
        shutil.copytree(cache, local, dirs_exist_ok=True, ignore=shutil.ignore_patterns("lock"))
    os.makedirs(local, exist_ok=True)
    env["TORCH_EXTENSIONS_DIR"] = local
    return cache


def _ext_cache_out(cache: str | None) -> None:
    """Save newly built extensions back to the volume (successful calls only)."""
    if not cache or not os.path.isdir(C.LOCAL_EXT_DIR) or not os.listdir(C.LOCAL_EXT_DIR):
        return
    try:
        shutil.copytree(C.LOCAL_EXT_DIR, cache, dirs_exist_ok=True, ignore=shutil.ignore_patterns("lock"))
    except OSError as exc:  # a cache miss next time is the only consequence
        print(f"[v2d-worker] could not save torch extension cache: {exc}", flush=True)


PREFETCH_MAX_BYTES = 2 << 30
PREFETCH_THREADS = 32


def _prefetch_inputs(spec: dict) -> dict:
    """Read every file the command's arguments point into, 32 at a time, before it runs.

    Module code reads its inputs one small file at a time; on the network volume each
    first read is a round trip (~10-25 ms), so a call over a few thousand JSON/PNG files
    spends minutes waiting (tracks_interpolate: 65 s in production vs ~1 s from local
    disk). Parallel reads fill the local cache first; results are unchanged.
    """
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    t0 = time.time()
    dst_to_src = {m["dst"].rstrip("/"): m["src"] for m in spec["mounts"] if m.get("mirrored")}
    paths: list[str] = []
    for token in spec["argv"]:
        for dst, src in dst_to_src.items():
            if token == dst or token.startswith(dst + "/"):
                local = src + token[len(dst):]
                if os.path.isfile(local):
                    paths.append(local)
                elif os.path.isdir(local) and os.path.realpath(local) != os.path.realpath(src):
                    for root, _, files in os.walk(local):
                        paths.extend(os.path.join(root, f) for f in files)
    paths = sorted(set(paths))
    total = 0
    selected = []
    for path in paths:
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        if total + size > PREFETCH_MAX_BYTES:
            break
        total += size
        selected.append(path)

    def read(path: str) -> None:
        try:
            with open(path, "rb") as stream:
                while stream.read(1 << 20):
                    pass
        except OSError:
            pass

    with ThreadPoolExecutor(PREFETCH_THREADS) as pool:
        list(pool.map(read, selected))
    return {"prefetch_files": len(selected), "prefetch_mb": round(total / 1e6, 1), "prefetch_s": round(time.time() - t0, 3)}


def _run_subprocess(spec: dict, env: dict, workdir: str, log_path: str, reporter) -> int:
    process = subprocess.Popen(
        spec["argv"],
        cwd=workdir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        errors="replace",
    )
    _tee(process, log_path, reporter)
    return process.wait()


def container_age_s() -> float | None:
    """Seconds since this container's first process started. At call entry, a small value
    means the container just booted (the value is its boot time: image load + runtime start);
    a large one means it was already warm."""
    try:
        with open("/proc/uptime", encoding="ascii") as stream:
            uptime = float(stream.read().split()[0])
        with open("/proc/1/stat", encoding="ascii") as stream:
            start_ticks = float(stream.read().rsplit(")", 1)[1].split()[19])
        return round(uptime - start_ticks / os.sysconf("SC_CLK_TCK"), 3)
    except (OSError, ValueError, IndexError):
        return None


def execute(spec: dict, *, state_volume, weights_volume, progress_store=None, resident=None) -> dict:
    """Run ``spec`` = {image, argv, env, workdir, mounts:[{src,dst,mirrored}], call_id}.

    ``resident``: a started, preloaded Zygote whose env/workdir match this spec; the call
    then runs inside it instead of in a fresh process (see serve_resident).
    """
    started = time.time()
    age_at_entry = container_age_s()
    call_id = spec.get("call_id") or uuid.uuid4().hex[:12]
    image = spec["image"]
    if resident is None:
        resident = _take_warm(spec)  # this container preloaded the module after its last call
    sampler = progress.GpuSampler()
    sampler.open("call")
    state_volume.reload()
    if resident is None and not _WARM:  # a preloaded process holds weight files open; weights never change mid-run
        weights_volume.reload()
    reloaded = time.time()

    mirrored_roots = [m["src"] for m in spec["mounts"] if m.get("mirrored")]
    before = sync.snapshot(mirrored_roots)

    if resident is None:
        env = containerenv.container_env(spec.get("env") or {}, image)
        ext_cache = _ext_cache_in(image, env)
        inductor_cache = _inductor_cache_in(image, env)
    else:
        env, ext_cache, inductor_cache = resident.env, resident.ext_cache, resident.inductor_cache
    workdir = spec.get("workdir") or containerenv.image_workdir()
    log_path = f"{C.CALL_LOG_ROOT}/{time.strftime('%Y%m%dT%H%M%S')}-{image}-{call_id}.log"

    reporter = progress.CallReporter(image=image, call_id=call_id, store=progress_store)
    reporter.publish("starting")
    print(f"[{image} {progress.utc()}] ▶ call {call_id}: {' '.join(spec['argv'])}", flush=True)
    prefetch = _prefetch_inputs(spec) if os.environ.get("V2D_MODAL_PREFETCH_INPUTS", "1") != "0" else {}
    applied = mounts.apply([{"src": m["src"], "dst": m["dst"]} for m in spec["mounts"]])
    exec_started = time.time()
    try:
        if workdir and not os.path.isdir(workdir):
            raise FileNotFoundError(f"workdir does not exist in {image}: {workdir}")
        if resident is None:
            returncode = _run_subprocess(spec, env, workdir, log_path, reporter)
        else:
            returncode = resident.run(spec["argv"], log_path, reporter)
    finally:
        exec_finished = time.time()
        mounts.revert(applied)
        gpu_stats = sampler.close("call")
        sampler.stop()

    if returncode == 0:
        _ext_cache_out(ext_cache)
        _inductor_cache_out(inductor_cache)
    changed, deleted, new_dirs = sync.diff(before, sync.snapshot(mirrored_roots))
    diffed = time.time()
    state_volume.commit()
    try:
        weights_volume.commit()
    except Exception as exc:  # noqa: BLE001  - only the torch-extension cache is written there
        if resident is None and not _WARM:
            raise
        print(f"[v2d-worker] weights commit skipped (resident process holds files open): {exc}", flush=True)
    elapsed = time.time() - started
    if returncode == 0 and spec.get("keep_warm") and not spec.get("exit_after"):
        _preload_next(spec)
    if spec.get("exit_after"):  # probes: release the GPU now instead of after the scaledown window
        try:
            import modal.experimental  # noqa: PLC0415

            modal.experimental.stop_fetching_inputs()
        except Exception:  # noqa: BLE001
            pass
    timing = {
        "worker_s": round(elapsed, 3),
        "reload_s": round(reloaded - started, 3),
        "exec_s": round(exec_finished - exec_started, 3),
        "diff_s": round(diffed - exec_finished, 3),
        "commit_s": round(time.time() - diffed, 3),
        "changed_files": len(changed),
        "container_age_s": age_at_entry,
        **({"resident": True, "preload_s": resident.preload_s, "resident_wait_s": resident.wait_s} if resident else {}),
        **prefetch,
        **gpu_stats,
    }
    reporter.publish("succeeded" if returncode == 0 else "failed", returncode, wait=True)
    print(
        f"[{image} {progress.utc()}] {'✔' if returncode == 0 else '✘'} call {call_id} exit={returncode} "
        f"in {progress.hms(elapsed)}; {len(changed)} changed, {len(deleted)} deleted files",
        flush=True,
    )
    return {
        "returncode": returncode,
        "changed": changed,
        "deleted": deleted,
        "dirs": new_dirs,
        "log": log_path,
        "elapsed_s": elapsed,
        "timing": timing,
    }


# --------------------------------------------------------------------------- warm models
# Batch mode (`keep_warm`): after a call of a resident module succeeds, this container
# starts preloading the same module for the next call (a new process, so each call still
# runs in a fresh interpreter, as with residents). A container reused by the next clip then
# skips imports and model construction (R3 smokes: SAM3D -30 s, FoundationPose -16/-13 s,
# WiLoR -15 s, DINO -6 s). module -> Zygote; one container serves one call at a time.
_WARM: dict[str, "Zygote"] = {}


def _module_of(spec: dict) -> str | None:
    argv = spec.get("argv") or []
    return argv[2] if len(argv) > 2 and argv[1] == "-m" else None


def _preload_next(spec: dict) -> None:
    module = _module_of(spec)
    if module not in C.RESIDENT_MODULES or module in _WARM:
        return
    hint = {"argv": list(spec["argv"]), "env": dict(spec.get("env") or {}), "workdir": spec.get("workdir"),
            "mounts": [m for m in spec.get("mounts") or [] if not m.get("mirrored")]}
    try:
        _WARM[module] = Zygote(spec["image"], hint)
        print(f"[{spec['image']} {progress.utc()}] preloading {module} for this container's next call", flush=True)
    except Exception as exc:  # noqa: BLE001  - a missed preload only costs the load time
        print(f"[{spec['image']} {progress.utc()}] preload of {module} failed: {exc}", flush=True)


def _take_warm(spec: dict) -> "Zygote | None":
    module = _module_of(spec)
    zygote = _WARM.pop(module, None) if module else None
    if zygote is None:
        return None
    if zygote.matches(spec):
        zygote.wait_s = 0.0
        return zygote
    zygote.kill()
    return None


# --------------------------------------------------------------------------- resident calls
class Zygote:
    """A preloaded process for one upcoming call (tools/modal_v2d/zygote.py)."""

    def __init__(self, image: str, hint: dict) -> None:
        import json  # noqa: PLC0415
        import threading  # noqa: PLC0415

        self.hint = hint
        self.module = hint["argv"][2]
        # Static mounts (weights on the weights volume) must be in place before the preload
        # reads them; the call's own mounts.apply then finds them already correct.
        self.static_mounts = [m for m in hint.get("mounts") or []
                              if not m.get("mirrored") and C.under(m["src"], C.WEIGHTS_MOUNT)]
        mounts.apply([{"src": m["src"], "dst": m["dst"]} for m in self.static_mounts])
        self.env = containerenv.container_env(hint.get("env") or {}, image)
        self.ext_cache = _ext_cache_in(image, self.env)
        self.inductor_cache = _inductor_cache_in(image, self.env)
        self.workdir = hint.get("workdir") or containerenv.image_workdir()
        self.preload_s = None
        self.wait_s = None
        self.ready = threading.Event()
        self._sink = None  # (log stream, reporter) once the call runs
        self._lock = threading.Lock()
        self.t0 = time.time()
        zenv = {**self.env, "V2D_ZYGOTE_ARGV_HINT": json.dumps(hint["argv"]), "PYTHONUNBUFFERED": "1"}
        self.process = subprocess.Popen(
            [hint["argv"][0], os.path.join(os.path.dirname(os.path.abspath(__file__)), "zygote.py"), self.module],
            cwd=self.workdir, env=zenv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace",
        )
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            if not self.ready.is_set():
                if line.startswith("\x1e[v2d-zygote] READY"):
                    self.preload_s = round(time.time() - self.t0, 3)
                    self.ready.set()
                    continue
                print(f"[preload {self.module}] {line}", end="", flush=True)
                continue
            with self._lock:
                if self._sink is not None:
                    self._sink[0].write(line)
                    self._sink[1].line(line)
        self.ready.set()  # died before READY: run() sees the exit and reports it

    def matches(self, spec: dict) -> bool:
        workdir = spec.get("workdir") or containerenv.image_workdir()
        spec_mounts = {m["dst"].rstrip("/"): m["src"] for m in spec.get("mounts") or []}
        return (
            all(spec_mounts.get(m["dst"].rstrip("/")) == m["src"] for m in self.static_mounts)
            and spec["argv"][:3] == self.hint["argv"][:3]
            and (spec.get("env") or {}) == (self.hint.get("env") or {})
            and workdir == self.workdir
            and self.process.poll() is None
        )

    def run(self, argv: list[str], log_path: str, reporter) -> int:
        import json  # noqa: PLC0415

        self.ready.wait()
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8", errors="replace") as log:
            with self._lock:
                self._sink = (log, reporter)
            if self.process.poll() is None:
                assert self.process.stdin is not None
                self.process.stdin.write(json.dumps({"argv": argv}) + "\n")
                self.process.stdin.flush()
            code = self.process.wait()
            self._reader.join(timeout=30)
            with self._lock:
                self._sink = None
        return code

    def kill(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()


def serve_resident(request: dict, *, state_volume, weights_volume, progress_store=None) -> dict:
    """Preload `request["resident"]`'s call, wait for the real spec on the resident queue,
    run it in the preloaded process. Falls back to a normal subprocess run when the spec
    does not match the preload (different env/workdir/module) or the preload died.

    Returns execute()'s result, or {"unclaimed": True} when no spec arrived in time.
    """
    import modal  # noqa: PLC0415

    hint = request["resident"]
    image = request["image"]
    zygote = Zygote(image, hint)
    queue = modal.Queue.from_name(C.RESIDENT_QUEUE, create_if_missing=True)
    t_wait = time.time()
    try:
        spec = queue.get(partition=hint["key"], timeout=C.RESIDENT_WAIT_S)
    except Exception:  # noqa: BLE001  - modal raises queue.Empty on timeout (final2 traceback)
        spec = None
    if spec is None:
        zygote.kill()
        print(f"[{image} {progress.utc()}] resident {zygote.module}: no call within {C.RESIDENT_WAIT_S}s; exiting", flush=True)
        return {"unclaimed": True}
    zygote.wait_s = round(time.time() - t_wait, 3)
    zygote.ready.wait()
    if not zygote.matches(spec):
        print(f"[{image} {progress.utc()}] resident {zygote.module}: call does not match the preload "
              f"(or preload died); running it fresh", flush=True)
        zygote.kill()
        return execute(spec, state_volume=state_volume, weights_volume=weights_volume, progress_store=progress_store)
    return execute(spec, state_volume=state_volume, weights_volume=weights_volume,
                   progress_store=progress_store, resident=zygote)


# --------------------------------------------------------------------------- hedged calls
def claim(spec: dict) -> bool:
    """First container to start a hedged call owns it (atomic Dict put). The loser stops
    taking inputs so its container exits instead of idling through the scaledown window."""
    import modal  # noqa: PLC0415

    lane = spec.get("lane", "primary")
    delay = float(spec.get("debug_delay_claim_s") or 0)
    if delay and lane == "primary":  # tests only: simulate a primary that starts late
        time.sleep(delay)
    claims = modal.Dict.from_name(C.CLAIM_DICT, create_if_missing=True)
    won = claims.put(spec["claim"], {"lane": lane, "t": time.time()}, skip_if_exists=True)
    if not won:
        print(f"[{spec.get('image')} {progress.utc()}] call {spec.get('call_id')}: already running on "
              f"another container ({lane} lane lost the claim); exiting", flush=True)
        try:
            import modal.experimental  # noqa: PLC0415

            modal.experimental.stop_fetching_inputs()
        except Exception:  # noqa: BLE001
            pass
    return bool(won)
