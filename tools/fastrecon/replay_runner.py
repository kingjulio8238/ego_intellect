"""Replay one recorded module call inside its module image (runs in a Modal Sandbox).

    python /opt/fastrecon/replay_runner.py <job.json>

job = {
  smoke, pack (tar of the stock run's reconstruction dir on the volume), run_root (that dir's
  volume path), spec (journal call_spec: image/argv/env/workdir/mounts), outputs (journal
  call_outputs.changed: volume paths the stock call wrote), strategy {set, add, env}
}

1. Extract the stock run locally (local NVMe; no network-volume I/O during the call).
2. Move the stock call's outputs aside as the reference (the call must recompute them).
3. Bind the recorded mounts to the local copy with symlinks, apply the strategy, run.
4. Diff the new outputs against the reference (outdiff.compare_trees); print one JSON line.
Stdlib + numpy only: it runs under each module image's own Python.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import outdiff  # noqa: E402

STAGE, REF = "/tmp/fr_stage", "/tmp/fr_ref"


def _read(path: str) -> str | None:
    try:
        with open(path) as stream:
            return stream.read().strip()
    except FileNotFoundError:
        return None


class GpuSampler(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.util: list[float] = []
        self.mem: list[float] = []
        self.stop = threading.Event()

    def run(self) -> None:
        if shutil.which("nvidia-smi") is None:
            return
        while not self.stop.wait(1.0):
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip().splitlines()[0]
                u, m = (float(x) for x in out.split(","))
                self.util.append(u)
                self.mem.append(m)
            except Exception:  # noqa: BLE001
                pass


def apply_strategy(argv: list[str], strategy: dict) -> list[str]:
    argv = list(argv)
    for flag, value in (strategy.get("set") or {}).items():
        if flag in argv:
            i = argv.index(flag)
            has_value = i + 1 < len(argv) and not argv[i + 1].startswith("--")
            del argv[i : i + (2 if has_value else 1)]
        if value is not None:
            argv += [flag] if value is True else [flag, str(value)]
    argv += list(strategy.get("add") or [])
    return argv


def main() -> int:
    job = json.load(open(sys.argv[1]))
    spec, run_root = job["spec"], job["run_root"].rstrip("/")
    t0 = time.time()
    os.makedirs(STAGE, exist_ok=True)
    with tarfile.open(job["pack"]) as tar:
        tar.extractall(STAGE)
    staged = time.time()

    def local(path: str) -> str:
        return STAGE + path[len(run_root):] if path == run_root or path.startswith(run_root + "/") else path

    rel_outputs = []
    for path in job["outputs"]:
        if not path.startswith(run_root + "/"):
            continue
        rel = path[len(run_root) + 1 :]
        rel_outputs.append(rel)
        src = os.path.join(STAGE, rel)
        if os.path.lexists(src):
            os.makedirs(os.path.dirname(os.path.join(REF, rel)), exist_ok=True)
            shutil.move(src, os.path.join(REF, rel))

    for mount in spec["mounts"]:
        dst, src = mount["dst"], local(mount["src"])
        os.makedirs(src, exist_ok=True) if not os.path.exists(src) else None
        if os.path.lexists(dst):
            os.rename(dst, dst + ".fr_aside")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.symlink(src, dst)

    strategy = job.get("strategy") or {}
    argv = apply_strategy(spec["argv"], strategy)
    profiling = bool(strategy.get("profile")) and argv[1:2] == ["-m"]
    if profiling:  # python -m cProfile -o F -m <module> ...: host-side hot spots
        argv = [argv[0], "-m", "cProfile", "-o", "/tmp/fr_prof.out", *argv[1:]]
    env = dict(os.environ)
    recorded = _read("/etc/v2d_image_pythonpath")
    if recorded is not None:
        env["PYTHONPATH"] = recorded
        if not recorded:
            env.pop("PYTHONPATH")
    env.setdefault("PYTHONUTF8", "1")
    env.update(spec.get("env") or {})
    env.update(strategy.get("env") or {})
    workdir = spec.get("workdir") or _read("/etc/v2d_image_workdir") or "/"

    sampler = GpuSampler()
    sampler.start()
    log_path = f"/tmp/{job['smoke']}.log"
    preload_s = None
    if strategy.get("resident") and argv[1:2] == ["-m"]:
        # Resident execution (tools/modal_v2d/zygote.py): preload first, untimed, as it
        # would overlap upstream work in a run; then time only the call itself.
        zenv = {**env, "V2D_ZYGOTE_ARGV_HINT": json.dumps(argv)}
        t_pre = time.time()
        with open(log_path, "w") as log:
            zygote = subprocess.Popen([argv[0], "/opt/fastrecon/zygote.py", argv[2]], cwd=workdir, env=zenv,
                                      stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, text=True)
            while zygote.poll() is None:
                with open(log_path, errors="replace") as stream:
                    if "[v2d-zygote] READY" in stream.read():
                        break
                time.sleep(0.2)
            preload_s = round(time.time() - t_pre, 1)
            started = time.time()
            zygote.stdin.write(json.dumps({"argv": argv}) + "\n")
            zygote.stdin.flush()
            rc = zygote.wait()
    else:
        started = time.time()
        with open(log_path, "w") as log:
            rc = subprocess.call(argv, cwd=workdir, env=env, stdout=log, stderr=subprocess.STDOUT)
    exec_s = time.time() - started
    sampler.stop.set()

    diff = outdiff.compare_trees(REF, STAGE, rel_outputs) if rc == 0 else {}
    if rc == 0 and strategy.get("keep"):  # keep this replay's outputs for cross-replay statistics
        keep_root = f"/vol/replay/out/{job['smoke']}"
        for rel in rel_outputs:
            src = os.path.join(STAGE, rel)
            if os.path.isfile(src):
                os.makedirs(os.path.dirname(os.path.join(keep_root, rel)), exist_ok=True)
                shutil.copy2(src, os.path.join(keep_root, rel))
    with open(log_path, errors="replace") as stream:
        tail = stream.read()[-1500:]
    try:  # full log kept on the volume for profiling (sandbox volume writes persist on exit)
        os.makedirs("/vol/replay/logs", exist_ok=True)
        shutil.copy(log_path, f"/vol/replay/logs/{job['smoke']}.log")
    except OSError:
        pass
    profile_top = None
    if profiling and os.path.exists("/tmp/fr_prof.out"):
        import io
        import pstats
        buf = io.StringIO()
        pstats.Stats("/tmp/fr_prof.out", stream=buf).sort_stats("cumulative").print_stats(45)
        profile_top = buf.getvalue()[-9000:]
        try:
            shutil.copy("/tmp/fr_prof.out", f"/vol/replay/logs/{job['smoke']}.prof")
        except OSError:
            pass
    result = {
        "smoke": job["smoke"],
        "rc": rc,
        "stage_s": round(staged - t0, 1),
        "exec_s": round(exec_s, 1),
        "preload_s": preload_s,
        "gpu_util_mean": round(sum(sampler.util) / len(sampler.util), 1) if sampler.util else None,
        "gpu_mem_max_mib": max(sampler.mem) if sampler.mem else None,
        "argv": argv,
        "diff": diff,
        "log_tail": tail,
        "profile_top": profile_top,
    }
    print("FASTRECON_RESULT " + json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
