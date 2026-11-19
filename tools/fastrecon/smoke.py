"""Fan out reconstruction speed-up smokes as isolated call replays on Modal Sandboxes.

    MODAL_PROFILE=<p> python -m tools.fastrecon.smoke --run tissue_box_stock3 \
        [--only NAME ...] [--parallel 12] [--dry-run] [--budget 5]

Each smoke = (one recorded stock call, one strategy, one hardware choice). Every call that
has a strategy also gets a CONTROL replay (unmodified call, same harness): it measures the
harness's own baseline time and the call's run-to-run noise, so a strategy is judged
against its control, not against production (whose timing includes volume/queue overhead).

Outputs: .e2e/fastrecon/<run>/smokes.jsonl (raw) and smokes.csv (the board).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal

from tools.fastrecon.strategies import STRATEGIES, Strategy
from tools.modal_v2d import config as C
from tools.modal_v2d.progress import hourly_rate

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT_ROOT = REPO / ".e2e" / "fastrecon"
REPLAY_ROOT = "/vol/replay"
CPU, MEM_GIB = 8, 32
APP_NAME = "v2d-fastrecon"


def log(message: str) -> None:
    print(f"[fastrecon {time.strftime('%H:%M:%SZ', time.gmtime())}] {message}", flush=True)


# --- recorded stock calls ------------------------------------------------------------
def fetch_journal(run: str, out: Path, refresh: bool = False) -> Path:
    """Download the newest reconstruct journal of <run> from /vol/progress.

    A journal already downloaded is used as is (it is immutable once the run ends), so the
    harness also works from a workspace that holds only the pack (e.g. after a move).
    """
    if not refresh:
        for local in sorted(out.glob("*.jsonl"), reverse=True):
            text = local.read_text()
            if f"/e2e/runs/{run}/" in text and '"call_spec"' in text:
                return local
    listing = subprocess.run(
        ["modal", "volume", "ls", C.STATE_VOLUME, "progress"], capture_output=True, text=True, check=True
    ).stdout.split()
    candidates = sorted(name for name in listing if name.endswith(".jsonl"))
    for name in reversed(candidates):
        local = out / Path(name).name
        if refresh or not local.exists():
            subprocess.run(["modal", "volume", "get", "--force", C.STATE_VOLUME, name, str(local)],
                           check=True, capture_output=True)
        text = local.read_text()
        if f"/e2e/runs/{run}/" in text and '"call_spec"' in text:
            return local
    raise SystemExit(f"no journal with call_spec records for run {run}; was it run with the recording runtime?")


def load_calls(journal: Path) -> list[dict]:
    calls: dict[str, dict] = {}
    order: list[str] = []
    for line in journal.read_text().splitlines():
        record = json.loads(line)
        cid = record.get("call_id")
        if not cid:
            continue
        entry = calls.setdefault(cid, {"call_id": cid})
        if record["kind"] == "call_spec":
            entry["spec"], entry["on_cpu"] = record["spec"], record.get("on_cpu", False)
            order.append(cid)
        elif record["kind"] == "call_outputs":
            entry["outputs"] = record["changed"]
        elif record["kind"] == "call_end":
            entry["end"] = record
    out = []
    for index, cid in enumerate(order):
        entry = calls[cid]
        argv = entry["spec"]["argv"]
        entry["index"] = index
        entry["image"] = entry["spec"]["image"]
        entry["module"] = argv[2] if len(argv) > 2 and argv[1] == "-m" else " ".join(argv[:3])
        out.append(entry)
    return out


def merge_calls(first: dict, second: dict, merge: dict) -> dict:
    """A synthetic recorded call that runs two recorded calls' work in one module process
    (e.g. FoundationPose scale + tracking): union of mounts and outputs, argv = new module
    + first call's args + second call's args minus `drop_second`."""
    mounts = {m["dst"]: m for m in first["spec"]["mounts"]}
    for m in second["spec"]["mounts"]:
        if m["dst"] in mounts and mounts[m["dst"]]["src"] != m["src"]:
            raise SystemExit(f"merge: mount {m['dst']} differs between the two calls")
        mounts.setdefault(m["dst"], m)
    def args_of(argv: list[str]) -> list[str]:
        return argv[3:] if argv[1:2] == ["-m"] else argv[1:]
    second_args, drop, i = args_of(second["spec"]["argv"]), set(merge.get("drop_second", ())), 0
    kept: list[str] = []
    while i < len(second_args):
        token = second_args[i]
        has_value = i + 1 < len(second_args) and not second_args[i + 1].startswith("--")
        if token in drop or token in args_of(first["spec"]["argv"]):
            i += 2 if has_value else 1
            continue
        kept += second_args[i : i + (2 if has_value else 1)]
        i += 2 if has_value else 1
    argv = [first["spec"]["argv"][0], "-m", merge["new_module"], *args_of(first["spec"]["argv"]), *kept]
    env = {**(second["spec"].get("env") or {}), **(first["spec"].get("env") or {})}
    return {
        **first,
        "spec": {**first["spec"], "argv": argv, "mounts": list(mounts.values()), "env": env},
        "outputs": sorted(set(first["outputs"]) | set(second["outputs"])),
        "end": {"exec_s": (first.get("end") or {}).get("exec_s", 0) + (second.get("end") or {}).get("exec_s", 0)},
    }


def find_call(calls: list[dict], image: str, module: str, nth: int = 0) -> dict | None:
    hits = [c for c in calls if c["image"] == image and c["module"].endswith(module) and "outputs" in c]
    return hits[nth] if len(hits) > nth else None


# --- Modal plumbing -------------------------------------------------------------------
def pack(app, run: str, run_root_vol: str, pack_path: str) -> None:
    script = f"""
set -e
mkdir -p {REPLAY_ROOT}
if [ -s {pack_path} ]; then echo "pack exists"; exit 0; fi
tar -cf {pack_path}.tmp -C {run_root_vol} --exclude='*overlay*.mp4' --exclude='progress_simple.mp4' --exclude='refined_simple_progress' .
mv {pack_path}.tmp {pack_path}
ls -la {pack_path}
"""
    vol = modal.Volume.from_name(C.STATE_VOLUME)
    sb = modal.Sandbox.create("bash", "-c", script, image=modal.Image.debian_slim(), app=app,
                              timeout=1800, cpu=4, memory=8192, volumes={"/vol": vol})
    sb.wait()
    log(f"pack: {sb.stdout.read().strip()} {sb.stderr.read()[-300:]}")
    if sb.returncode != 0:
        raise SystemExit("pack failed")


def runner_image(images: dict, name: str):
    return (
        images[name]
        .add_local_file(str(HERE / "replay_runner.py"), "/opt/fastrecon/replay_runner.py")
        .add_local_file(str(HERE / "outdiff.py"), "/opt/fastrecon/outdiff.py")
        .add_local_file(str(HERE.parent / "modal_v2d" / "zygote.py"), "/opt/fastrecon/zygote.py")
    )


def upload_jobs(jobs: list[dict], tmp: Path) -> None:
    """Job files go through the volume: env vars cap at 32 KB and output lists are longer."""
    tmp.mkdir(parents=True, exist_ok=True)
    vol = modal.Volume.from_name(C.STATE_VOLUME)
    with vol.batch_upload(force=True) as batch:
        for job in jobs:
            local = tmp / f"{job['smoke']}.json"
            local.write_text(json.dumps(job))
            batch.put_file(str(local), f"replay/jobs/{job['smoke']}.json")


def run_job(app, image, job: dict, gpu: str | None, timeout: int) -> dict:
    script = (
        "set -e; command -v python >/dev/null && PY=python || PY=python3; "
        f"$PY /opt/fastrecon/replay_runner.py {REPLAY_ROOT}/jobs/{job['smoke']}.json"
    )
    vols = {"/vol": modal.Volume.from_name(C.STATE_VOLUME), C.WEIGHTS_MOUNT: modal.Volume.from_name(C.WEIGHTS_VOLUME)}
    secrets = []
    t0 = time.time()
    sb = modal.Sandbox.create(
        "bash", "-c", script, image=image, app=app, gpu=gpu, cpu=CPU, memory=MEM_GIB * 1024,
        timeout=timeout, volumes=vols, secrets=secrets,
    )
    sb.wait()
    wall = time.time() - t0
    stdout, stderr = sb.stdout.read(), sb.stderr.read()
    result = {"smoke": job["smoke"], "rc": sb.returncode, "wall_s": round(wall, 1)}
    for line in stdout.splitlines():
        if line.startswith("FASTRECON_RESULT "):
            result.update(json.loads(line[len("FASTRECON_RESULT "):]))
    if "exec_s" not in result:
        result["error"] = (stdout[-800:] + "\n" + stderr[-800:]).strip()
    result["cost"] = round(wall / 3600 * hourly_rate(gpu, CPU, MEM_GIB), 4)
    return result


# --- verdicts -------------------------------------------------------------------------
TOLERANCE_KEYS = ("json_max_abs", "img_mask_err_max", "img_mean_rel_max", "arr_max_abs", "obj_max_vertex")


def verdict(row: dict, control: dict | None) -> str:
    if row.get("rc") not in (0, None) or "exec_s" not in row:
        return "FAILED"
    diff = row.get("diff") or {}
    if diff.get("missing"):
        return f"MISSING {diff['missing']} outputs"
    if diff.get("files") and diff.get("identical") == diff.get("files"):
        return "identical"
    if row["name"].endswith(":control"):
        return "control (nondeterministic)" if diff else "control"
    cdiff = (control or {}).get("diff") or {}
    worse = [k for k in TOLERANCE_KEYS if diff.get(k, 0.0) > max(2.0 * cdiff.get(k, 0.0), 1e-6)]
    struct = diff.get("json_struct_diff", 0) + diff.get("img_shape_diff", 0) + diff.get("arr_struct_diff", 0)
    if struct:
        return f"STRUCT DIFF ({struct})"
    return "within noise" if not worse else "OUTSIDE noise: " + ",".join(worse)


def all_rows(raw_path: Path) -> list[dict]:
    """Every smoke ever recorded for this run, newest result per name wins."""
    latest: dict[str, dict] = {}
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            row = json.loads(line)
            latest[row["name"]] = row
    return list(latest.values())


def write_board(rows: list[dict], path: Path) -> None:
    controls = {r["call"]: r for r in rows if r["name"].endswith(":control")}
    fields = ["name", "call", "hw", "verdict", "exec_s", "ctrl_exec_s", "speedup", "prod_exec_s",
              "gpu_util_mean", "wall_s", "cost", "diff"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["call"], not r["name"].endswith(":control"), r["name"])):
            control = controls.get(row["call"])
            ctrl = (control or {}).get("exec_s")
            writer.writerow({
                "name": row["name"], "call": row["call"], "hw": row["hw"],
                "verdict": verdict(row, control),
                "exec_s": row.get("exec_s"), "ctrl_exec_s": ctrl,
                "speedup": round(ctrl / row["exec_s"], 2) if ctrl and row.get("exec_s") else "",
                "prod_exec_s": row.get("prod_exec_s"), "gpu_util_mean": row.get("gpu_util_mean"),
                "wall_s": row.get("wall_s"), "cost": row.get("cost"),
                "diff": json.dumps({k: v for k, v in (row.get("diff") or {}).items() if v}, separators=(",", ":"))[:300],
            })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--only", nargs="*", help="strategy names (prefix match)")
    parser.add_argument("--parallel", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=1800, help="server-side cap per sandbox (s)")
    parser.add_argument("--budget", type=float, default=5.0, help="refuse to launch above this projected $")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-controls", action="store_true")
    parser.add_argument("--pack-suffix", default="", help="separate pack (e.g. for a harness test mid-run)")
    parser.add_argument("--refresh-journal", action="store_true", help="re-download the journal")
    parser.add_argument("--show-builds", action="store_true", help="stream Modal image-build logs")
    args = parser.parse_args()

    out = OUT_ROOT / args.run
    out.mkdir(parents=True, exist_ok=True)
    journal = fetch_journal(args.run, out, refresh=args.refresh_journal)
    calls = load_calls(journal)
    log(f"{len(calls)} recorded calls in {journal.name}")
    run_root_vol = f"{C.MIRROR_ROOT}/e2e/runs/{args.run}/reconstruction"
    pack_path = f"{REPLAY_ROOT}/{args.run}{args.pack_suffix}.tar"

    chosen: list[Strategy] = [s for s in STRATEGIES if not args.only or any(s.name.startswith(o) for o in args.only)]
    jobs = []
    seen_controls = set()
    for strat in chosen:
        call = find_call(calls, strat.image, strat.module, strat.nth)
        merge = (strat.patch or {}).get("merge")
        if call is not None and merge:
            second = find_call(calls, merge["image"], merge["module"], merge.get("nth", 0))
            call = merge_calls(call, second, merge) if second else None
        if call is None:
            log(f"skip {strat.name}: no recorded call {strat.image} {strat.module}")
            continue
        key = f"{strat.image}:{strat.module.rsplit('.', 1)[-1]}#{strat.nth}"
        if not args.no_controls and key not in seen_controls:
            seen_controls.add(key)
            jobs.append((Strategy(f"{key}:control", strat.image, strat.module, strat.nth, {}, gpu="L40S"), call, key))
        jobs.append((strat, call, key))
    projected = 0.0
    for strat, call, _ in jobs:
        prod = (call.get("end") or {}).get("exec_s") or 120
        projected += (prod * strat.time_factor + 90) / 3600 * hourly_rate(strat.gpu, CPU, MEM_GIB)
    log(f"{len(jobs)} smokes, projected ${projected:.2f} (budget ${args.budget:.2f})")
    for strat, call, key in jobs:
        log(f"  {strat.name:44s} {strat.gpu or 'cpu':6s} call {call['index']:2d} {call['image']} {call['module']}")
    if args.dry_run:
        return 0
    if projected > args.budget:
        raise SystemExit("projected cost exceeds --budget")

    from tools.modal_v2d.images import build_images  # noqa: PLC0415 - heavy, only when launching

    if args.show_builds:
        modal.enable_output().__enter__()  # image builds otherwise run silently for minutes
    app = modal.App.lookup(APP_NAME, create_if_missing=True)
    pack(app, args.run, run_root_vol, pack_path)
    images = build_images()
    built: dict[str, object] = {}
    lock = threading.Lock()

    def image_for(name: str):
        with lock:
            if name not in built:
                built[name] = runner_image(images, name)
            return built[name]

    rows: list[dict] = []
    raw = (out / "smokes.jsonl").open("a")
    board = out / "smokes.csv"

    def make_job(strat, call) -> dict:
        return {
            "smoke": strat.name.replace(":", "_").replace("/", "_"),
            "pack": pack_path,
            "run_root": run_root_vol,
            "spec": call["spec"],
            "outputs": call["outputs"],
            "strategy": {k: v for k, v in (strat.patch or {}).items() if k != "merge"},
        }

    upload_jobs([make_job(s, c) for s, c, _ in jobs], out / "jobs")

    def launch(item):
        strat, call, key = item
        job = make_job(strat, call)
        result = run_job(app, image_for(strat.image), job, strat.gpu, args.timeout)
        result.update({"name": strat.name, "call": key, "hw": strat.gpu or "cpu",
                       "prod_exec_s": (call.get("end") or {}).get("exec_s")})
        return result

    spent = 0.0
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(launch, item): item for item in jobs}
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001
                strat, call, key = futures[future]
                row = {"name": strat.name, "call": key, "hw": strat.gpu or "cpu", "rc": -1,
                       "error": f"{type(exc).__name__}: {exc}"[:500]}
            rows.append(row)
            spent += row.get("cost") or 0.0
            raw.write(json.dumps(row) + "\n")
            raw.flush()
            write_board(all_rows(out / "smokes.jsonl"), board)
            log(f"done {row['name']:44s} rc={row.get('rc')} exec={row.get('exec_s')}s "
                f"gpu={row.get('gpu_util_mean')}% ${row.get('cost')} | {len(rows)}/{len(jobs)} spent ${spent:.2f}"
                + (f" | ERROR {row.get('error', '')[:160]}" if row.get("error") else ""))
    log(f"board: {board}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
