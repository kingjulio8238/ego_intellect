"""Throughput and cost of one batch (tools/modal_v2d/batch.py), from the run journals.

    MODAL_PROFILE=<profile> python -m tools.modal_v2d.batch_report <batch_id> [--offline]

Per clip: wall, GPU calls, GPU execution seconds, container start waits (startup_s: Modal
scheduling + boot), GPU-slot waits (our own queue), hedges. Batch: makespan, clips/hour,
start-wait p50/p95/max, estimated $ per clip (list prices) and billed $ (tagged billing,
hourly, lags up to a few hours; UNKNOWN when the query fails, never $0).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
from pathlib import Path

from tools.modal_v2d import bench
from tools.modal_v2d import config as C

REPO = Path(__file__).resolve().parents[2]


def _pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    index = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
    return values[index]


def clip_rows(invocations: dict[str, dict], batch_id: str) -> list[dict]:
    clips: dict[str, list[dict]] = {}
    for inv_id, inv in invocations.items():
        if not inv_id.startswith(batch_id + "-"):
            continue
        name = inv_id[len(batch_id) + 1:].rsplit("-s", 1)[0]
        clips.setdefault(name, []).append(inv)
    rows = []
    for name, invs in sorted(clips.items()):
        begins = [i["begin"].get("ts") for i in invs if i["begin"].get("ts")]
        ends = [i["finish"].get("ts") for i in invs if i["finish"].get("ts")]
        calls = [c for i in invs for c in i["calls"] if c.get("gpu")]
        failed = [i["id"] for i in invs if i["finish"].get("returncode") not in (0, None)]
        control_cost = sum(
            ((i["finish"].get("ts") or 0) - (i["begin"].get("ts") or 0)) / 3600 * float(i["begin"].get("rate_per_h") or 0)
            for i in invs if i["begin"].get("ts") and i["finish"].get("ts")
        )
        recon = [i for i in invs if i["begin"].get("command") == "reconstruct"]
        rows.append({
            "clip": name,
            "ok": not failed and len(ends) == len(invs),
            "wall_s": round(max(ends) - min(begins), 1) if begins and ends else None,
            "reconstruct_s": round(recon[0]["finish"]["ts"] - recon[0]["begin"]["ts"], 1) if recon and recon[0]["finish"].get("ts") else None,
            "gpu_calls": len(calls),
            "gpu_exec_s": round(sum(c.get("exec_s") or 0 for c in calls), 1),
            "gpu_call_s": round(sum(c.get("seconds") or 0 for c in calls), 1),
            "start_wait_sum_s": round(sum(c.get("startup_s") or 0 for c in calls), 1),
            "start_wait_max_s": round(max([c.get("startup_s") or 0 for c in calls] or [0]), 1),
            "slot_wait_sum_s": round(sum(c.get("slot_wait_s") or 0 for c in calls), 1),
            "sync_sum_s": round(sum((c.get("push_s") or 0) + (c.get("pull_s") or 0) for c in calls), 1),
            "worker_overhead_s": round(sum((c.get("worker_s") or 0) - (c.get("exec_s") or 0) for c in calls), 1),
            "hedged": sum(bool(c.get("hedged")) for c in calls),
            "hedge_won": sum(c.get("lane") == "hedge" for c in calls),
            "est_cost": round(sum(c.get("cost_est") or 0 for c in calls) + control_cost, 3),
            "start": min(begins) if begins else None,
            "end": max(ends) if ends else None,
            "_waits": [c.get("startup_s") or 0 for c in calls],
        })
    return rows


def image_rows(invocations: dict[str, dict], batch_id: str) -> str:
    """Execution time per module image and GPU architecture (hedges may run on H100)."""
    groups: dict[tuple[str, str], list[float]] = {}
    for inv_id, inv in invocations.items():
        if not inv_id.startswith(batch_id + "-"):
            continue
        for call in inv["calls"]:
            if call.get("gpu"):
                key = (call.get("image", "?"), f"sm{call.get('gpu_sm') or '?'}")
                groups.setdefault(key, []).append(call.get("exec_s") or 0)
    lines = ["| image | gpu | calls | exec mean s | exec max s |", "|---|---|---|---|---|"]
    for (image, sm), values in sorted(groups.items()):
        lines.append(f"| {image} | {sm} | {len(values)} | {statistics.mean(values):.1f} | {max(values):.1f} |")
    return "\n".join(lines)


def billed(batch_id: str, since: dt.datetime) -> tuple[float | None, str | None]:
    rows, error = bench.billing_rows(since)
    if error:
        return None, error
    mine = [r for r in rows if r.get("v2d_invocation") == batch_id]
    if not mine:
        return None, "no billing rows yet (hourly, lags up to a few hours)"
    return round(sum(r["cost"] for r in mine), 4), None


def report(rows: list[dict], billed_total: float | None, billing_error: str | None) -> str:
    done = [r for r in rows if r["start"] and r["end"]]
    if not done:
        return "no finished clips in the journals"
    makespan = max(r["end"] for r in done) - min(r["start"] for r in done)
    ok = [r for r in rows if r["ok"]]
    waits = [w for r in rows for w in r["_waits"]]
    cols = ["clip", "ok", "wall_s", "reconstruct_s", "gpu_calls", "gpu_exec_s", "start_wait_sum_s",
            "start_wait_max_s", "slot_wait_sum_s", "sync_sum_s", "worker_overhead_s", "hedged", "hedge_won", "est_cost"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    est = sum(r["est_cost"] for r in rows)
    lines += [
        "",
        f"makespan {makespan / 60:.1f} min | {len(ok)}/{len(rows)} ok | {len(ok) / makespan * 3600:.1f} clips/h",
        f"container start wait per call: p50 {_pct(waits, 0.5):.1f}s p95 {_pct(waits, 0.95):.1f}s max {max(waits or [0]):.1f}s "
        f"(n={len(waits)})",
        f"GPU exec per clip: {statistics.mean(r['gpu_exec_s'] for r in rows):.0f}s | "
        f"est $/clip {est / max(len(rows), 1):.3f} (list price, calls + control)",
        "billed: " + (f"${billed_total:.3f} total, ${billed_total / max(len(rows), 1):.3f}/clip" if billed_total is not None
                      else f"UNKNOWN ({billing_error})"),
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("batch_id")
    parser.add_argument("--offline", action="store_true", help="reuse downloaded journals; skip billing")
    args = parser.parse_args()
    out = REPO / ".e2e" / "batch" / args.batch_id
    raw = out / "raw"
    if not args.offline:
        bench.fetch_journals(raw)
    rows = clip_rows(bench.load_invocations(raw), args.batch_id)
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)
    total, error = (None, "offline") if args.offline else billed(args.batch_id, since)
    text = report(rows, total, error) + "\n\n" + image_rows(bench.load_invocations(raw), args.batch_id)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(f"# Batch {args.batch_id}\n\n{text}\n")
    (out / "clips.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "_waits"} for r in rows], indent=1))
    print(text)
    print(f"\n→ {out}/report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
