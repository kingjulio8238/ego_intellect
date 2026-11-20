"""Benchmark report for a Modal pipeline run: time and cost per step, stage and module call.

    MODAL_PROFILE=<profile> python -m tools.modal_v2d.bench --run tissue_box
    MODAL_PROFILE=<profile> python -m tools.modal_v2d.bench --run tissue_box --offline   # reuse downloaded journals

Sources:
* journals that every invocation writes to the state volume (`progress/*.jsonl`): stage
  wall times, module-call timing breakdowns (startup, sync, execution), GPU utilisation
  samples, and list-price cost estimates;
* `modal billing report` (hourly, tagged per invocation): the **billed** cost of each step;
* local image-build journal (`.e2e/modal_build_logs/builds.jsonl`).

Writes to `--out` (default `.e2e/bench/<run>/`): `report.md`, `summary.json`,
`invocations.csv`, `stages.csv`, `calls.csv`, `builds.csv`, `billing.csv`, `charts.pdf`
(light theme, one chart per page), and `raw/` (the journals). Billing lags by up to a few
hours; re-run the report later for final numbers.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from tools.modal_v2d import config as C
from tools.modal_v2d.progress import hms

REPO = Path(__file__).resolve().parents[2]
PIPELINE_ORDER = ["init", "use", "status", "setup", "doctor", "reconstruct", "import-reconstruction", "inspect",
                  "retarget", "simulate", "solvability", "render", "train-expert", "collect", "finetune", "evaluate"]
V2D_APPS = {C.APP_NAME, "v2d-e2e-build"}
TAGS = ["v2d_invocation", "v2d_command", "v2d_role"]


# --------------------------------------------------------------------------- loading
def fetch_journals(raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    command = ["modal", "volume", "get", C.STATE_VOLUME, "progress", str(raw), "--force"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"could not download journals ({' '.join(command)}):\n{result.stderr or result.stdout}")


def load_invocations(raw: Path) -> dict[str, dict]:
    invocations: dict[str, dict] = {}
    for path in sorted(raw.rglob("*.jsonl")):
        if path.name == "builds.jsonl":
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            inv = invocations.setdefault(event["invocation"], {"id": event["invocation"], "stages": [], "calls": [], "begin": {}, "finish": {}})
            kind = event["kind"]
            if kind == "begin":
                inv["begin"] = event
            elif kind == "finish":
                inv["finish"] = event
            elif kind == "call_end":
                inv["calls"].append(event)
            elif kind in ("stage_done", "stage_failed", "stage_skip"):
                inv["stages"].append(event)
    return invocations


def belongs_to(inv: dict, run: str | None) -> bool:
    if not run:
        return True
    roots = list(inv["begin"].get("run_roots") or []) + [inv["finish"].get("active_run") or ""]
    return any(root.rstrip("/").endswith("/" + run) for root in roots if root)


def billing_rows(start: dt.datetime) -> tuple[list[dict], str | None]:
    """Hourly billing rows for v2d apps since ``start``; (rows, error)."""
    try:
        import modal.billing

        items = modal.billing.workspace_billing_report(start=start, resolution="h", tag_names=TAGS)
    except Exception as exc:  # reported as UNKNOWN, never as $0
        return [], f"{type(exc).__name__}: {exc}"
    rows = []
    for item in items:
        if item["description"] not in V2D_APPS:
            continue
        rows.append({
            "object_id": item["object_id"],
            "app": item["description"],
            "interval_start": item["interval_start"].isoformat(),
            "cost": float(item["cost"]),
            **{tag: (item.get("tags") or {}).get(tag, "") for tag in TAGS},
        })
    return rows, None


# --------------------------------------------------------------------------- tables
def utc_iso(ts: float | None) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ") if ts else ""


def build_tables(invocations: dict[str, dict], billing: list[dict], billing_error: str | None) -> dict:
    billed_by_inv: dict[str, float] = defaultdict(float)
    billed_by_app: dict[str, float] = defaultdict(float)
    for row in billing:
        billed_by_app[row["object_id"]] += row["cost"]
        if row["v2d_invocation"]:
            billed_by_inv[row["v2d_invocation"]] += row["cost"]

    inv_rows, stage_rows, call_rows = [], [], []
    attributed_apps = set()
    for inv in sorted(invocations.values(), key=lambda i: i["begin"].get("started", 0)):
        begin, finish = inv["begin"], inv["finish"]
        est = finish.get("cost_est")
        if est is None:  # crashed or still running: sum what was journaled
            est = sum(s.get("cost_est", 0) for s in inv["stages"])
        billed = None
        if not billing_error:
            billed = billed_by_inv.get(inv["id"])
            if billed is None and begin.get("app_id"):
                billed = billed_by_app.get(begin["app_id"])
            if begin.get("app_id"):
                attributed_apps.add(begin["app_id"])
        ratio = (billed / est) if billed and est else None
        inv_rows.append({
            "invocation": inv["id"],
            "command": begin.get("command") or C.subcommand(begin.get("argv") or ["?"]),
            "role": begin.get("role"),
            "gpu": begin.get("gpu") or "cpu",
            "started_utc": utc_iso(begin.get("started")),
            "wall_s": finish.get("seconds"),
            "status": finish.get("status", "incomplete"),
            "stages_done": len(finish.get("stages_done") or []),
            "stages_skipped": len(finish.get("stages_skipped") or []),
            "failed_stage": finish.get("failed_stage") or "",
            "module_calls": finish.get("calls", len(inv["calls"])),
            "est_cost": round(est or 0, 4),
            "billed_cost": round(billed, 4) if billed is not None else ("UNKNOWN" if billing_error else ""),
            "billed_over_est": round(ratio, 3) if ratio else "",
            "app_id": begin.get("app_id", ""),
        })
        for stage in inv["stages"]:
            cost = stage.get("cost_est")
            stage_rows.append({
                "invocation": inv["id"],
                "command": inv_rows[-1]["command"],
                "stage": stage["stage"],
                "status": stage["kind"].replace("stage_", ""),
                "started_utc": utc_iso(stage.get("started")),
                "seconds": stage.get("seconds", 0),
                "wall": hms(stage.get("seconds", 0)),
                "module_calls": stage.get("calls", 0),
                "module_seconds": stage.get("call_seconds", 0),
                "container_cost_est": stage.get("container_cost_est", 0),
                "module_cost_est": stage.get("call_cost_est", 0),
                "cost_est": cost or 0,
                # estimate scaled so the step's stages sum to its billed cost
                "cost_calibrated": round(cost * ratio, 4) if cost and ratio else "",
                "gpu_util_mean": stage.get("gpu_util_mean", ""),
                "gpu_util_max": stage.get("gpu_util_max", ""),
                "gpu_mem_max_mib": stage.get("gpu_mem_max_mib", ""),
            })
        for call in inv["calls"]:
            call_rows.append({
                "invocation": inv["id"],
                "stage": call.get("stage") or "",
                "image": call["image"],
                "call_id": call["call_id"],
                "returncode": call["returncode"],
                "started_utc": utc_iso(call.get("started")),
                "total_s": call.get("seconds"),
                "exec_s": call.get("exec_s", ""),
                "startup_s": call.get("startup_s", ""),
                "push_s": call.get("push_s", ""),
                "pull_s": call.get("pull_s", ""),
                "reload_s": call.get("reload_s", ""),
                "commit_s": call.get("commit_s", ""),
                "overhead_s": call.get("overhead_s", ""),
                "push_mb": round(call.get("push_bytes", 0) / 1e6, 2),
                "pull_mb": round(call.get("pull_bytes", 0) / 1e6, 2),
                "cost_est": call.get("cost_est", 0),
                "gpu_util_mean": call.get("gpu_util_mean", ""),
                "gpu_util_max": call.get("gpu_util_max", ""),
                "gpu_mem_max_mib": call.get("gpu_mem_max_mib", ""),
            })

    unattributed = defaultdict(float)
    for row in billing:
        if row["v2d_invocation"] in billed_by_inv and row["v2d_invocation"] in invocations:
            continue
        if row["object_id"] in attributed_apps:
            continue
        label = row["v2d_command"] or ("image builds" if row["app"] == "v2d-e2e-build" else "untagged v2d-e2e runs")
        unattributed[label] += row["cost"]
    return {"invocations": inv_rows, "stages": stage_rows, "calls": call_rows, "unattributed": dict(unattributed)}


def load_builds() -> list[dict]:
    path = REPO / ".e2e" / "modal_build_logs" / "builds.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        row["when_utc"] = utc_iso(row.pop("ts", None))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- report
def md_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_none_\n"
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines) + "\n"


def money(value) -> str:
    return f"${value:.2f}" if isinstance(value, (int, float)) else str(value)


def summarize(tables: dict, billing_error: str | None, billing_total: float | None) -> dict:
    by_command: dict[str, dict] = {}
    for row in tables["invocations"]:
        entry = by_command.setdefault(row["command"], {"command": row["command"], "invocations": 0, "wall_s": 0.0, "est_cost": 0.0, "billed_cost": 0.0, "billed_known": True})
        entry["invocations"] += 1
        entry["wall_s"] += row["wall_s"] or 0
        entry["est_cost"] += row["est_cost"] or 0
        if isinstance(row["billed_cost"], (int, float)):
            entry["billed_cost"] += row["billed_cost"]
        else:
            entry["billed_known"] = False
    steps = sorted(by_command.values(), key=lambda e: PIPELINE_ORDER.index(e["command"]) if e["command"] in PIPELINE_ORDER else 99)
    calls = tables["calls"]
    exec_total = sum(c["exec_s"] or 0 for c in calls if c["exec_s"] != "")
    total_calls = sum(c["total_s"] or 0 for c in calls)
    return {
        "generated_utc": utc_iso(dt.datetime.now(dt.timezone.utc).timestamp()),
        "steps": steps,
        "wall_s_total": sum(e["wall_s"] for e in steps),
        "est_cost_total": round(sum(e["est_cost"] for e in steps), 4),
        "billed_cost_total": None if billing_error else round(sum(e["billed_cost"] for e in steps), 4),
        "billed_v2d_total_in_window": billing_total,
        "unattributed_billing": tables["unattributed"],
        "billing_error": billing_error,
        "module_calls": len(calls),
        "module_exec_share": round(exec_total / total_calls, 3) if total_calls else None,
    }


def write_report(out: Path, run: str | None, tables: dict, summary: dict, builds: list[dict]) -> None:
    steps = [
        {**s, "wall": hms(s["wall_s"]), "est": money(s["est_cost"]), "billed": money(s["billed_cost"]) if s["billed_known"] else "UNKNOWN"}
        for s in summary["steps"]
    ]
    lines = [f"# Modal benchmark: {run or 'all runs'}", "", f"Generated {summary['generated_utc']}.", ""]
    if summary["billing_error"]:
        lines += [f"> **Billing UNKNOWN**: {summary['billing_error']}. Costs below are list-price estimates only.", ""]
    else:
        lines += ["> Billing is hourly and lags by up to a few hours; re-run this report later for final numbers.", ""]
    lines += [
        f"- Wall time across steps: **{hms(summary['wall_s_total'])}**",
        f"- Estimated cost: **{money(summary['est_cost_total'])}**; billed (tagged to these steps): **{money(summary['billed_cost_total']) if summary['billed_cost_total'] is not None else 'UNKNOWN'}**",
        f"- Other v2d billing in the window (builds, doctor, other runs): {', '.join(f'{k} {money(v)}' for k, v in summary['unattributed_billing'].items()) or 'none'}",
        f"- Module calls: {summary['module_calls']}; share of call time spent executing (vs startup and sync): {summary['module_exec_share'] if summary['module_exec_share'] is not None else 'n/a'}",
        "",
        "## Per pipeline step",
        "",
        md_table(steps, ["command", "invocations", "wall", "est", "billed"]),
        "## Per stage",
        "",
        md_table(tables["stages"], ["command", "stage", "status", "wall", "module_calls", "cost_est", "cost_calibrated", "gpu_util_mean", "gpu_mem_max_mib"]),
        "## Module calls (slowest first)",
        "",
        md_table(sorted(tables["calls"], key=lambda c: -(c["total_s"] or 0)), ["image", "stage", "total_s", "exec_s", "startup_s", "push_s", "pull_s", "push_mb", "cost_est", "gpu_util_mean", "gpu_mem_max_mib"]),
        "## Invocations",
        "",
        md_table(tables["invocations"], ["invocation", "command", "gpu", "started_utc", "wall_s", "status", "stages_done", "stages_skipped", "module_calls", "est_cost", "billed_cost", "billed_over_est"]),
        "## Image builds",
        "",
        md_table(builds, ["when_utc", "image", "attempt", "ok", "seconds", "cached"]),
    ]
    low = [c for c in tables["calls"] if isinstance(c["gpu_util_mean"], (int, float)) and c["gpu_util_mean"] < 30 and (c["exec_s"] or 0) > 60]
    if low:
        lines += ["## Flags", "", "Module calls over 60 s with mean GPU utilisation under 30% (a smaller GPU may do):", ""]
        lines += [f"- {c['image']} ({c['stage']}): {c['gpu_util_mean']}% mean, {c['exec_s']} s" for c in low]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_charts(out: Path, tables: dict, summary: dict) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError as exc:
        return f"charts skipped: {exc}"
    plt.style.use("default")  # light theme: white surface, dark ink
    path = out / "charts.pdf"
    with PdfPages(path) as pdf:
        stages = [s for s in tables["stages"] if s["status"] != "skip"]
        if stages:
            fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(stages) + 1)))
            labels = [s["stage"] for s in stages]
            ax.barh(labels, [s["seconds"] / 60 for s in stages], color="#3b6ea5")
            ax.invert_yaxis()
            ax.set_xlabel("minutes")
            ax.set_title("Wall time per stage")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        steps = summary["steps"]
        if steps:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            x = range(len(steps))
            ax.bar([i - 0.2 for i in x], [s["est_cost"] for s in steps], width=0.4, label="estimate", color="#9bb7d4")
            ax.bar([i + 0.2 for i in x], [s["billed_cost"] if s["billed_known"] else 0 for s in steps], width=0.4, label="billed", color="#2f5d8a")
            ax.set_xticks(list(x), [s["command"] for s in steps], rotation=30, ha="right")
            ax.set_ylabel("USD")
            ax.set_title("Cost per pipeline step")
            ax.legend()
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        calls = [c for c in tables["calls"] if c["exec_s"] != ""]
        if calls:
            fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(calls) + 1)))
            labels = [f"{c['image']} · {c['stage']}" for c in calls]
            execs = [c["exec_s"] / 60 for c in calls]
            startup = [(c["startup_s"] or 0) / 60 for c in calls]
            sync_ = [((c["push_s"] or 0) + (c["pull_s"] or 0) + (c["reload_s"] or 0) + (c["commit_s"] or 0)) / 60 for c in calls]
            ax.barh(labels, execs, label="execute", color="#2f5d8a")
            ax.barh(labels, startup, left=execs, label="startup (schedule, cold start)", color="#d08c3a")
            ax.barh(labels, sync_, left=[a + b for a, b in zip(execs, startup)], label="volume sync", color="#b9b9b9")
            ax.invert_yaxis()
            ax.set_xlabel("minutes")
            ax.set_title("Module calls: execution vs overhead")
            ax.legend(loc="lower right")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        gpu = [r for r in tables["stages"] + tables["calls"] if isinstance(r.get("gpu_util_mean"), (int, float))]
        if gpu:
            fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(gpu) + 1)))
            labels = [r.get("stage") if "image" not in r else f"{r['image']} · {r['stage']}" for r in gpu]
            ax.barh(labels, [r["gpu_util_max"] for r in gpu], color="#c9d8e8", label="peak")
            ax.barh(labels, [r["gpu_util_mean"] for r in gpu], color="#2f5d8a", label="mean")
            ax.invert_yaxis()
            ax.set_xlim(0, 100)
            ax.set_xlabel("GPU utilisation (%)")
            ax.set_title("GPU utilisation per stage and module call")
            ax.legend(loc="lower right")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="run name (the last path component of the run root), e.g. tissue_box")
    parser.add_argument("--out", type=Path, help="output directory (default .e2e/bench/<run>)")
    parser.add_argument("--offline", action="store_true", help="reuse journals already in <out>/raw; skip billing")
    args = parser.parse_args()
    if not os.environ.get("MODAL_PROFILE") and not args.offline:
        raise SystemExit("set MODAL_PROFILE explicitly (never rely on the globally active profile)")
    out = args.out or REPO / ".e2e" / "bench" / (args.run or "all")
    raw = out / "raw"
    out.mkdir(parents=True, exist_ok=True)
    if not args.offline:
        fetch_journals(raw)
    invocations = {k: v for k, v in load_invocations(raw).items() if belongs_to(v, args.run)}
    if not invocations:
        print(f"no journaled invocations for run {args.run!r} in {raw}", file=sys.stderr)
        return 1
    first = min(v["begin"].get("started", 0) for v in invocations.values())
    start = dt.datetime.fromtimestamp(first, dt.timezone.utc).replace(minute=0, second=0, microsecond=0, tzinfo=None)
    if args.offline:
        billing, billing_error = [], "offline mode (billing not queried)"
    else:
        billing, billing_error = billing_rows(start)
    tables = build_tables(invocations, billing, billing_error)
    billing_total = None if billing_error else round(sum(r["cost"] for r in billing), 4)
    summary = summarize(tables, billing_error, billing_total)
    builds = load_builds()
    write_csv(out / "invocations.csv", tables["invocations"])
    write_csv(out / "stages.csv", tables["stages"])
    write_csv(out / "calls.csv", tables["calls"])
    write_csv(out / "builds.csv", builds)
    write_csv(out / "billing.csv", billing)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(out, args.run, tables, summary, builds)
    chart_note = write_charts(out, tables, summary)
    print((out / "report.md").read_text())
    if chart_note:
        print(chart_note)
    print(f"[v2d-bench] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
