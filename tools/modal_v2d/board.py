"""Status board for Modal pipeline runs: one row per invocation, plus real billing.

    MODAL_PROFILE=<profile> python -m tools.modal_v2d.board            # print once
    MODAL_PROFILE=<profile> python -m tools.modal_v2d.board --watch     # refresh every 60 s
    MODAL_PROFILE=<profile> python -m tools.modal_v2d.board --journal <invocation>

Rows come from the live snapshots that role containers publish (``inv:<id>`` in the
progress Dict). ``$ est`` is a list-price estimate; the header's month-to-date figure is
`modal billing report` for the workspace, and a failed billing query is shown as UNKNOWN,
never as $0.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from tools.modal_v2d import config as C
from tools.modal_v2d.progress import hms, utc

STALE_S = 3 * C.HEARTBEAT_S


def billing() -> str:
    command = ["modal", "billing", "report", "--for", "this month", "--json"]
    result = subprocess.run(command, capture_output=True, text=True, env=os.environ)
    if result.returncode != 0:
        return f"MTD UNKNOWN (billing query failed: {(result.stderr or result.stdout).strip()[-160:]})"
    try:
        rows = json.loads(result.stdout)
        total = sum(float(row.get("cost", row.get("Cost"))) for row in rows)
    except (ValueError, TypeError) as exc:
        return f"MTD UNKNOWN (could not parse billing output: {exc})"
    state = f"OVER by ${total - 30:.2f}" if total > 30 else f"headroom ${30 - total:.2f}"
    return f"MTD ${total:.2f} ({state} of $30 credit)"


def row(snapshot: dict, now: float) -> list[str]:
    status = snapshot.get("status", "?")
    if status == "running" and now - snapshot.get("updated", 0) > STALE_S:
        status = f"STALE {hms(now - snapshot['updated'])}"
    finished, total = snapshot.get("stages_finished", 0), snapshot.get("stages_total")
    stage = snapshot.get("stage")
    phase = stage or (snapshot.get("failed_stage") and f"failed at {snapshot['failed_stage']}") or status
    call = snapshot.get("call")
    if call:
        phase += f" › {call.get('image')}"
    progress = f"{finished}/{total or '?'} stages"
    inner = snapshot.get("inner")
    if inner:
        progress += f" · {inner[0]}/{inner[1]}"
    end = snapshot.get("updated", now) if status in ("succeeded", "failed") else now
    progress += f" · {hms(end - snapshot.get('started', now))}"
    last = (call or {}).get("remote_last") or snapshot.get("last_line") or ""
    result = status if status in ("succeeded", "failed") or status.startswith("STALE") else (last[-60:] or "…")
    eta = snapshot.get("eta")
    eta_text = utc(eta, seconds=False) if eta and status == "running" else ("—" if status != "running" else f"cap {utc(snapshot.get('deadline'), seconds=False)}")
    return [
        f"{snapshot.get('command', '?')} [{snapshot.get('invocation', '?')}]",
        phase,
        progress,
        result,
        eta_text,
        f"~${snapshot.get('cost_est', 0):.2f}",
    ]


def render(store, limit: int, include_done: bool) -> str:
    now = time.time()
    snapshots = [value for key, value in store.items() if str(key).startswith("inv:")]
    snapshots.sort(key=lambda s: s.get("started", 0), reverse=True)
    if not include_done:
        recent = [s for s in snapshots if s.get("status") == "running" or now - s.get("updated", 0) < 6 * 3600]
        snapshots = recent or snapshots[:1]
    rows = [["job", "phase", "progress", "result so far", "ETA (UTC)", "$ est"]]
    rows += [row(s, now) for s in snapshots[:limit]]
    widths = [min(max(len(r[i]) for r in rows), 64) for i in range(len(rows[0]))]
    lines = [f"[v2d-board {utc()}] profile {os.environ.get('MODAL_PROFILE')} | {billing()}"]
    for index, r in enumerate(rows):
        lines.append(" | ".join(cell[:64].ljust(widths[i]) for i, cell in enumerate(r)))
        if index == 0:
            lines.append("-+-".join("-" * w for w in widths))
    if len(rows) == 1:
        lines.append("(no runs recorded yet)")
    return "\n".join(lines)


def journal(invocation: str) -> int:
    path = "progress/"
    listing = subprocess.run(["modal", "volume", "ls", C.STATE_VOLUME, path, "--json"], capture_output=True, text=True)
    if listing.returncode != 0:
        print(listing.stderr, file=sys.stderr)
        return 1
    names = [entry.get("Filename") or entry.get("filename") or entry.get("path", "") for entry in json.loads(listing.stdout)]
    matches = [name for name in names if invocation in name]
    if not matches:
        print(f"no journal for {invocation} under {C.STATE_VOLUME}:{path}", file=sys.stderr)
        return 1
    for name in matches:
        subprocess.run(["modal", "volume", "get", C.STATE_VOLUME, name, "-", "--force"], check=False)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--watch", nargs="?", const=60, type=int, help="refresh every N seconds (default 60)")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--all", action="store_true", help="include runs finished more than 6 h ago")
    parser.add_argument("--journal", metavar="INVOCATION", help="print the event journal of one invocation")
    args = parser.parse_args()
    if not os.environ.get("MODAL_PROFILE"):
        raise SystemExit("set MODAL_PROFILE explicitly (never rely on the globally active profile)")
    if args.journal:
        return journal(args.journal)

    import modal

    store = modal.Dict.from_name(C.PROGRESS_DICT, create_if_missing=True)
    while True:
        print(render(store, args.limit, args.all), flush=True)
        if not args.watch:
            return 0
        time.sleep(args.watch)
        print()


if __name__ == "__main__":
    sys.exit(main())
