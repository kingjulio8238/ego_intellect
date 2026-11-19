"""Split one batch manifest across Modal workspaces and launch a batch on each.

    python -m tools.modal_v2d.batch_shard manifest.json --profiles sales-32662,newacc \
        --usd-per-clip 0.6 [--cap 25] [--max-parallel 4] [--dry-run]

Each workspace has its own GPU concurrency cap (10 on Modal Starter) and its own monthly
credit, so throughput and budget both scale with the number of prepared workspaces
(images built, weights on `v2d-weights`, inputs on `v2d-e2e`). Clips are assigned in
proportion to each workspace's headroom below `--cap` (month-to-date spend, queried with the
profile pinned; a failed query stops the launch: never treated as $0). A workspace whose
headroom cannot cover its share at `--usd-per-clip` gets fewer clips or none.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def month_to_date(profile: str) -> float:
    env = {**os.environ, "MODAL_PROFILE": profile}
    result = subprocess.run(["modal", "billing", "report", "--for", "this month", "--json"],
                            capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise SystemExit(f"{profile}: billing query failed (spend UNKNOWN, not $0):\n{result.stderr.strip()}")
    rows = json.loads(result.stdout)
    return sum(float(row.get("cost", row.get("Cost"))) for row in rows)


def assign(clips: list[dict], headroom: dict[str, float], usd_per_clip: float) -> dict[str, list[dict]]:
    """Clips in proportion to headroom (more workspaces in parallel = more GPUs), never more
    than a workspace's headroom covers at `usd_per_clip`."""
    capacity = {profile: max(0, int(room // usd_per_clip)) for profile, room in headroom.items()}
    if sum(capacity.values()) < len(clips):
        raise SystemExit(f"not enough headroom for {len(clips)} clips at ${usd_per_clip}/clip: {headroom}")
    total = sum(max(room, 0) for room in headroom.values())
    counts = {profile: min(capacity[profile], int(len(clips) * max(room, 0) / total)) for profile, room in headroom.items()}
    while sum(counts.values()) < len(clips):  # hand out the remainder where capacity is left
        profile = max((p for p in counts if counts[p] < capacity[p]), key=lambda p: headroom[p] - counts[p] * usd_per_clip)
        counts[profile] += 1
    shards, index = {}, 0
    for profile, count in counts.items():
        if count:
            shards[profile] = clips[index:index + count]
            index += count
    return shards


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--usd-per-clip", type=float, required=True, help="projected cost per clip (from a smoke batch)")
    parser.add_argument("--cap", type=float, default=25.0, help="stop launching at this month-to-date spend")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    headroom = {}
    for profile in args.profiles.split(","):
        spent = month_to_date(profile)
        headroom[profile] = args.cap - spent
        print(f"{profile}: MTD ${spent:.2f}, headroom ${headroom[profile]:.2f} below ${args.cap:.0f}")
    shards = assign(manifest["clips"], headroom, args.usd_per_clip)
    out = REPO / ".e2e" / "batch" / "shards"
    out.mkdir(parents=True, exist_ok=True)
    procs = []
    for profile, clips in shards.items():
        path = out / f"{args.manifest.stem}.{profile}.json"
        path.write_text(json.dumps({**manifest, "clips": clips}, indent=1))
        print(f"{profile}: {len(clips)} clips, projected ${len(clips) * args.usd_per_clip:.2f} "
              f"(headroom ${headroom[profile]:.2f}) -> {path}")
        if args.dry_run:
            continue
        log = out / f"{args.manifest.stem}.{profile}.log"
        env = {**os.environ, "MODAL_PROFILE": profile}
        procs.append(subprocess.Popen(
            [str(REPO / "tools/modal_v2d/run_e2e_modal.sh"), "--batch", str(path), "--max-parallel", str(args.max_parallel)],
            cwd=REPO, env=env, stdout=log.open("w"), stderr=subprocess.STDOUT))
        print(f"  launched (log {log})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
