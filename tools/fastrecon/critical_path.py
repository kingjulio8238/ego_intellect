"""Critical path of the --fast reconstruction DAG from measured call times (no runs needed).

    python -m tools.fastrecon.critical_path --run tissue_box_stock3 [--overhead 15] [--use smokes]

Durations come from the smoke board (best within-noise variant per call) or the stock
journal. The DAG is the data dependency graph of run_ego_wilor (diagnostic calls removed):
it answers "which calls are worth speeding up" — only the ones on the critical path cut
latency; the rest only cut cost.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# node: (call key prefix as in smokes.csv `call`, dependencies)
DAG = {
    "anycalib": ("v2d_anycalib:", []),
    "moge": ("v2d_moge:", ["anycalib"]),
    "geocalib": ("v2d_geocalib:", ["anycalib"]),
    "droid": ("v2d_droid_slam:", ["moge"]),
    "wilor_ref": ("v2d_wilor:video_to_hands", ["anycalib"]),
    "gdino": ("v2d_grounding_dino:", ["anycalib"]),
    "prompt_masks": ("v2d_wilor:render_prompt_masks", ["wilor_ref"]),
    "sam2": ("v2d_sam2:", ["prompt_masks", "gdino"]),
    "sam3d": ("v2d_sam3d:", ["sam2", "moge"]),
    "fp_scale": ("v2d_foundation_pose:estimate_mesh_scale", ["sam3d"]),
    "fp_track": ("v2d_foundation_pose:video_to_poses", ["fp_scale"]),
    "hamer": ("v2d_hamer:masks_to_hands", ["sam2"]),
    "align": ("v2d_hamer:align_hands", ["hamer", "moge"]),
    "interp": ("v2d_wilor:tracks_interpolate", ["align"]),
    "gsplat": ("v2d_gsplat_refinement:", ["fp_track", "interp", "droid"]),
    "package": (None, ["gsplat", "geocalib"]),
}


def schedule(durations: dict[str, float], overhead: float) -> tuple[float, list[str], dict]:
    finish: dict[str, float] = {}
    parent: dict[str, str | None] = {}
    for node in DAG:  # insertion order is topological
        deps = DAG[node][1]
        start = max((finish[d] for d in deps), default=0.0)
        parent[node] = max(deps, key=lambda d: finish[d]) if deps else None
        cost = durations.get(node, 0.0)
        finish[node] = start + cost + (overhead if cost else 0.0)
    path, node = [], "package"
    while node:
        path.append(node)
        node = parent[node]
    return finish["package"], path[::-1], finish


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", type=Path, required=True, help="smokes.csv")
    parser.add_argument("--overhead", type=float, default=15.0, help="per-call dispatch overhead (s)")
    args = parser.parse_args()
    rows = list(csv.DictReader(args.board.open()))
    stock, best = {}, {}
    for node, (prefix, _) in DAG.items():
        if not prefix:
            continue
        mine = [r for r in rows if r["call"].startswith(prefix.split(":")[0]) and prefix.split(":")[1] in r["call"]]
        ctrl = [r for r in mine if r["name"].endswith(":control") and r["exec_s"]]
        if ctrl:
            stock[node] = float(ctrl[0]["exec_s"])
        ok = [r for r in mine if r["exec_s"] and (r["verdict"] in ("identical", "within noise") or r["name"].endswith(":control"))]
        if ok:
            pick = min(ok, key=lambda r: float(r["exec_s"]))
            best[node] = (float(pick["exec_s"]), pick["name"])
    for label, durations in (("stock calls, DAG-parallel", stock), ("best smokes, DAG-parallel", {k: v[0] for k, v in best.items()})):
        total, path, _ = schedule(durations, args.overhead)
        print(f"{label}: {total / 60:.1f} min | critical path: {' -> '.join(path)}")
    print("serial sum of stock calls:", round(sum(stock.values()) / 60, 1), "min")
    print(json.dumps({k: v[1] for k, v in best.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
