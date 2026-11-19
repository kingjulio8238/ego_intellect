"""Run-to-run spread of one call's outputs across replays kept with patch {"keep": True}.

    python -m tools.fastrecon.variance <dir with one subdir per replay> [--groups stock fast]

For gsplat refinement (same inputs every replay), per pair of replays:
  camera_rot_deg_p95 / camera_trans_mm_p95  (camera_refined_simple, same world frame)
  obj_rot_deg_p95 / obj_trans_mm_p95        (poses_refined_simple)
  hand_trans_mm_p95                         (hamer_refined_simple camera.cam_t)
  hand_rot_deg_p95                          (global_orient, axis-angle)
  betas_l2_mean                             (per-frame betas L2, averaged)
Pairs are grouped as within-group (e.g. stock-stock, fast-fast) and across groups
(stock-fast): "no degradation" means across-group spread ~ within-stock spread.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


def _load_dir(root: Path, sub: str) -> dict[str, dict]:
    """Parallel reads: on a network volume each small-file read is a round trip."""
    from concurrent.futures import ThreadPoolExecutor

    base = root / sub
    paths = sorted(base.rglob("*.json"))
    with ThreadPoolExecutor(64) as pool:
        texts = list(pool.map(lambda path: path.read_text(), paths))
    return {str(path.relative_to(base)): json.loads(text) for path, text in zip(paths, texts)}


def _quat_wxyz(q):
    w, x, y, z = q
    return R.from_quat([x, y, z, w])


def _p95(values) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), 95)) if len(values) else float("nan")


def load(root: Path) -> dict:
    return {
        "camera": _load_dir(root, "camera_refined_simple"),
        "object": _load_dir(root, "poses_refined_simple"),
        "hands": _load_dir(root, "hamer_refined_simple"),
    }


def pair_metrics(a: dict, b: dict) -> dict:
    m: dict[str, float] = {}
    for key, label in (("camera", "camera"), ("object", "obj")):
        rot, trans = [], []
        for name, ja in a[key].items():
            jb = b[key].get(name)
            if jb is None:
                continue
            ra, rb = _quat_wxyz(ja["rotation"]), _quat_wxyz(jb["rotation"])
            rot.append(np.degrees((ra.inv() * rb).magnitude()))
            trans.append(1000 * np.linalg.norm(np.subtract(ja["translation"], jb["translation"])))
        m[f"{label}_rot_deg_p95"] = _p95(rot)
        m[f"{label}_trans_mm_p95"] = _p95(trans)
    ht, hr, betas = [], [], []
    for name, ja in a["hands"].items():
        jb = b["hands"].get(name)
        if jb is None:
            continue
        ht.append(1000 * np.linalg.norm(np.subtract(ja["camera"]["cam_t"], jb["camera"]["cam_t"])))
        ga = R.from_rotvec(np.asarray(ja["mano"]["global_orient"], dtype=np.float64).reshape(-1, 3)[0])
        gb = R.from_rotvec(np.asarray(jb["mano"]["global_orient"], dtype=np.float64).reshape(-1, 3)[0])
        hr.append(np.degrees((ga.inv() * gb).magnitude()))
        betas.append(np.linalg.norm(np.subtract(ja["mano"]["betas"], jb["mano"]["betas"])))
    m["hand_trans_mm_p95"] = _p95(ht)
    m["hand_rot_deg_p95"] = _p95(hr)
    m["betas_l2_mean"] = float(np.mean(betas)) if betas else float("nan")
    return m


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path, nargs="+", help="directories holding one subdir per replay")
    p.add_argument("--only-prefix", nargs="*", help="only replay dirs with these name prefixes")
    p.add_argument("--groups", nargs="+", default=["stock", "fast"])
    p.add_argument("--json", type=Path)
    a = p.parse_args()
    dirs = [d for root in a.root for d in sorted(root.iterdir()) if d.is_dir() and not d.name.startswith(".")]
    if a.only_prefix:
        dirs = [d for d in dirs if any(d.name.startswith(x) for x in a.only_prefix)]
    runs = {d.name: load(d) for d in dirs}
    group_of = {name: next((g for g in a.groups if f"-{g}-" in name or name.endswith(g)), "other") for name in runs}
    buckets: dict[str, list[dict]] = {}
    for x, y in itertools.combinations(sorted(runs), 2):
        gx, gy = sorted((group_of[x], group_of[y]))
        buckets.setdefault(f"{gx}-{gy}", []).append(pair_metrics(runs[x], runs[y]))
    keys = list(next(iter(buckets.values()))[0])
    print(f"{'metric':22s}" + "".join(f"{b:>26s}" for b in sorted(buckets)))
    summary = {}
    for k in keys:
        row = f"{k:22s}"
        for b in sorted(buckets):
            vals = [d[k] for d in buckets[b]]
            summary.setdefault(b, {})[k] = vals
            row += f"{np.mean(vals):12.4f} (max {np.max(vals):7.4f})"
        print(row)
    print("pairs:", {b: len(v) for b, v in buckets.items()})
    if a.json:
        a.json.write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
