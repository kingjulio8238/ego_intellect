#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render the Vega retarget verification MP4 from the written motion parquet.

The same video ``ego_recon_to_dexmate_sharpa.py --video`` renders in-process (pyrender/EGL,
the IK robot's own visual model, the object mesh), but from the saved ``motion_v1``
partition, so it can run after the clip is delivered instead of on its critical path. The
parquet holds the robot joints in the fixed-model order; they are mapped by name onto the
reduced IK model the video draws (same values: both orders are written from one solution).

Usage:
    python scripts/retarget/render_vega_retarget_video.py \
        --motion_dir <hmd>/ego_recon/processed/sequence_id=<seq>/robot_name=vega_sharpa \
        --output <run>/vega_retarget_qa/<seq>_vega_sharpa.mp4
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ego_recon_to_dexmate_sharpa import (  # noqa: E402
    load_ik_module,
    render_verification_video,
)


def _resolve_mesh(recorded: str, motion_dir: Path) -> Path:
    """The parquet records the mesh path as written; fall back to processed/ beside it."""
    path = Path(recorded)
    if path.is_file():
        return path
    candidate = motion_dir.parent.parent / path.name
    if candidate.is_file():
        return candidate
    raise SystemExit(f"render_vega_retarget_video: object mesh not found: {recorded}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--motion_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--azimuth_deg", type=float, default=225.0)
    parser.add_argument("--elevation_deg", type=float, default=30.0)
    parser.add_argument("--padding", type=float, default=1.5)
    args = parser.parse_args()

    files = sorted(glob.glob(str(args.motion_dir / "*.parquet")))
    if not files:
        raise SystemExit(f"render_vega_retarget_video: no parquet in {args.motion_dir}")
    row = {key: value[0] for key, value in pq.read_table(files[0]).to_pydict().items()}
    names = [str(n) for n in row["robot_joint_names"]]
    q_fixed = np.asarray(row["robot_joint_positions"], dtype=np.float64)
    object_position = np.asarray(row["object_body_position"], dtype=np.float64)[:, 0]
    object_wxyz = np.asarray(row["object_body_wxyz"], dtype=np.float64)[:, 0]
    ee_pose_w = np.asarray(row["ee_pose_w"], dtype=np.float64)

    ik = load_ik_module().DexmateSharpaIK()
    model = ik.robot.model
    index = {name: k for k, name in enumerate(names)}
    q = np.tile(np.asarray(ik.q_reference, dtype=np.float64), (q_fixed.shape[0], 1))
    for j in range(1, model.njoints):
        name = model.names[j]
        if name not in index:
            raise SystemExit(f"render_vega_retarget_video: joint {name} not in the parquet")
        q[:, model.joints[j].idx_q] = q_fixed[:, index[name]]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    render_verification_video(
        ik,
        q,
        str(row["object_name"]),
        _resolve_mesh(str(row["object_mesh_paths"][0]), args.motion_dir),
        object_position,
        object_wxyz,
        ee_pose_w,
        float(row["fps"]),
        args.output,
        camera_azimuth_deg=args.azimuth_deg,
        camera_elevation_deg=args.elevation_deg,
        camera_padding=args.padding,
    )
    print(f"wrote verification video -> {args.output} ({q.shape[0]} frames @ {float(row['fps']):g} fps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
