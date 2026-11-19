"""Free pre-simulation check: do the retargeted robot fingertips end up inside the object?

    python tools/learn/check_penetration.py <motion_v1 partition dir or data.parquet> <object mesh .obj>

Deep, frequent penetration means the scene scale is wrong (oversized object vs hands); in
Isaac Lab it overflows the contact sensor buffer and the task crashes on the first steps
(simulate, 2026-11-17). NVIDIA's reference tissue_box_simple motion: 0% of frames.
Needs trimesh + rtree + scipy + pyarrow (the v2d-learn venv).
"""

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation

THRESHOLD_M = 0.005


def main() -> int:
    source, mesh_path = Path(sys.argv[1]), Path(sys.argv[2])
    parquet = source / "data.parquet" if source.is_dir() else source
    motion = pq.read_table(parquet).to_pylist()[0]
    mesh = trimesh.load(mesh_path, force="mesh")
    positions = np.asarray(motion["object_body_position"])[:, 0]
    wxyz = np.asarray(motion["object_body_wxyz"])[:, 0]
    tips = np.asarray(motion["hand_frames_w"])[..., :3]  # (2 hands, T, 5 tips, 3)
    inside, deepest = [], []
    for t in range(positions.shape[0]):
        rotation = Rotation.from_quat(wxyz[t][[1, 2, 3, 0]])
        local = rotation.inv().apply(tips[:, t].reshape(-1, 3) - positions[t])
        signed = trimesh.proximity.signed_distance(mesh, local)  # > 0 inside
        inside.append(int((signed > THRESHOLD_M).sum()))
        deepest.append(float(signed.max()))
    inside, deepest = np.array(inside), np.array(deepest)
    extent = np.sort(mesh.bounds[1] - mesh.bounds[0])[::-1]
    share = 100 * float(np.mean(inside > 0))
    print(f"object {np.round(100 * extent, 1)} cm | frames with a fingertip >{THRESHOLD_M * 1000:.0f} mm inside: "
          f"{share:.0f}% | max penetration {100 * deepest.max():.1f} cm | frames: {len(inside)}")
    verdict = "OK" if share < 5 and deepest.max() < 0.01 else "PENETRATION: check the scene scale before simulating"
    print(verdict)
    return 0 if verdict == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
