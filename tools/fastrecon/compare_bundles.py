"""Quality gate for the fast reconstruction: compare a candidate bundle against a reference.

    python -m tools.fastrecon.compare_bundles <reference_bundle> <candidate_bundle> \
        [--tolerances tol.json] [--json out.json]

A bundle is a reconstruction result directory (result.npz + mesh.obj + manifest.json).
"No quality degradation" is defined operationally: every metric below must stay within the
run-to-run noise floor of the stock pipeline (measure it by comparing two stock runs with
--calibrate, which writes the tolerances file), or inside the explicit defaults otherwise.

Metrics (p95 over frames valid in both bundles unless noted):
  camera      position error (mm, after a global rigid alignment) and rotation error (deg)
  object      posed object surface in the camera frame: per-frame symmetric chamfer (mm)
              between the two meshes, each placed by its own pose. Invariant to the mesh's
              canonical frame (SAM3D picks a different one per run), unlike raw pose deltas,
              which are reported but not gated.
  hands       wrist translation (mm), wrist rotation (deg), finger joint angle (deg) error;
              MANO scale difference; betas L2
  validity    fraction of frames whose valid flags disagree (camera/object/each hand)
  gravity     angle between the two alignment rotations (deg)
Exit code 0 = within tolerance, 1 = regression, 2 = usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_TOLERANCES = {
    # Deliberately tight placeholders; replace with --calibrate output (noise floor x margin).
    "camera_pos_mm_p95": 2.0,
    "camera_rot_deg_p95": 0.2,
    "object_surface_mm_p95": 3.0,
    "object_scale_rel": 0.005,
    "hand_trans_mm_p95": 2.0,
    "hand_rot_deg_p95": 1.0,
    "finger_deg_p95": 1.0,
    "hand_scale_abs_p95": 0.005,
    "betas_l2": 0.02,
    "validity_disagree_frac": 0.0,
    "gravity_deg": 0.1,
}
CALIBRATION_MARGIN = 2.0  # tolerance = max(noise floor x margin, an absolute floor)
ABSOLUTE_FLOOR = {k: v / 20 for k, v in DEFAULT_TOLERANCES.items()}


def _load(bundle: Path) -> dict[str, np.ndarray]:
    with np.load(bundle / "result.npz", allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _rot_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geodesic angle between rotation matrices (..., 3, 3), in degrees."""
    rel = np.einsum("...ji,...jk->...ik", a.astype(np.float64), b.astype(np.float64))
    flat = rel.reshape(-1, 3, 3)
    # Orthonormalise first: float32 inputs make arccos(trace) noisy near 0 deg.
    u, _, vt = np.linalg.svd(flat)
    angles = Rotation.from_matrix(u @ vt).magnitude()
    return np.degrees(angles).reshape(rel.shape[:-2])


def _p95(x: np.ndarray) -> float:
    return float(np.percentile(x, 95)) if x.size else 0.0


def _rigid_align(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kabsch: R, t minimising |R src + t - dst| (no scale: both runs claim metric)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s))
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return r, mu_d - r @ mu_s


def _chamfer_mm(ref: Path, cand: Path, n: int = 20000) -> float:
    import trimesh  # optional dependency, only for the mesh metric

    a = trimesh.load(ref, force="mesh")
    b = trimesh.load(cand, force="mesh")
    # Same seed for both: identical meshes give identical samples, i.e. exactly 0.
    pa = np.asarray(a.sample(n, seed=0), dtype=np.float64)
    pb = np.asarray(b.sample(n, seed=0), dtype=np.float64)
    from scipy.spatial import cKDTree

    d_ab = cKDTree(pb).query(pa)[0]
    d_ba = cKDTree(pa).query(pb)[0]
    return float(1000 * (d_ab.mean() + d_ba.mean()) / 2)


def _posed_surface_mm_p95(ref_mesh: Path, cand_mesh: Path, ro: np.ndarray, co: np.ndarray,
                          ref_scale: float, cand_scale: float, n: int = 3000, stride: int = 5) -> float:
    import trimesh  # noqa: PLC0415
    from scipy.spatial import cKDTree  # noqa: PLC0415

    pa = np.asarray(trimesh.load(ref_mesh, force="mesh").sample(n, seed=0), dtype=np.float64) * ref_scale
    pb = np.asarray(trimesh.load(cand_mesh, force="mesh").sample(n, seed=0), dtype=np.float64) * cand_scale
    per_frame = []
    for ta, tb in zip(ro[::stride].astype(np.float64), co[::stride].astype(np.float64)):
        qa = pa @ ta[:3, :3].T + ta[:3, 3]
        qb = pb @ tb[:3, :3].T + tb[:3, 3]
        per_frame.append(1000 * (cKDTree(qb).query(qa)[0].mean() + cKDTree(qa).query(qb)[0].mean()) / 2)
    return _p95(np.asarray(per_frame))


def compare(ref_dir: Path, cand_dir: Path, mesh: bool = True) -> dict[str, float]:
    ref, cand = _load(ref_dir), _load(cand_dir)
    m: dict[str, float] = {}
    info: dict[str, float] = {}  # reported, not gated (suffix _raw)

    # Validity agreement.
    disagree = []
    for key in ("camera_is_valid", "object_is_valid", "hand_left_is_valid", "hand_right_is_valid"):
        if key in ref and key in cand:
            disagree.append(float(np.mean(ref[key] != cand[key])))
    m["validity_disagree_frac"] = max(disagree) if disagree else 0.0

    # Camera trajectory: rigidly align candidate positions to reference first (a SLAM run
    # may legitimately differ by a global rigid motion; it should not differ in shape).
    cv = ref["camera_is_valid"] & cand["camera_is_valid"]
    rc, cc = ref["camera_to_world_transform"][cv], cand["camera_to_world_transform"][cv]
    r, t = _rigid_align(cc[:, :3, 3], rc[:, :3, 3])
    pos = (cc[:, :3, 3] @ r.T) + t
    aligned_mm = _p95(1000 * np.linalg.norm(pos - rc[:, :3, 3], axis=1))
    raw_mm = _p95(1000 * np.linalg.norm(cc[:, :3, 3] - rc[:, :3, 3], axis=1))
    info["camera_rot_deg_p95_kabsch_raw"] = _p95(_rot_deg(rc[:, :3, :3], r @ cc[:, :3, :3]))
    if raw_mm <= 1.05 * aligned_mm + 0.5:
        # Already in the same world frame (deterministic SLAM + gravity): the fitted rotation
        # is noise. On a near-linear camera path the roll about the path is unconstrained by
        # positions, so applying it rotates every camera by up to tens of degrees (seen as
        # 8-28 deg "errors" when the unaligned cameras agree to ~0.12 deg).
        r, t = np.eye(3), np.zeros(3)
    pos = (cc[:, :3, 3] @ r.T) + t
    m["camera_pos_mm_p95"] = _p95(1000 * np.linalg.norm(pos - rc[:, :3, 3], axis=1))
    m["camera_rot_deg_p95"] = _p95(_rot_deg(rc[:, :3, :3], r @ cc[:, :3, :3]))

    # Object in the camera frame (independent of the world frame choice).
    ov = ref["object_is_valid"] & cand["object_is_valid"]
    ro, co = ref["object_to_camera_transform"][ov], cand["object_to_camera_transform"][ov]
    info["object_trans_mm_p95_raw"] = _p95(1000 * np.linalg.norm(ro[:, :3, 3] - co[:, :3, 3], axis=1))
    info["object_rot_deg_p95_raw"] = _p95(_rot_deg(ro[:, :3, :3], co[:, :3, :3]))
    if mesh and (ref_dir / "mesh.obj").exists() and (cand_dir / "mesh.obj").exists():
        m["object_surface_mm_p95"] = _posed_surface_mm_p95(
            ref_dir / "mesh.obj", cand_dir / "mesh.obj", ro, co,
            float(ref["object_scale"]), float(cand["object_scale"]),
        )
    m["object_scale_rel"] = float(abs(float(cand["object_scale"]) / float(ref["object_scale"]) - 1))

    # Hands in the camera frame.
    trans, rot, finger, scale, betas = [], [], [], [], []
    for side in ("left", "right"):
        v = ref[f"hand_{side}_is_valid"] & cand[f"hand_{side}_is_valid"]
        rt, ct = ref[f"hand_{side}_wrist_to_camera_transform"][v], cand[f"hand_{side}_wrist_to_camera_transform"][v]
        trans.append(1000 * np.linalg.norm(rt[:, :3, 3] - ct[:, :3, 3], axis=1))
        rot.append(_rot_deg(rt[:, :3, :3], ct[:, :3, :3]))
        rf = Rotation.from_rotvec(ref[f"hand_{side}_finger_pose"][v].reshape(-1, 3))
        cf = Rotation.from_rotvec(cand[f"hand_{side}_finger_pose"][v].reshape(-1, 3))
        finger.append(np.degrees((rf.inv() * cf).magnitude()))
        scale.append(np.abs(ref[f"hand_{side}_scale"][v] - cand[f"hand_{side}_scale"][v]))
        betas.append(float(np.linalg.norm(ref[f"hand_{side}_betas"] - cand[f"hand_{side}_betas"])))
    m["hand_trans_mm_p95"] = _p95(np.concatenate(trans))
    m["hand_rot_deg_p95"] = _p95(np.concatenate(rot))
    m["finger_deg_p95"] = _p95(np.concatenate(finger))
    m["hand_scale_abs_p95"] = _p95(np.concatenate(scale))
    m["betas_l2"] = max(betas)

    if "gravity_alignment_transform" in ref and "gravity_alignment_transform" in cand:
        m["gravity_deg"] = float(_rot_deg(ref["gravity_alignment_transform"][:3, :3], cand["gravity_alignment_transform"][:3, :3]))

    if mesh and (ref_dir / "mesh.obj").exists() and (cand_dir / "mesh.obj").exists():
        info["mesh_chamfer_mm_raw"] = _chamfer_mm(ref_dir / "mesh.obj", cand_dir / "mesh.obj")
    m.update({k: v for k, v in info.items()})
    return m


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--tolerances", type=Path, help="JSON of metric -> max allowed value")
    p.add_argument("--calibrate", type=Path, help="treat candidate as a 2nd STOCK run; write tolerances here")
    p.add_argument("--json", type=Path, help="write the metrics + verdict here")
    p.add_argument("--no-mesh", action="store_true")
    a = p.parse_args()
    for b in (a.reference, a.candidate):
        if not (b / "result.npz").exists():
            print(f"error: {b}/result.npz not found", file=sys.stderr)
            return 2

    metrics = compare(a.reference, a.candidate, mesh=not a.no_mesh)
    if a.calibrate:
        tol = {k: max(metrics.get(k, 0.0) * CALIBRATION_MARGIN, ABSOLUTE_FLOOR[k]) for k in DEFAULT_TOLERANCES}
        a.calibrate.write_text(json.dumps({"noise_floor": metrics, "tolerances": tol}, indent=2) + "\n")
        print(f"wrote tolerances (noise floor x {CALIBRATION_MARGIN}) to {a.calibrate}")
    tol = DEFAULT_TOLERANCES
    if a.tolerances:
        tol = json.loads(a.tolerances.read_text()).get("tolerances", tol)

    failed = []
    print(f"{'metric':26} {'value':>10} {'limit':>10}")
    for k, v in metrics.items():
        limit = tol.get(k)
        bad = limit is not None and v > limit
        failed += [k] if bad else []
        print(f"{k:26} {v:10.4f} {limit if limit is not None else float('nan'):10.4f} {'FAIL' if bad else 'ok'}")
    verdict = "PASS" if not failed else "REGRESSION: " + ", ".join(failed)
    print(verdict)
    if a.json:
        a.json.write_text(json.dumps({"metrics": metrics, "tolerances": tol, "verdict": verdict}, indent=2) + "\n")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
