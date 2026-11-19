"""Uniformly rescale a reconstruction bundle to metric using the hands as the ruler.

Monocular reconstruction fixes geometry only up to a global scale; this pipeline takes it
from MoGe's metric depth, which can be off at egocentric close range. The fitted MANO hand
scale is the clip's only absolute ruler: an adult hand is MANO-sized to within the
+/-10-15 % that the shape betas absorb, so a fitted scale of 1.44 means the whole world is
1.44x too large (box, hands, camera path). That is the onboard_ego_clip Step-0 check, whose
preferred fix is this one: a uniform whole-scene rescale of the bundle (mesh plus every
translation), done once so every embodiment's retarget inherits it.

    python -m tools.e2e.rescale_bundle <bundle_dir> [--factor auto|<float>] [--out <dir>] [--min-deviation 0.15]

`auto` = 2 / (mean(hand_left_scale) + mean(hand_right_scale)), the same estimator as the
Vega retargeter's `--scene_scale auto`. Refuses to act when the hands are already within
--min-deviation of 1.0 (a large hand is not an oversized world). Writes result.npz,
mesh.obj and manifest.json (with a `metric_rescale` record) to --out (default: in place,
after copying the originals to <bundle>.unscaled/).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

TRANSFORM_KEYS = (
    "camera_to_world_transform",
    "object_to_world_transform",
    "object_to_camera_transform",
    "hand_left_wrist_to_world_transform",
    "hand_right_wrist_to_world_transform",
    "hand_left_wrist_to_camera_transform",
    "hand_right_wrist_to_camera_transform",
)
VECTOR_KEYS = ("hand_left_wrist_trans_in_camera", "hand_right_wrist_trans_in_camera")
HAND_SCALE_KEYS = ("hand_left_scale", "hand_right_scale")


def auto_factor(npz: dict[str, np.ndarray]) -> float:
    left, right = (float(np.mean(npz[key])) for key in HAND_SCALE_KEYS)
    return 2.0 / (left + right)


def rescale_arrays(npz: dict[str, np.ndarray], factor: float) -> dict[str, np.ndarray]:
    out = dict(npz)
    for key in TRANSFORM_KEYS:
        if key not in out:
            continue
        transforms = np.array(out[key], copy=True)
        det = np.linalg.det(transforms[..., :3, :3].astype(np.float64))
        if not np.allclose(det, 1.0, atol=1e-3):
            raise ValueError(f"{key}: rotation blocks carry scale (det {det.min():.4f}..{det.max():.4f}); polar-decompose first")
        transforms[..., :3, 3] *= factor
        out[key] = transforms.astype(npz[key].dtype)
    for key in VECTOR_KEYS:
        if key in out:
            out[key] = (np.asarray(out[key]) * factor).astype(npz[key].dtype)
    for key in HAND_SCALE_KEYS:
        if key in out:
            out[key] = (np.asarray(out[key]) * factor).astype(npz[key].dtype)
    return out


def rescale_obj(text: str, factor: float) -> str:
    lines = []
    for line in text.splitlines(keepends=True):
        if line.startswith("v "):
            parts = line.split()
            xyz = [float(value) * factor for value in parts[1:4]]
            rest = parts[4:]  # optional vertex colours
            line = "v " + " ".join(f"{v:.8f}" for v in xyz) + ("" if not rest else " " + " ".join(rest)) + "\n"
        lines.append(line)
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--factor", default="auto")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--min-deviation", type=float, default=0.15, help="act only when |mean hand scale - 1| exceeds this")
    args = parser.parse_args()

    bundle: Path = args.bundle
    with np.load(bundle / "result.npz", allow_pickle=False) as data:
        npz = {key: data[key] for key in data.files}
    manifest = json.loads((bundle / "manifest.json").read_text())
    if "metric_rescale" in manifest:
        raise SystemExit(f"{bundle} was already rescaled: {manifest['metric_rescale']}")

    hands = {key: float(np.mean(npz[key])) for key in HAND_SCALE_KEYS}
    mean_hand = (hands["hand_left_scale"] + hands["hand_right_scale"]) / 2
    factor = auto_factor(npz) if args.factor == "auto" else float(args.factor)
    if args.factor == "auto" and abs(mean_hand - 1.0) <= args.min_deviation:
        print(f"hand scale {mean_hand:.3f} is within {args.min_deviation} of 1.0: bundle treated as metric, nothing to do")
        return 0

    out = args.out or bundle
    if out == bundle:
        backup = bundle.with_name(bundle.name + ".unscaled")
        if backup.exists():
            raise SystemExit(f"backup {backup} already exists; refusing to overwrite it")
        shutil.copytree(bundle, backup)
        print(f"original bundle copied to {backup}")
    else:
        shutil.copytree(bundle, out, dirs_exist_ok=True)

    np.savez(out / "result.npz", **rescale_arrays(npz, factor))
    (out / "mesh.obj").write_text(rescale_obj((bundle / "mesh.obj").read_text() if out != bundle else (backup / "mesh.obj").read_text(), factor))
    manifest["metric_rescale"] = {
        "factor": factor,
        "source": "hand_scale" if args.factor == "auto" else "manual",
        "hand_scale_before": hands,
        "applied_to": ["mesh.obj vertices", *TRANSFORM_KEYS, *VECTOR_KEYS, *HAND_SCALE_KEYS],
        "note": "threejs_scene/ is not regenerated by this tool",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"rescaled {out} by {factor:.4f} (mean hand scale {mean_hand:.3f} -> {mean_hand * factor:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
