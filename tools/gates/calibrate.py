"""Calibrate and check the gate limits (tools/gates/limits.json) against labelled clips.

Reconstruction QA: every good run must pass, and every real bad run plus every synthetic
defect made from each good run must be rejected.

    python3 -m tools.gates.calibrate recon --good DIR... [--bad DIR...] [--defects-from DIR...]

Each DIR is a downloaded run's `reconstruction/` folder: result_slam_gravity_aligned/ and masks/.
Triage: every good video must be accepted, and every bad or synthetic video rejected.

    python3 -m tools.gates.calibrate triage --good VIDEO... --bad VIDEO... [--defects-from VIDEO...]

Exit 0 only when every label is matched. The tables printed are the calibration record
(docs/pipeline/GATES.md).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .common import evaluate

RNG_SEED = 0


def _write_bundle(src: Path, dst: Path, arrays: dict[str, np.ndarray], mesh_scale: float = 1.0) -> Path:
    dst.mkdir(parents=True)
    for name in ("manifest.json", "material.mtl", "material_0.png"):
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    np.savez(dst / "result.npz", **arrays)
    lines = []
    for line in (src / "mesh.obj").read_text(errors="replace").splitlines():
        if line.startswith("v ") and mesh_scale != 1.0:
            parts = line.split()
            xyz = [float(v) * mesh_scale for v in parts[1:4]]
            line = "v " + " ".join(f"{v:.7f}" for v in xyz) + (" " + " ".join(parts[4:]) if len(parts) > 4 else "")
        lines.append(line)
    (dst / "mesh.obj").write_text("\n".join(lines) + "\n")
    return dst


def _camera_consistent(a: dict[str, np.ndarray], world_key: str, camera_key: str) -> None:
    a[camera_key] = np.linalg.inv(a["camera_to_world_transform"]) @ a[world_key]


def recon_defects(run: Path, out: Path) -> dict[str, tuple[Path, Path]]:
    """Synthetic defects of one good run: name -> (bundle, masks). Each changes one thing
    a reconstruction can plausibly get wrong, by an amount that matters downstream."""
    rng = np.random.default_rng(RNG_SEED)
    bundle, masks = run / "result_slam_gravity_aligned", run / "masks"
    base = {k: np.array(v) for k, v in np.load(bundle / "result.npz").items()}
    n = len(base["camera_is_valid"])
    made: dict[str, tuple[Path, Path]] = {}

    def make(name: str, edit: Callable[[dict[str, np.ndarray]], None], mesh_scale: float = 1.0, mask_dir: Path = masks) -> None:
        arrays = {k: v.copy() for k, v in base.items()}
        edit(arrays)
        made[name] = (_write_bundle(bundle, out / name, arrays, mesh_scale), mask_dir)

    def object_shift(a):  # object 3 cm off sideways in the camera view
        a["object_to_camera_transform"][:, 0, 3] += 0.03
        a["object_to_world_transform"] = a["camera_to_world_transform"] @ a["object_to_camera_transform"]

    def object_lag(a):  # object trajectory 10 frames late relative to the hands and camera
        world = a["object_to_world_transform"].copy()
        a["object_to_world_transform"] = world[np.maximum(np.arange(n) - 10, 0)]
        _camera_consistent(a, "object_to_world_transform", "object_to_camera_transform")

    def object_jitter(a):  # 5 mm per-frame noise on the object position
        a["object_to_world_transform"][:, :3, 3] += rng.normal(0, 0.005, (n, 3))
        _camera_consistent(a, "object_to_world_transform", "object_to_camera_transform")

    def wrist_jitter(a):  # 1 cm per-frame noise on one wrist
        a["hand_right_wrist_to_world_transform"][:, :3, 3] += rng.normal(0, 0.01, (n, 3))
        _camera_consistent(a, "hand_right_wrist_to_world_transform", "hand_right_wrist_to_camera_transform")

    def hand_depth(a):  # both hands 50% too far from the camera (hand scale/depth error)
        for side in ("left", "right"):
            a[f"hand_{side}_wrist_to_camera_transform"][:, :3, 3] *= 1.5
            a[f"hand_{side}_wrist_to_world_transform"] = a["camera_to_world_transform"] @ a[f"hand_{side}_wrist_to_camera_transform"]

    def camera_jump(a):  # SLAM re-localisation: the second half moves 30 cm in world
        for key in ("camera_to_world_transform", "object_to_world_transform",
                    "hand_left_wrist_to_world_transform", "hand_right_wrist_to_world_transform"):
            a[key][n // 2:, 0, 3] += 0.3

    def object_dropout(a):  # object pose lost on 40% of frames
        a["object_is_valid"][rng.random(n) < 0.4] = False

    def hands_swapped(a):  # left/right identities exchanged
        for key in [k for k in a if k.startswith("hand_left_")]:
            other = key.replace("hand_left_", "hand_right_")
            a[key], a[other] = a[other].copy(), a[key].copy()

    make("object_shift_3cm", object_shift)
    make("object_lag_10f", object_lag)
    make("object_jitter_5mm", object_jitter)
    make("wrist_jitter_1cm", wrist_jitter)
    make("hand_depth_+50pct", hand_depth)
    make("camera_jump_30cm", camera_jump)
    make("object_dropout_40pct", object_dropout)
    make("hands_swapped", hands_swapped)
    make("object_scale_x1.3", lambda a: None, mesh_scale=1.3)
    make("object_scale_x0.7", lambda a: None, mesh_scale=0.7)
    make("object_units_x0.1", lambda a: None, mesh_scale=0.1)
    wrong = out / "wrong_object_masks"
    shutil.copytree(masks, wrong)
    shutil.rmtree(wrong / "1")
    hand_id = Path(json.loads((bundle / "manifest.json").read_text())["sources"]["right_hand_dir"]).name
    shutil.copytree(masks / hand_id, wrong / "1")  # the segmenter tracked a hand, not the object
    make("wrong_object_segmented", lambda a: None, mask_dir=wrong)
    return made


def triage_defects(video: Path, out: Path) -> dict[str, Path]:
    """Synthetic bad videos derived from one good video (ffmpeg)."""
    out.mkdir(parents=True, exist_ok=True)
    made = {}
    filters = {
        "blurred": ["-vf", "gblur=sigma=12"],
        "dark": ["-vf", "eq=brightness=-0.45:contrast=0.5"],
        "overexposed": ["-vf", "eq=brightness=0.6:contrast=0.6"],
        "frozen": ["-vf", "fps=3,fps=30"],  # each frame repeated 10 times (a stalled encoder or capture)
        "scene_cut": ["-filter_complex", "[0:v]split[a][b];[b]hue=h=150,hflip[c];[a][c]concat=n=2:v=1"],
        "too_short": ["-t", "1.2"],
        "low_res": ["-vf", "scale=320:-2"],
    }
    for name, args in filters.items():
        dst = out / f"{video.stem}_{name}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), *args, "-an", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", str(dst)], check=True)
        made[name] = dst
    return made


def _table(rows: dict[str, dict[str, Any]], keys: list[str]) -> str:
    head = f"{'clip':40s} {'label':6s} {'got':7s} " + " ".join(f"{k[:12]:>12s}" for k in keys)
    lines = [head]
    for name, row in rows.items():
        cells = " ".join(f"{row['metrics'][k]:12.4g}" if row["metrics"].get(k) is not None else f"{'-':>12s}" for k in keys)
        lines.append(f"{name[:40]:40s} {row['label']:6s} {row['got']:7s} {cells}")
    return "\n".join(lines)


def calibrate_recon(good: list[Path], bad: list[Path], defects_from: list[Path], fps: float) -> int:
    from .recon_qa import SECTION, analyze_bundle

    rows: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        cases = [(run.name, run / "result_slam_gravity_aligned", run / "masks", "good") for run in good]
        cases += [(run.name, run / "result_slam_gravity_aligned", run / "masks", "bad") for run in bad]
        for run in defects_from:
            for name, (bundle, masks) in recon_defects(run, Path(tmp) / run.name).items():
                cases.append((f"{run.name}:{name}", bundle, masks, "bad"))
        for name, bundle, masks, label in cases:
            metrics = analyze_bundle(bundle, fps=fps, masks=masks if masks.is_dir() else None)
            _, reasons = evaluate(metrics, SECTION)
            rows[name] = {"label": label, "got": "reject" if reasons else "pass", "metrics": metrics, "reasons": reasons}
    keys = [k for k in next(iter(rows.values()))["metrics"] if k not in ("frames", "fps")]
    print(_table(rows, keys))
    return _summary(rows, positive="pass", negative="reject")


def calibrate_triage(good: list[Path], bad: list[Path], defects_from: list[Path], hand_model: Path | None) -> int:
    from .triage import SECTION, analyze_video

    rows: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        cases = [(v.name, v, "good") for v in good] + [(v.name, v, "bad") for v in bad]
        for video in defects_from:
            cases += [(f"{video.stem}:{name}", path, "bad") for name, path in triage_defects(video, Path(tmp)).items()]
        for name, video, label in cases:
            metrics = analyze_video(video, hand_model=hand_model)
            _, reasons = evaluate(metrics, SECTION)
            rows[name] = {"label": label, "got": "reject" if reasons else "accept", "metrics": metrics, "reasons": reasons}
    keys = [k for k in next(iter(rows.values()))["metrics"] if not k.startswith("_")]
    print(_table(rows, keys))
    return _summary(rows, positive="accept", negative="reject")


def _summary(rows: dict[str, dict[str, Any]], *, positive: str, negative: str) -> int:
    wrong = [(n, r) for n, r in rows.items() if r["got"] != (positive if r["label"] == "good" else negative)]
    print(f"\n{len(rows) - len(wrong)}/{len(rows)} labels matched")
    for name, row in wrong:
        print(f"  MISMATCH {name}: label {row['label']}, got {row['got']} {row['reasons']}")
    for name, row in rows.items():
        if row["got"] == negative and row["label"] == "bad":
            print(f"  {name}: {'; '.join(r.split(':')[0] for r in row['reasons'])}")
    return 0 if not wrong else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="gate", required=True)
    for name in ("recon", "triage"):
        p = sub.add_parser(name)
        p.add_argument("--good", type=Path, nargs="*", default=[])
        p.add_argument("--bad", type=Path, nargs="*", default=[])
        p.add_argument("--defects-from", type=Path, nargs="*", default=[])
    sub.choices["recon"].add_argument("--fps", type=float, default=30.0)
    sub.choices["triage"].add_argument("--hand-model", type=Path)
    args = parser.parse_args(argv)
    if args.gate == "recon":
        return calibrate_recon(args.good, args.bad, args.defects_from, args.fps)
    return calibrate_triage(args.good, args.bad, args.defects_from, args.hand_model)


if __name__ == "__main__":
    sys.exit(main())
