#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render the physics replay of a clip (``replay_metrics.py --states_out``) as an MP4.

The "environment on video" visual check without Isaac RTX cameras: pyrender (off-screen
EGL) draws what physics did, frame by frame, from the states the camera-free replay saved:

* the robot (grey), posed by pinocchio FK from the simulated joint positions;
* the object where physics put it (orange) and where the reference says it should be
  (translucent green ghost);
* the reference wrist targets (green spheres) and the simulated wrists (red spheres);
* the support surface(s) from ``reconstructed_stage/<seq>[_<robot>]_support.usda``.

An object that floats, sinks, slips or drops separates from its ghost; hands that miss
leave their spheres apart. The replay stops at the environment's first termination, so
the video ends at the failure (the last frame is held for a second).

No Isaac Sim: runs anywhere pinocchio, pyrender and EGL do (Modal GPUs included).

Usage:
    python scripts/render_replay_overlay.py --states <run>/simulate/vega/replay_states.npz \
        --output <run>/render/vega/overlay.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402
import pinocchio as pin  # noqa: E402
import pyrender  # noqa: E402
import trimesh  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "retarget"))
from _offline_video import OfflineVideoRenderer  # noqa: E402

PHYSICS_COLOR = (235, 140, 40, 255)
GHOST_COLOR = (60, 200, 90, 90)
SUPPORT_COLOR = (150, 120, 100, 255)


class _Kinematics:
    """The surface OfflineVideoRenderer.add_robot needs (``.robot`` with models/data)."""

    def __init__(self, robot: pin.RobotWrapper) -> None:
        self.robot = robot


def _transform(pos: np.ndarray, wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in wxyz)
    quat = pin.Quaternion(w, x, y, z)
    quat.normalize()
    out = np.eye(4)
    out[:3, :3] = quat.toRotationMatrix()
    out[:3, 3] = pos
    return out


def _support_path(motion_dir: Path) -> Path | None:
    """Same lookup as the scene builder: robot-specific support first, then shared."""
    seq = motion_dir.parent.name.split("=", 1)[-1]
    robot = motion_dir.name.split("=", 1)[-1]
    stage = motion_dir.parent.parent.parent / "reconstructed_stage"
    for candidate in (stage / f"{seq}_{robot}_support.usda", stage / f"{seq}_support.usda"):
        if candidate.is_file():
            return candidate
    return None


def _support_meshes(usda: Path) -> list[tuple[trimesh.Trimesh, np.ndarray]]:
    """Cylinder/Cube support prims (the shapes reconstruct_support_surfaces.py writes)."""
    text = usda.read_text()
    meshes = []
    for kind, body in re.findall(r'def (Cylinder|Cube) "[^"]*"[^{]*\{(.*?)\n\s*\}', text, re.S):
        translate = re.search(r"xformOp:translate = \(([^)]*)\)", body)
        pose = np.eye(4)
        if translate:
            pose[:3, 3] = [float(v) for v in translate.group(1).split(",")]
        if kind == "Cylinder":
            radius = float(re.search(r"double radius = ([-0-9.e]+)", body).group(1))
            height = float(re.search(r"double height = ([-0-9.e]+)", body).group(1))
            mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=48)
        else:
            size = re.search(r"double size = ([-0-9.e]+)", body)
            scale = re.search(r"xformOp:scale = \(([^)]*)\)", body)
            extents = np.full(3, float(size.group(1)) if size else 2.0)
            if scale:
                extents = extents * np.array([float(v) for v in scale.group(1).split(",")])
            mesh = trimesh.creation.box(extents=extents)
        mesh.visual.face_colors = np.array(SUPPORT_COLOR, dtype=np.uint8)
        meshes.append((mesh, pose))
    return meshes


def _resolve_mesh(recorded: str, motion_dir: Path) -> Path:
    """The parquet records the mesh under the container's asset tree; fall back to the
    processed/ directory next to the motion when that path is absent."""
    path = Path(recorded)
    if path.is_file():
        return path
    for candidate in (motion_dir.parent.parent / path.name, motion_dir.parent.parent.parent / path.name):
        if candidate.is_file():
            return candidate
    raise SystemExit(f"render_replay_overlay: object mesh not found: {recorded}")


def _caption(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    """Draw text lines top-left (PIL when available; the video is still useful without)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    y = 12
    for text, color in lines:
        draw.text((14, y), text, fill=color, font=font)
        y += 28
    return np.asarray(image)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--azimuth_deg", type=float, default=225.0)
    parser.add_argument("--elevation_deg", type=float, default=30.0)
    parser.add_argument("--padding", type=float, default=1.6)
    parser.add_argument("--hold_s", type=float, default=1.0, help="hold the last frame")
    args = parser.parse_args()

    st = np.load(args.states, allow_pickle=False)
    joint_pos = st["joint_pos"]
    names = [str(n) for n in st["joint_names"]]
    urdf = Path(str(st["urdf_path"]))
    motion_dir = Path(str(st["motion_dir"]))
    fps = max(1, int(round(1.0 / float(st["control_dt_s"]))))

    robot = pin.RobotWrapper.BuildFromURDF(str(urdf), package_dirs=[str(urdf.parent)], root_joint=None)
    model = robot.model
    index = {name: k for k, name in enumerate(names)}
    missing = [model.names[j] for j in range(1, model.njoints) if model.names[j] not in index]
    if missing:
        raise SystemExit(f"render_replay_overlay: joints in the URDF but not in the replay: {missing[:5]}")
    order = [(model.joints[j].idx_q, index[model.names[j]]) for j in range(1, model.njoints)]

    renderer = OfflineVideoRenderer(width=args.width, height=args.height, fps=fps)
    try:
        renderer.add_robot("robot", _Kinematics(robot))
        mesh = trimesh.load(str(_resolve_mesh(str(st["object_mesh_path"]), motion_dir)), force="mesh")

        def _flat(color: tuple[int, int, int, int]) -> trimesh.Trimesh:
            # Plain colours (drop the scan texture) so physics and reference read apart.
            out = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
            out.visual = trimesh.visual.ColorVisuals(out, face_colors=np.tile(np.array(color, dtype=np.uint8), (len(out.faces), 1)))
            return out

        physics = _flat(PHYSICS_COLOR)
        ghost = _flat(GHOST_COLOR)
        renderer.add_object("physics", physics)
        renderer.add_object("reference", ghost)
        support = _support_path(motion_dir)
        for k, (shape, pose) in enumerate(_support_meshes(support) if support else []):
            renderer._scene.add(pyrender.Mesh.from_trimesh(shape, smooth=False), pose=pose, name=f"support:{k}")
        sphere = trimesh.creation.icosphere(radius=0.02)
        marker_nodes = {}
        for label, color in (("ref_ee", (60, 200, 90, 255)), ("ee", (220, 50, 50, 255))):
            for side in range(st["ee_pos"].shape[1]):
                ball = sphere.copy()
                ball.visual.face_colors = np.array(color, dtype=np.uint8)
                marker_nodes[(label, side)] = renderer._scene.add(
                    pyrender.Mesh.from_trimesh(ball, smooth=False), pose=np.eye(4), name=f"{label}:{side}"
                )
        points = np.concatenate(
            [st["ee_pos"].reshape(-1, 3), st["ref_ee_pos"].reshape(-1, 3), st["ref_object_pos"]], axis=0
        )
        renderer.fit_camera(points, elevation_deg=args.elevation_deg, azimuth_deg=args.azimuth_deg,
                            padding=args.padding)

        q = pin.neutral(model)
        frames = len(joint_pos)
        reasons = json.loads(str(st["flag_reasons"]))
        status = (
            [("FLAGGED: " + reason, (200, 30, 30)) for reason in reasons]
            if bool(st["flag"])
            else [("QA checks pass", (20, 130, 50))]
        )
        legend = ("orange: object (physics)   green ghost: object (reference)   "
                  "green/red spheres: wrist target/simulated", (60, 60, 60))
        for t in range(frames):
            for q_index, replay_index in order:
                q[q_index] = joint_pos[t, replay_index]
            renderer.update_robot("robot", q)
            renderer.update_object("physics", _transform(st["object_pos"][t], st["object_wxyz"][t]))
            renderer.update_object("reference", _transform(st["ref_object_pos"][t], st["ref_object_wxyz"][t]))
            for (label, side), node in marker_nodes.items():
                pose = np.eye(4)
                pose[:3, 3] = st[f"{label}_pos"][t, side]
                renderer._scene.set_pose(node, pose)
            renderer.capture()
            renderer._frames[-1] = _caption(
                renderer._frames[-1],
                [(f"t = {t * float(st['control_dt_s']):5.2f} s   step {t}/{frames - 1}", (40, 40, 40)),
                 *status, legend],
            )
        for _ in range(int(round(args.hold_s * fps))):
            renderer._frames.append(renderer._frames[-1])
        renderer.save(args.output)
    finally:
        renderer.close()
    sidecar = {
        "states": str(args.states),
        "frames": frames,
        "fps": fps,
        "flag": bool(st["flag"]),
        "flag_reasons": json.loads(str(st["flag_reasons"])),
        "legend": {"orange": "object (physics)", "green ghost": "object (reference)",
                   "green spheres": "wrist targets", "red spheres": "simulated wrists"},
    }
    args.output.with_suffix(".json").write_text(json.dumps(sidecar, indent=1))
    print(f"RENDER_OVERLAY {json.dumps(sidecar)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
