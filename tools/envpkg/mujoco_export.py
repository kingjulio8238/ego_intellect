"""MuJoCo export (mujoco_env/v1): the package's ``mujoco/`` folder.

One self-contained MJCF scene per robot, built from the package and the clip's resolved Isaac
Lab training environment (task spec), so it states the same physical setup:

- robot: the robot's URDF converted by MuJoCo's own URDF compiler, collision geometry only
  (meshes copied into ``mujoco/<robot>/assets``), fixed base at the origin of the robot frame,
  gravity compensated (Isaac disables gravity on the robot links);
- actuators: one position servo per joint with Isaac's stiffness, damping, effort limit,
  armature and joint friction; the arm's 4-physics-step command delay is a runtime property
  (recorded in env.json, applied by mujoco_replay.py);
- object: the core mesh (referenced, not copied) on a free joint with the package's mass,
  centre of mass and inertia; support disks and a floor as static geoms;
- contacts: PhysX averages the two materials' friction (hand 2.0, object 0.5: 1.25), MuJoCo
  takes the maximum, so hand geoms get 1.25 and the rest 0.5, giving the same pairs. The robot
  collides with itself (Isaac: self-collisions on) and the object, not the support (Isaac:
  robot-support collisions off).

Contract and the cross-simulator check: docs/pipeline/MUJOCO_EXPORT.md.
"""

from __future__ import annotations

import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .common import REPO_ROOT, ContractError, write_json

MUJOCO_SCHEMA = "mujoco_env/v1"
ROOT = PurePosixPath("mujoco")
ROBOT_BITS = (1, 1)  # contype, conaffinity: robot-robot and robot-object
OBJECT_BITS = (2, 1 | 4)  # object-robot, object-support
SUPPORT_BITS = (4, 2)  # support-object only
HAND_FRICTION = None  # derived from the env spec (PhysX average of hand and object friction)
LICENSE_FILES = ("robotic_grounding/source/robotic_grounding/robotic_grounding/assets/urdfs/sharpawave/LICENSE",
                 "robotic_grounding/source/robotic_grounding/robotic_grounding/assets/urdfs/sharpawave/NOTICE")


def _regex_value(table: dict[str, Any] | None, joint: str) -> Any:
    """Isaac actuator tables map joint-name regexes to values; first full match wins."""
    for pattern, value in (table or {}).items():
        if re.fullmatch(pattern, joint):
            return value
    return None


def _robot_spec(urdf: Path, mesh_dir_name: str) -> tuple[Any, dict[str, bytes], dict[str, str]]:
    """MjSpec of the robot (collision only) from its URDF; (spec, mesh bytes by name, link parent)."""
    import mujoco

    root = ET.parse(urdf).getroot()
    assets: dict[str, bytes] = {}
    for link in root.iter("link"):
        for visual in list(link.findall("visual")):
            link.remove(visual)
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename", "")
        source = (urdf.parent / filename).resolve()
        flat = str(PurePosixPath(filename.replace("\\", "/"))).replace("../", "").replace("/", "__")
        assets[flat] = source.read_bytes()
        mesh.set("filename", flat)
    parent = {j.find("child").get("link"): j.find("parent").get("link") for j in root.iter("joint")}
    jointed = {j.find("child").get("link") for j in root.iter("joint") if j.get("type") != "fixed"}
    compiler = ET.SubElement(ET.SubElement(root, "mujoco"), "compiler")
    compiler.attrib.update({"strippath": "false", "fusestatic": "false", "discardvisual": "true",
                            "balanceinertia": "true", "meshdir": mesh_dir_name})
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"), assets=assets)
    owner = {}
    for link in parent:  # the Isaac body a link is merged into (fixed joints are merged)
        current = link
        while current not in jointed and current in parent:
            current = parent[current]
        owner[link] = current
    return spec, assets, owner


def _owner(name: str, owner: dict[str, str]) -> str:
    return owner.get(name, name)


def _matches(name: str, patterns: list[str]) -> bool:
    return any(re.fullmatch(p, name) for p in patterns)


def export_mujoco(staging: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Write ``mujoco/`` into a package being assembled; returns ``simulators.mujoco``."""
    import mujoco
    import pyarrow.parquet as pq
    from scipy.spatial.transform import Rotation

    task = json.loads((staging / manifest["task"]["spec"]).read_text())
    core_object = manifest["core"]["object"]
    physics = core_object["physics"]
    robots_env: dict[str, Any] = {}
    for robot, entry in manifest["robots"].items():
        part = task["robots"][robot]
        env_spec = json.loads((staging / part["env_spec"]).read_text())
        base = ROOT / robot
        urdf = REPO_ROOT / entry["model"]["description_file"]
        spec, mesh_bytes, owner = _robot_spec(urdf, "assets")
        (staging / base / "assets").mkdir(parents=True, exist_ok=True)
        for name, data in mesh_bytes.items():
            (staging / base / "assets" / name).write_bytes(data)
        for license_file in LICENSE_FILES:
            shutil.copy2(REPO_ROOT / license_file, staging / base / "assets" / f"sharpa_{Path(license_file).name}")

        timing = part["timing"]
        spec.option.timestep = timing["physics_dt_s"]
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC

        material = part["physics"]["hand_contact_material"]
        hand_patterns = list(material["bodies"] or [])
        hand_mu = float(material["static_friction"][0]) if material.get("static_friction") else 0.5
        object_mu = float(physics["friction"]["static"])
        pair_mu = (hand_mu + object_mu) / 2.0  # PhysX default combine mode: average
        for body in spec.bodies:
            if body.name == "world":
                continue
            body.gravcomp = 0.0 if part["physics"]["robot_gravity"] else 1.0
            hand = _matches(_owner(body.name, owner), hand_patterns)
            for geom in body.geoms:
                geom.contype, geom.conaffinity = ROBOT_BITS
                geom.friction = [pair_mu if hand else object_mu, 0.005, 0.0001]

        actuators = env_spec["robot"]["actuators"]
        motion = pq.read_table(staging / entry["motion"]["path"]).to_pylist()[0]
        joints = list(motion["robot_joint_names"])
        joint_table = []
        for joint in joints:
            group = next((g for g, a in actuators.items() if any(re.fullmatch(p, joint) for p in a["joint_names_expr"])), None)
            if group is None:
                raise ContractError(f"mujoco export: no Isaac actuator drives joint {joint}")
            a = actuators[group]
            kp, kd = _regex_value(a["stiffness"], joint), _regex_value(a["damping"], joint)
            effort = _regex_value(a.get("effort_limit_sim"), joint) or _regex_value(a.get("effort_limit"), joint)
            armature = _regex_value(a.get("armature"), joint) or 0.0
            friction = _regex_value(a.get("friction"), joint) or 0.0
            mj_joint = spec.joint(joint)
            if mj_joint is None:
                raise ContractError(f"mujoco export: joint {joint} not in the converted robot")
            mj_joint.armature = float(armature)
            mj_joint.frictionloss = float(friction)
            act = spec.add_actuator(name=joint, target=joint, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            act.gainprm[0] = float(kp)
            act.biasprm[0:3] = [0.0, -float(kp), -float(kd)]
            act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.ctrlrange = list(mj_joint.range)
            if effort:
                act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
                act.forcerange = [-float(effort), float(effort)]
            delay = int(a.get("max_delay") or 0) if "Delayed" in str(a.get("class_type", "")) else 0
            joint_table.append({"joint": joint, "actuator_group": group, "kp": kp, "kd": kd, "effort": effort,
                                "armature": armature, "friction": friction, "delay_physics_steps": delay})

        # Object on a free joint at its reference frame-0 pose (robot frame), core mesh referenced.
        mesh_file = "../../../" + core_object["mesh"]
        spec.add_mesh(name="object_mesh", file=mesh_file)
        pos0 = np.asarray(motion["object_root_position"][0], dtype=float)
        quat0 = np.asarray(motion["object_body_wxyz"][0][0], dtype=float)
        body = spec.worldbody.add_body(name="object", pos=pos0.tolist(), quat=quat0.tolist())
        body.add_freejoint(name="object_free")
        body.explicitinertial = True
        body.mass = float(physics["mass_kg"])
        body.ipos = list(physics["com_m"])
        inertia = np.asarray(physics["inertia_kg_m2"], dtype=float)
        body.fullinertia = [inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]]
        geom = body.add_geom(name="object_geom", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="object_mesh")
        geom.contype, geom.conaffinity = OBJECT_BITS
        geom.friction = [object_mu, 0.005, 0.0001]
        geom.rgba = [0.8, 0.3, 0.3, 1.0]

        surfaces = json.loads((staging / entry["support_surfaces"]).read_text())
        for disk in surfaces["surfaces"]:
            g = spec.worldbody.add_geom(name=f"support_{disk['name']}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                        size=[disk["radius"], disk["thickness"] / 2.0, 0.0], pos=list(disk["center"]))
            g.contype, g.conaffinity = SUPPORT_BITS
            g.friction = [object_mu, 0.005, 0.0001]
            g.rgba = [0.6, 0.6, 0.6, 1.0]
        floor = spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3.0, 3.0, 0.05])
        floor.contype, floor.conaffinity = SUPPORT_BITS

        scene = base / "scene.xml"
        spec.modelfiledir = str(staging / base)  # resolve every mesh from disk, as a reader of scene.xml will
        spec.compile()
        (staging / scene).write_text(spec.to_xml())
        try:  # the written file must load on its own, from its own directory
            model = mujoco.MjModel.from_xml_path(str(staging / scene))
        except Exception as exc:  # noqa: BLE001
            raise ContractError(f"mujoco export: {scene} does not load: {exc}") from exc

        sides = {"left": ("L_", "left_"), "right": ("R_", "right_")}
        contact_bodies = part["command"].get("hand_contact_bodies") or hand_patterns
        bodies = [model.body(i).name for i in range(model.nbody)]
        robots_env[robot] = {
            "scene": str(scene),
            "isaaclab_task": part["isaaclab_task"],
            "joints": joint_table,
            "control": {
                "physics_dt_s": timing["physics_dt_s"], "decimation": timing["decimation"],
                "control_dt_s": timing["control_dt_s"], "playback_speed": part["reference"]["playback_speed"],
                "target": "reference joint position at the current control step (+ the policy's residual)",
            },
            "ee_bodies": list(motion["ee_link_names"]),
            "hand_contact_bodies": {
                side: sorted(b for b in bodies if b.startswith(prefixes) and _matches(_owner(b, owner), contact_bodies))
                for side, prefixes in sides.items()
            },
            "object": {"body": "object", "geom": "object_geom", "mesh": core_object["mesh"],
                       "initial_pose": {"pos": pos0.tolist(), "wxyz": quat0.tolist(),
                                        "euler_xyz_deg": Rotation.from_quat(quat0[[1, 2, 3, 0]]).as_euler("xyz", degrees=True).round(3).tolist()}},
            "friction": {"hand_object": pair_mu, "object_support": object_mu,
                         "note": "PhysX averages friction (Isaac); MuJoCo takes the max: values chosen so each pair matches"},
            "not_modelled": ["Isaac joint velocity limits (arm 2.4 rad/s)", "PhysX solver iteration counts"],
            "robot_assets": {"collision_meshes": f"{base}/assets", "visual_meshes": "not included",
                             "licenses": "Sharpa hand meshes: Apache-2.0 (copied); Dexmate Vega meshes: licence not "
                                         "recorded in the source repository: confirm before external delivery"},
        }
    env = {"schema": MUJOCO_SCHEMA, "mujoco_version": mujoco.__version__, "robots": robots_env}
    write_json(staging / ROOT / "env.json", env)
    return {"env": str(ROOT / "env.json"), "schema": MUJOCO_SCHEMA}
