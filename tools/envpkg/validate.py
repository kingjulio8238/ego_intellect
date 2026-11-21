"""Validate an environment package (env_package/v1) end to end.

Schema first, then the facts the schema cannot express: every file present and hashed, nothing
extra, every reference resolvable inside the package, frames rigid, and the motion files agreeing
with the reconstruction through the declared frame transforms. Built to be run by labs on a
delivered package, and by the builder before it publishes one.
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .common import (
    PACKAGE_SCHEMA,
    SUPPORT_SCHEMA,
    obj_references,
    obj_vertices,
    rigid_errors,
    schema_errors,
    sha256_file,
)

FRAME_TOL_M = 1e-3
RESULT_KEYS = {
    "camera_to_world_transform": (4, 4),
    "camera_is_valid": (),
    "object_to_world_transform": (4, 4),
    "object_is_valid": (),
    "hand_left_wrist_to_world_transform": (4, 4),
    "hand_left_finger_pose": (15, 3),
    "hand_left_is_valid": (),
    "hand_right_wrist_to_world_transform": (4, 4),
    "hand_right_finger_pose": (15, 3),
    "hand_right_is_valid": (),
}


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _resample(values: np.ndarray, fps_from: float, fps_to: float, n_to: int) -> np.ndarray:
    t_from = np.arange(len(values)) / fps_from
    t_to = np.arange(n_to) / fps_to
    return np.stack([np.interp(t_to, t_from, values[:, i]) for i in range(values.shape[1])], axis=1)


def _apply(transform: Any, points: np.ndarray) -> np.ndarray:
    t = np.asarray(transform, dtype=float)
    return points @ t[:3, :3].T + t[:3, 3]


def _row(path: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    if table.num_rows != 1:
        raise ValueError(f"expected 1 row, found {table.num_rows}")
    return table.to_pylist()[0]


def validate_package(root: Path, *, strict: bool = False) -> Report:
    """Validate the package at ``root``. ``strict``: also require what delivery requires."""
    report = Report()
    err, warn = report.errors.append, report.warnings.append
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        err("manifest.json missing")
        return report
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        err(f"manifest.json is not JSON: {exc}")
        return report
    report.errors += schema_errors(PACKAGE_SCHEMA, manifest)
    if report.errors:
        return report  # the deep checks below rely on the schema's structure

    # Files: exactly the listed set, each intact, no links, nothing outside the root.
    listed = {}
    for entry in manifest["files"]:
        if entry["path"] in listed:
            err(f"files: {entry['path']} listed twice")
        listed[entry["path"]] = entry
    on_disk = set()
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            full = Path(dirpath) / name
            if full.is_symlink():
                err(f"{full.relative_to(root).as_posix()}: symlinks are not allowed")
        for name in filenames:
            rel = (Path(dirpath) / name).relative_to(root).as_posix()
            if rel != "manifest.json":
                on_disk.add(rel)
    for rel in sorted(on_disk - set(listed)):
        err(f"{rel}: present but not listed in manifest files")
    for rel, entry in sorted(listed.items()):
        path = root / rel
        if not path.is_file():
            err(f"{rel}: listed but missing")
        elif path.stat().st_size != entry["bytes"]:
            err(f"{rel}: {path.stat().st_size} bytes, manifest says {entry['bytes']}")
        elif sha256_file(path) != entry["sha256"]:
            err(f"{rel}: sha256 mismatch")
    if report.errors:
        return report

    core, robots = manifest["core"], manifest["robots"]
    obj = core["object"]
    references = [
        core["reconstruction"]["result"], core["reconstruction"]["manifest"], core["human_tracks"]["path"],
        obj["mesh"], obj["urdf"], obj["collision"]["mesh"], *obj["materials"], *obj["textures"],
    ]
    for robot in robots.values():
        references += [robot["motion"]["path"], robot["support_surfaces"], robot["retarget_report"]]
        if robot["qa"]["physics_replay"]:
            references.append(robot["qa"]["physics_replay"])
    for ref in references:
        if ref not in listed:
            err(f"manifest references {ref}, which is not a package file")

    # Frames.
    t_tracks = manifest["frames"]["tracks"]["T_from_world"]
    report.errors += rigid_errors(t_tracks, "frames.tracks.T_from_world")
    for name, robot in robots.items():
        report.errors += rigid_errors(robot["frame"]["T_from_world"], f"robots.{name}.frame.T_from_world")
    if report.errors:
        return report

    # Reconstruction.
    n, fps = core["n_frames"], core["fps"]
    npz = np.load(root / core["reconstruction"]["result"])
    for key, tail in RESULT_KEYS.items():
        if key not in npz.files:
            err(f"result.npz: missing {key}")
        elif npz[key].shape != (n, *tail):
            err(f"result.npz: {key} shape {npz[key].shape}, expected {(n, *tail)}")
    if report.errors:
        return report
    world_object = np.asarray(npz["object_to_world_transform"], dtype=float)[:, :3, 3]

    def check_paths(row: dict[str, Any], label: str) -> None:
        for key in ("object_mesh_paths", "object_urdf_paths"):
            for value in row.get(key) or []:
                if value not in listed:
                    err(f"{label}: {key} entry {value!r} is not a package file")
        if row.get("raw_motion_file") not in (None, "", core["reconstruction"]["result"]):
            err(f"{label}: raw_motion_file {row['raw_motion_file']!r} is not the package result.npz")

    # Human tracks (tracks frame, core rate).
    tracks_meta = core["human_tracks"]
    try:
        tracks = _row(root / tracks_meta["path"])
    except Exception as exc:  # noqa: BLE001 - any unreadable file is a contract error
        err(f"{tracks_meta['path']}: unreadable ({exc})")
        tracks = None
    if tracks is not None:
        check_paths(tracks, tracks_meta["path"])
        if abs(float(tracks["fps"]) - tracks_meta["fps"]) > 1e-6 or abs(tracks_meta["fps"] - fps) > 1e-6:
            err(f"{tracks_meta['path']}: fps {tracks['fps']} vs manifest {tracks_meta['fps']} / core {fps}")
        for side in ("left", "right"):
            for key, tail in ((f"mano_{side}_trans", (3,)), (f"mano_{side}_finger_pose", (45,)), (f"mano_{side}_joints", (21, 3))):
                values = np.asarray(tracks.get(key) or [], dtype=float)
                if values.shape != (tracks_meta["n_frames"], *tail) or not np.all(np.isfinite(values)):
                    err(f"{tracks_meta['path']}: {key} shape {values.shape}, expected {(tracks_meta['n_frames'], *tail)} finite")
        if tracks_meta["n_frames"] != n:
            err(f"{tracks_meta['path']}: {tracks_meta['n_frames']} frames, reconstruction {n}")
        else:
            track_object = np.asarray(tracks["object_root_position"], dtype=float)
            gap = float(np.abs(track_object - _apply(t_tracks, world_object)).max())
            if gap > FRAME_TOL_M:
                err(f"{tracks_meta['path']}: object track is {gap * 1000:.1f} mm from T_from_world(reconstruction)")

    # Object.
    mesh_path = root / obj["mesh"]
    materials, textures = obj_references(mesh_path)
    base = PurePosixPath(obj["mesh"]).parent
    for kind, names, declared in (("material", materials, obj["materials"]), ("texture", textures, obj["textures"])):
        resolved = sorted(str(base / name) for name in names)
        if resolved != sorted(declared):
            err(f"object {kind}s: mesh references {resolved}, manifest declares {sorted(declared)}")
    vertices = obj_vertices(mesh_path)
    if len(vertices) < 4:
        err(f"{obj['mesh']}: {len(vertices)} vertices")
    else:
        if np.abs(np.array([vertices.min(0), vertices.max(0)]) - np.array(obj["bounds_m"])).max() > 1e-6:
            err(f"{obj['mesh']}: bounds differ from manifest bounds_m")
        extent = vertices.max(0) - vertices.min(0)
        if extent.max() > 2.0 or extent.min() < 0.005:
            err(f"{obj['mesh']}: extent {np.round(extent, 4).tolist()} m is outside 5 mm .. 2 m (unit error?)")
    physics = obj["physics"]
    inertia = np.asarray(physics["inertia_kg_m2"], dtype=float)
    if np.abs(inertia - inertia.T).max() > 1e-12 or np.linalg.eigvalsh(inertia).min() <= 0:
        err("object inertia is not symmetric positive definite")
    try:
        urdf = ET.parse(root / obj["urdf"]).getroot()
        urdf_dir = PurePosixPath(obj["urdf"]).parent
        for mesh in urdf.iter("mesh"):
            target = str(urdf_dir / mesh.get("filename", ""))
            if target not in listed:
                err(f"{obj['urdf']}: mesh {mesh.get('filename')!r} is not a package file")
        mass = float(urdf.find("link/inertial/mass").get("value"))
        if abs(mass - physics["mass_kg"]) > 1e-9:
            err(f"{obj['urdf']}: mass {mass} != manifest {physics['mass_kg']}")
    except (ET.ParseError, AttributeError, TypeError, ValueError) as exc:
        err(f"{obj['urdf']}: unreadable ({exc})")
    if physics["mass_source"] == "default_constant":
        warn(f"object mass {physics['mass_kg']} kg is a constant default, not estimated from the object")
    if obj["collision"]["source"] == "visual_mesh":
        warn("collision uses the visual mesh; simulators approximate it (convex hull / decomposition)")

    # Robots.
    statuses = []
    for name, robot in robots.items():
        label = robot["motion"]["path"]
        meta = robot["motion"]
        try:
            motion = _row(root / label)
        except Exception as exc:  # noqa: BLE001
            err(f"{label}: unreadable ({exc})")
            continue
        check_paths(motion, label)
        if motion.get("schema_version") != "motion_v1" or motion.get("motion_kind") != "single_robot":
            err(f"{label}: not motion_v1/single_robot")
        if abs(float(motion["fps"]) - meta["fps"]) > 1e-6:
            err(f"{label}: fps {motion['fps']} != manifest {meta['fps']}")
        joints = np.asarray(motion["robot_joint_positions"], dtype=float)
        if joints.shape != (meta["n_frames"], meta["joint_count"]) or len(motion["robot_joint_names"]) != meta["joint_count"]:
            err(f"{label}: joints {joints.shape}, manifest {(meta['n_frames'], meta['joint_count'])}")
        elif not np.all(np.isfinite(joints)):
            err(f"{label}: non-finite joint positions")
        duration, core_duration = (meta["n_frames"] - 1) / meta["fps"], (n - 1) / fps
        if abs(duration - core_duration) > 1.0 / fps + 1e-9:
            err(f"{label}: {duration:.3f} s long, reconstruction {core_duration:.3f} s")
        robot_object = np.asarray(motion["object_root_position"], dtype=float)
        if robot_object.shape == (meta["n_frames"], 3):
            expected = _apply(robot["frame"]["T_from_world"], _resample(world_object, fps, meta["fps"], meta["n_frames"]))
            gap = float(np.abs(robot_object - expected).max())
            if gap > FRAME_TOL_M:
                err(f"{label}: object is {gap * 1000:.1f} mm from frame.T_from_world(reconstruction)")
        else:
            err(f"{label}: object_root_position shape {robot_object.shape}")

        support = json.loads((root / robot["support_surfaces"]).read_text())
        report.errors += schema_errors(SUPPORT_SCHEMA, support)
        if support.get("frame") != f"robot:{name}":
            err(f"{robot['support_surfaces']}: frame {support.get('frame')!r}, expected 'robot:{name}'")

        qa = robot["qa"]
        statuses.append(qa["status"])
        if qa["physics_replay"]:
            metrics = json.loads((root / qa["physics_replay"]).read_text())
            if metrics.get("schema") != "v2d_replay_metrics_v1":
                err(f"{qa['physics_replay']}: schema {metrics.get('schema')!r}")
            elif ("flagged" if metrics.get("flag") else "pass") != qa["status"]:
                err(f"robots.{name}.qa.status {qa['status']!r} disagrees with the replay flag {metrics.get('flag')}")
            elif metrics.get("completion") != qa["completion"]:
                err(f"robots.{name}.qa.completion disagrees with the replay report")
        elif qa["status"] != "not_run":
            err(f"robots.{name}.qa: status {qa['status']!r} without a physics replay report")

    gates = manifest["qa"]["gates"]
    for gate, entry in gates.items():
        if entry["report"] is None:
            if entry["status"] != "not_run":
                err(f"qa.gates.{gate}: status {entry['status']!r} without a report")
            continue
        if entry["report"] not in listed:
            err(f"qa.gates.{gate}: report {entry['report']} is not a package file")
            continue
        data = json.loads((root / entry["report"]).read_text())
        if ("pass" if data.get("verdict") in ("accept", "pass") else "reject") != entry["status"]:
            err(f"qa.gates.{gate}: status {entry['status']!r} disagrees with the report verdict {data.get('verdict')!r}")
    from .build import package_verdict

    verdict = package_verdict(statuses, [g["status"] for g in gates.values()])
    if manifest["qa"]["verdict"] != verdict:
        err(f"qa.verdict {manifest['qa']['verdict']!r}, QA statuses imply {verdict!r}")

    if manifest["task"] is not None:
        _check_task(root, manifest, listed, report)
    if "isaaclab" in manifest["simulators"]:
        _check_isaaclab(root, manifest, listed, report)

    provenance = manifest["provenance"]
    if manifest["task"] is None:
        (err if strict else warn)("no task spec attached (task/task.json)")
    if not provenance["complete"]:
        (err if strict else warn)(f"provenance incomplete: missing {', '.join(provenance.get('missing', []))}")
    if strict:
        if verdict != "pass":
            err(f"qa verdict {verdict!r}; delivery requires 'pass'")
        if provenance["builder"]["dirty"] is not False:
            err("package built from a dirty or unknown worktree; delivery requires a clean commit")
        if manifest["source"]["video_sha256"] is None:
            err("source video hash missing")
    return report


def _check_task(root: Path, manifest: dict[str, Any], listed: dict[str, Any], report: Report) -> None:
    """task_spec/v1: schema, references, definitions as catalogued, timing and dimensions
    consistent, and the success criterion recomputed on the reference motion."""
    from .common import TASK_SCHEMA
    from .success import EVALUATORS
    from .task import ACTIONS, OBSERVATIONS, REWARDS, SUCCESS, TERMINATIONS, reference_object_heights

    err = report.errors.append
    path = manifest["task"]["spec"]
    if path not in listed:
        err(f"task spec {path} is not a package file")
        return
    spec = json.loads((root / path).read_text())
    errors = schema_errors(TASK_SCHEMA, spec)
    if errors:
        report.errors += errors
        return
    success = spec["success"]
    if success["definition"] != SUCCESS[success["evaluator"]]:
        err(f"task: success definition differs from the catalogued {success['evaluator']} definition")
    for robot, part in spec["robots"].items():
        label = f"task.robots.{robot}"
        if robot not in manifest["robots"]:
            err(f"{label}: robot not in the package")
            continue
        meta = manifest["robots"][robot]["motion"]
        if part["env_spec"] not in listed:
            err(f"{label}: env_spec {part['env_spec']} is not a package file")
        ref = part["reference"]
        if ref["motion"] != meta["path"] or ref["frames"] != meta["n_frames"] or abs(ref["fps"] - meta["fps"]) > 1e-6:
            err(f"{label}: reference {ref} does not match the package motion {meta}")
        timing = part["timing"]
        if abs(timing["control_dt_s"] - timing["physics_dt_s"] * timing["decimation"]) > 1e-9:
            err(f"{label}: control_dt_s != physics_dt_s x decimation")
        expected = int(np.floor((ref["frames"] - 1) / ref["fps"] / (timing["control_dt_s"] * ref["playback_speed"]) + 1e-6)) + 1
        if abs(expected - ref["control_steps"]) > 1:
            err(f"{label}: {ref['control_steps']} control steps, the motion plays for {expected}")
        if part["episode"]["evaluation"]["max_steps"] != ref["control_steps"]:
            err(f"{label}: evaluation max_steps {part['episode']['evaluation']['max_steps']} != reference control steps")
        for group, catalog in (("rewards", REWARDS), ("terminations", TERMINATIONS)):
            for term in part[group]:
                short = term["function"].rsplit(":", 1)[1]
                if catalog.get(short) != term["definition"]:
                    err(f"{label}: {group} term {term['name']} ({short}) is not described as catalogued")
        obs = part["observations"]
        for term in obs["terms"]:
            if OBSERVATIONS.get(term["function"].rsplit(":", 1)[1]) != term["definition"]:
                err(f"{label}: observation {term['name']} is not described as catalogued")
        dims = sum(int(np.prod(term["dim"])) for term in obs["terms"] if term["dim"])
        if obs["dim"] and dims != int(np.prod(obs["dim"])):
            err(f"{label}: observation terms sum to {dims}, observation dim {obs['dim']}")
        action = part["action"]
        if action["joint_names"] and len(action["joint_names"]) != action["dim"]:
            err(f"{label}: action dim {action['dim']} != {len(action['joint_names'])} joints")
        if action["dim"] != manifest["robots"][robot]["motion"]["joint_count"]:
            err(f"{label}: action dim {action['dim']} != motion joint count")
        for term in action["terms"]:
            if ACTIONS.get(term["kind"]) != term["definition"]:
                err(f"{label}: action {term['name']} is not described as catalogued")
        try:
            motion = _row(root / ref["motion"])
            heights = reference_object_heights(motion, control_dt=timing["control_dt_s"], motion_speed=ref["playback_speed"])
            check = EVALUATORS[success["evaluator"]](heights, success["params"])
        except Exception as exc:  # noqa: BLE001
            err(f"{label}: cannot recompute the reference check ({exc})")
            continue
        if not check["success"]:
            err(f"{label}: the reference motion fails the success criterion ({check})")
        stored = part["reference_check"]
        if (check["success"], check["steps"]) != (stored["success"], stored["steps"]):
            err(f"{label}: reference_check {stored} disagrees with the recomputed {check}")


def _check_isaaclab(root: Path, manifest: dict[str, Any], listed: dict[str, Any], report: Report) -> None:
    """isaaclab_env/v1: schema; the exported motion is the core motion (asset paths aside);
    the support USDA states the same disks as support_surfaces.json; object files are the
    core object's; the task id matches the task spec."""
    from .build import parse_support_usda
    from .common import ISAACLAB_SCHEMA
    from .isaaclab import HMD, PATH_FIELDS, partition

    err = report.errors.append
    entry = manifest["simulators"]["isaaclab"]
    if entry["schema"] != ISAACLAB_SCHEMA or entry["env"] not in listed:
        err(f"simulators.isaaclab: {entry} is not an {ISAACLAB_SCHEMA} package file")
        return
    env = json.loads((root / entry["env"]).read_text())
    errors = schema_errors(ISAACLAB_SCHEMA, env)
    if errors:
        report.errors += errors
        return
    task = json.loads((root / manifest["task"]["spec"]).read_text()) if manifest["task"] else {"robots": {}}
    obj = manifest["core"]["object"]
    for name in (obj["mesh"], obj["urdf"], *obj["materials"], *obj["textures"]):
        copy = f"{HMD}/ego_recon/processed/{PurePosixPath(name).name}"
        if copy not in listed or listed[copy]["sha256"] != listed[name]["sha256"]:
            err(f"isaaclab: {copy} is not a copy of {name}")
    for robot, part in env["robots"].items():
        label = f"isaaclab.robots.{robot}"
        if robot not in manifest["robots"]:
            err(f"{label}: robot not in the package")
            continue
        entry_robot = manifest["robots"][robot]
        if part["motion_file"] != str(partition(manifest["sequence_id"], robot)):
            err(f"{label}: motion_file {part['motion_file']} is not this clip's partition")
        for key in ("motion", "support_usda", "object_urdf", "env_spec"):
            if part[key] not in listed:
                err(f"{label}: {key} {part[key]} is not a package file")
        if report.errors:
            continue
        if part["robot_description_sha256"] != entry_robot["model"]["sha256"]:
            err(f"{label}: robot description hash differs from robots.{robot}.model")
        task_part = task["robots"].get(robot)
        if task_part is None or task_part["isaaclab_task"] != part["task"] or task_part["env_spec"] != part["env_spec"]:
            err(f"{label}: task {part['task']} / env_spec do not match the task spec")
        exported, core = _row(root / part["motion"]), _row(root / entry_robot["motion"]["path"])
        differing = sorted(k for k in core if k not in PATH_FIELDS and _differs(core[k], exported.get(k)))
        if differing or set(core) != set(exported):
            err(f"{label}: exported motion differs from the core motion in {differing or 'columns'}")
        surfaces = json.loads((root / entry_robot["support_surfaces"]).read_text())
        try:
            parsed = parse_support_usda(root / part["support_usda"], robot)
        except Exception as exc:  # noqa: BLE001
            err(f"{label}: support USDA unreadable ({exc})")
            continue
        if parsed["surfaces"] != surfaces["surfaces"]:
            err(f"{label}: support USDA states different disks than {entry_robot['support_surfaces']}")
        qa = entry_robot["qa"]
        if qa["physics_replay"] and part["expected_replay"] is not None:
            replay = json.loads((root / qa["physics_replay"]).read_text())
            if (replay["flag"], replay["completion"]) != (part["expected_replay"]["flag"], part["expected_replay"]["completion"]):
                err(f"{label}: expected_replay differs from {qa['physics_replay']}")


def _differs(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return not (a == b or (np.isnan(a) and np.isnan(b)))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) != len(b) or any(_differs(x, y) for x, y in zip(a, b))
    return a != b
