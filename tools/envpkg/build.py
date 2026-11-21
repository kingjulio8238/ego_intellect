"""Assemble the simulator-neutral core of an environment package from one clip's run root.

Reads the layout the e2e stages write (tools/e2e/stages.py) and writes env_package/v1
(docs/pipeline/ENV_PACKAGE.md). The Isaac Lab and MuJoCo folders and the task spec are added
by later stages; this module never invents data: anything it cannot find or verify is an error,
or is recorded as missing in the manifest.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    PACKAGE_SCHEMA,
    REPO_ROOT,
    SUPPORT_SCHEMA,
    ContractError,
    git_state,
    obj_references,
    obj_vertices,
    sha256_file,
    write_json,
)

BUNDLE = Path("reconstruction/result_slam_gravity_aligned")
LOADED = Path("intermediate/ego_recon/loaded")
PROCESSED = Path("human_motion_data/ego_recon/processed")
STAGE = Path("human_motion_data/ego_recon/reconstructed_stage")
PROVENANCE_STAGES = ("reconstruct", "retarget", "simulate")
GATES = ("triage", "reconstruction")  # RUN_ROOT/qa/<gate>.json, written by tools/gates


def package_verdict(robot_statuses: list[str], gate_statuses: list[str]) -> str:
    """rejected (a gate rejected) > flagged (physics QA) > incomplete (a check not run) > pass."""
    if "reject" in gate_statuses:
        return "rejected"
    if "flagged" in robot_statuses:
        return "flagged"
    if "not_run" in robot_statuses or "not_run" in gate_statuses:
        return "incomplete"
    return "pass"

# PhysX default material. The Vega task sets only the hand friction (2.0), so the object uses
# this in the physics QA replay (vega_sharpa_manip_env_cfg.py, hand_physics_material).
PHYSX_DEFAULT_FRICTION = {"static": 0.5, "dynamic": 0.5, "restitution": 0.0, "source": "physx_default"}
# scripts/generate_rigid_urdfs.py and the Vega retargeter write a constant mass for every object.
DEFAULT_OBJECT_MASS_KG = 0.3

# Max deviation (m) between a motion file's object positions and the reconstruction's, after the
# declared frame transform. Same data must agree to float precision (measured: 6e-8 m).
FRAME_TOL_M = 1e-3


@dataclass(frozen=True)
class RobotSource:
    embodiment: str  # tools/e2e embodiment name (simulate/<embodiment>/)
    report: str  # retarget report under the run root; {seq} = sequence id
    model: str  # robot description file, relative to the repository root


ROBOTS = {
    "vega_sharpa": RobotSource(
        embodiment="vega",
        report="vega_retarget_qa/{seq}_vega_sharpa_report.json",
        model="robotic_grounding/source/robotic_grounding/robotic_grounding/assets/urdfs/vega_sharpa/vega_sharpa_reduced.urdf",
    ),
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _scrub_paths(value: Any) -> Any:
    """Replace absolute filesystem paths (producer-internal) with their basename."""
    if isinstance(value, dict):
        return {k: _scrub_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub_paths(v) for v in value]
    if isinstance(value, str) and value.startswith("/") and "\n" not in value:
        return os.path.basename(value.rstrip("/"))
    return value


def _one_row(path: Path) -> tuple[Any, dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    if table.num_rows != 1:
        raise ContractError(f"{path}: expected 1 row per sequence, found {table.num_rows}")
    return table, table.to_pylist()[0]


def _write_row(table: Any, row: dict[str, Any], dst: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([row], schema=table.schema), dst)


def parse_support_usda(path: Path, robot: str) -> dict[str, Any]:
    """support_surfaces/v1 from the USDA that scripts/reconstruct_support_surfaces.py writes.

    Only the shapes that writer emits are accepted (z-axis Cylinder disks, translate-only);
    anything else is an error rather than a silently dropped surface.
    """
    text = path.read_text()
    header = text.split("def ", 1)[0]
    if not re.search(r'upAxis\s*=\s*"Z"', header) or not re.search(r"metersPerUnit\s*=\s*1(\.0*)?\b", header):
        raise ContractError(f"{path}: expected upAxis Z and metersPerUnit 1")
    prims = list(re.finditer(r'def\s+(\w+)\s+"([^"]+)"', text))
    kinds = [m.group(1) for m in prims]
    if not prims or kinds[0] != "Xform" or prims[0].group(2) != "support_surfaces" or set(kinds[1:]) - {"Cylinder"}:
        raise ContractError(f"{path}: expected one Xform 'support_surfaces' holding Cylinder disks, got {kinds}")
    surfaces = []
    for i, prim in enumerate(prims[1:], start=1):
        block = text[prim.end(): prims[i + 1].start() if i + 1 < len(prims) else len(text)]

        def attr(pattern: str) -> str:
            found = re.search(pattern, block)
            if not found:
                raise ContractError(f"{path}: {prim.group(2)} has no {pattern!r}")
            return found.group(1)

        if attr(r'axis\s*=\s*"(\w+)"') != "Z":
            raise ContractError(f"{path}: {prim.group(2)} is not a z-axis disk")
        order = attr(r"xformOpOrder\s*=\s*\[([^\]]*)\]")
        if [s.strip().strip('"') for s in order.split(",")] != ["xformOp:translate"]:
            raise ContractError(f"{path}: {prim.group(2)} has transform ops {order}; only translate is supported")
        center = [float(v) for v in attr(r"xformOp:translate\s*=\s*\(([^)]*)\)").split(",")]
        surfaces.append({
            "name": prim.group(2),
            "shape": "disk",
            "center": center,
            "axis": "z",
            "radius": float(attr(r"double\s+radius\s*=\s*([-0-9.eE+]+)")),
            "thickness": float(attr(r"double\s+height\s*=\s*([-0-9.eE+]+)")),
        })
    return {"schema": SUPPORT_SCHEMA, "frame": f"robot:{robot}", "units": "m", "surfaces": surfaces}


def _urdf_physics(urdf: Path) -> dict[str, Any]:
    inertial = ET.parse(urdf).getroot().find("link/inertial")
    if inertial is None:
        raise ContractError(f"{urdf}: no <inertial>")
    mass = float(inertial.find("mass").get("value"))
    origin = inertial.find("origin")
    com = [float(v) for v in (origin.get("xyz") if origin is not None else "0 0 0").split()]
    i = {k: float(inertial.find("inertia").get(k, 0)) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
    inertia = [[i["ixx"], i["ixy"], i["ixz"]], [i["ixy"], i["iyy"], i["iyz"]], [i["ixz"], i["iyz"], i["izz"]]]
    return {
        "mass_kg": mass,
        "mass_source": "default_constant" if abs(mass - DEFAULT_OBJECT_MASS_KG) < 1e-9 else "estimated",
        "com_m": com,
        "inertia_kg_m2": inertia,
        "friction": dict(PHYSX_DEFAULT_FRICTION),
    }


def _task(run_root: Path, staging: Path, sequence: str, context: dict[str, Any], first_motion: dict[str, Any],
          available: dict[str, Path], selected: list[str], instruction: str | None, env_id: str
          ) -> tuple[dict[str, Any] | None, list[str]]:
    """task/task.json from each robot's resolved training environment (simulate's env_spec.json)
    and the clip's task profile; (None, reasons) when either is missing."""
    from .task import build_task_spec, robot_task

    profile = context.get("profile")
    if not profile:
        return None, ["task spec not built: no task profile (run config contracts.task or --task-profile)"]
    parts, missing = {}, []
    for robot in selected:
        source = run_root / "simulate" / ROBOTS[robot].embodiment / "env_spec.json"
        if not source.is_file():
            missing.append(robot)
            continue
        env_spec = _load_json(source)
        target = f"task/{robot}/isaaclab_env_spec.json"
        write_json(staging / target, _scrub_paths(env_spec))
        _, motion = _one_row(available[robot])
        parts[robot] = robot_task(env_spec, motion, f"robots/{robot}/motion.parquet", profile["evaluator"], target)
    if missing:
        return None, [f"task spec not built: no resolved training environment (simulate env_spec.json) for {missing}"]
    spec = build_task_spec(env_id=env_id, sequence=sequence, instruction=instruction,
                           object_name=first_motion["safe_object_name"] or sequence, profile=profile, robots=parts)
    write_json(staging / "task/task.json", spec)
    return {"spec": "task/task.json", "schema": spec["schema"]}, []


def _provenance(run_root: Path, allow_missing: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    """(provenance block, e2e workflow dict) from run_manifest.json / e2e_config.json."""
    missing: list[str] = []
    stages: dict[str, Any] = {}
    workflow: dict[str, Any] = {}
    profile: dict[str, Any] | None = None
    parameters: dict[str, Any] = {}
    video: dict[str, Any] = {}
    config_path, manifest_path = run_root / "e2e_config.json", run_root / "run_manifest.json"
    if config_path.is_file():
        config = _load_json(config_path)
        workflow = config.get("workflow", {})
        profile = (config.get("contracts") or {}).get("task")
        parameters = _scrub_paths(config.get("stage_parameters", {}))
    else:
        missing.append("e2e_config.json")
    if manifest_path.is_file():
        run_manifest = _load_json(manifest_path)
        for name in PROVENANCE_STAGES:
            stage = run_manifest.get("stages", {}).get(name)
            if not stage or not stage.get("completed_at"):
                missing.append(f"stage {name}")
                continue
            repo = stage.get("repository", {})
            stages[name] = {
                "revision": repo.get("revision"),
                "runtime_tree_sha256": repo.get("runtime_tree_sha256"),
                "completed_at": stage.get("completed_at"),
            }
            if name == "reconstruct":
                inputs = list(stage.get("inputs", {}).values())
                video = inputs[0] if len(inputs) == 1 else {}
    else:
        missing.append("run_manifest.json")
    if missing and not allow_missing:
        raise ContractError(
            f"{run_root}: provenance incomplete ({', '.join(missing)}); "
            "pass --allow-missing-provenance for a non-deliverable package"
        )
    revision, dirty = git_state()
    block = {
        "complete": not missing,
        "missing": missing,
        "stages": stages,
        "stage_parameters": parameters,
        "builder": {"revision": revision, "dirty": dirty},
    }
    return block, {"workflow": workflow, "video": video, "profile": profile}


def _tracks_transform(npz: Any, tracks: dict[str, Any], path: Path) -> np.ndarray:
    """T_world_to_tracks. The loader re-centres the scene by a constant translation
    (object centred in xy, bottom on z); anything else means mismatched files."""
    from scipy.spatial.transform import Rotation

    world = np.asarray(npz["object_to_world_transform"], dtype=float)
    position = np.asarray(tracks["object_root_position"], dtype=float)
    if position.shape != (len(world), 3):
        raise ContractError(f"{path}: object track has {position.shape[0]} frames, reconstruction {len(world)}")
    delta = position - world[:, :3, 3]
    spread = float(np.abs(delta - delta[0]).max())
    rotation = Rotation.from_rotvec(np.asarray(tracks["object_root_axis_angle"], dtype=float))
    angle = float((rotation * Rotation.from_matrix(world[:, :3, :3]).inv()).magnitude().max())
    if spread > FRAME_TOL_M or angle > 1e-3:
        raise ContractError(
            f"{path}: tracks are not a constant translation of the reconstruction world "
            f"(position spread {spread:.2e} m, rotation {angle:.2e} rad)"
        )
    transform = np.eye(4)
    transform[:3, 3] = delta.mean(axis=0)
    return transform


def build_package(
    run_root: Path,
    out: Path,
    *,
    robots: list[str] | None = None,
    sequence_id: str | None = None,
    instruction: str | None = None,
    object_prompt: str | None = None,
    allow_flagged: bool = False,
    allow_missing_provenance: bool = False,
    force: bool = False,
    task_profile: Path | None = None,
) -> Path:
    """Write env_package/v1 for ``run_root`` to ``out``; returns ``out``. Validates before publishing."""
    run_root = run_root.resolve()
    provenance, context = _provenance(run_root, allow_missing_provenance)
    if task_profile is not None:
        context["profile"] = _load_json(task_profile)
    workflow = context["workflow"]
    sequence = sequence_id or workflow.get("sequence_id")
    if not sequence:
        found = sorted(p.name.split("=", 1)[1] for p in (run_root / PROCESSED).glob("sequence_id=*"))
        if len(found) != 1:
            raise ContractError(f"{run_root}: cannot infer the sequence id (found {found}); pass --sequence-id")
        sequence = found[0]

    bundle = run_root / BUNDLE
    for name in ("result.npz", "manifest.json"):
        if not (bundle / name).is_file():
            raise ContractError(f"missing reconstruction file {bundle / name}")
    npz = np.load(bundle / "result.npz")
    n_frames = int(npz["camera_to_world_transform"].shape[0])
    bundle_manifest = _load_json(bundle / "manifest.json")
    if bundle_manifest.get("n_frames") != n_frames:
        raise ContractError(f"{bundle}: manifest n_frames {bundle_manifest.get('n_frames')} != result.npz {n_frames}")

    track_files = glob.glob(str(run_root / LOADED / f"sequence_id={sequence}" / "robot_name=*" / "*.parquet"))
    if len(track_files) != 1:
        raise ContractError(f"{run_root / LOADED}: expected one loaded MANO track file for {sequence}, found {len(track_files)}")
    tracks_path = Path(track_files[0])
    tracks_table, tracks = _one_row(tracks_path)
    t_tracks = _tracks_transform(npz, tracks, tracks_path)
    fps = float(tracks["fps"])

    partitions = sorted((run_root / PROCESSED / f"sequence_id={sequence}").glob("robot_name=*"))
    available = {p.name.split("=", 1)[1]: p / "data.parquet" for p in partitions if (p / "data.parquet").is_file()}
    selected = robots or sorted(available)
    if not selected:
        raise ContractError(f"{run_root / PROCESSED}: no retargeted robot motion for {sequence}")
    for robot in selected:
        if robot not in available:
            raise ContractError(f"robot {robot}: no motion at {run_root / PROCESSED}/sequence_id={sequence}/robot_name={robot}")
        if robot not in ROBOTS:
            raise ContractError(f"robot {robot}: no packaging rule (frame transform source); known: {sorted(ROBOTS)}")

    staging = out.with_name(f".{out.name}.staging")
    shutil.rmtree(staging, ignore_errors=True)
    try:
        return _assemble(run_root, out, staging, sequence, provenance, context, npz, n_frames, bundle,
                         bundle_manifest, tracks_table, tracks, t_tracks, fps, available, selected,
                         instruction, object_prompt, allow_flagged, force)
    finally:
        shutil.rmtree(staging, ignore_errors=True)  # published packages are renamed away first


def _assemble(run_root, out, staging, sequence, provenance, context, npz, n_frames, bundle, bundle_manifest,
              tracks_table, tracks, t_tracks, fps, available, selected, instruction, object_prompt,
              allow_flagged, force) -> Path:
    from .validate import validate_package

    workflow = context["workflow"]
    env_id = f"{sequence}-{sha256_file(bundle / 'result.npz')[:10]}"
    (staging / "core/object").mkdir(parents=True)

    # Object: the metric mesh every robot motion references, with its URDF and materials.
    first_table, first_motion = _one_row(available[selected[0]])
    mesh_name = Path(first_motion["object_mesh_paths"][0]).name
    urdf_name = Path(first_motion["object_urdf_paths"][0]).name
    object_dir = run_root / PROCESSED
    materials, textures = obj_references(object_dir / mesh_name)
    for name in (mesh_name, urdf_name, *materials, *textures):
        if not (object_dir / name).is_file():
            raise ContractError(f"object file missing: {object_dir / name}")
        shutil.copy2(object_dir / name, staging / "core/object" / name)
    urdf_tree = ET.parse(staging / "core/object" / urdf_name)
    for mesh in urdf_tree.getroot().iter("mesh"):
        if Path(mesh.get("filename", "")).name != mesh_name:
            raise ContractError(f"{urdf_name}: references {mesh.get('filename')}, expected {mesh_name}")
        mesh.set("filename", mesh_name)
    urdf_tree.write(staging / "core/object" / urdf_name, xml_declaration=True, encoding="utf-8")
    vertices = obj_vertices(object_dir / mesh_name)
    object_rel = {name: f"core/object/{name}" for name in (mesh_name, urdf_name, *materials, *textures)}

    path_updates = {
        "raw_motion_file": "core/reconstruction/result.npz",
        "object_mesh_paths": [object_rel[mesh_name]],
        "object_urdf_paths": [object_rel[urdf_name]],
    }

    # Core: reconstruction (world frame) and the loader's MANO tracks (tracks frame).
    (staging / "core/reconstruction").mkdir(parents=True)
    shutil.copy2(bundle / "result.npz", staging / "core/reconstruction/result.npz")
    write_json(staging / "core/reconstruction/manifest.json", _scrub_paths(bundle_manifest))
    _write_row(tracks_table, {**tracks, **path_updates}, staging / "core/human/tracks.parquet")

    robot_entries: dict[str, Any] = {}
    reasons: list[str] = []
    statuses: list[str] = []
    for robot in selected:
        rule = ROBOTS[robot]
        table, motion = (first_table, first_motion) if robot == selected[0] else _one_row(available[robot])
        if [Path(p).name for p in motion["object_mesh_paths"]] != [mesh_name]:
            raise ContractError(f"robot {robot}: object mesh {motion['object_mesh_paths']} differs from {mesh_name}")
        report_path = run_root / rule.report.format(seq=sequence)
        if not report_path.is_file():
            raise ContractError(f"robot {robot}: retarget report missing at {report_path}")
        report = _load_json(report_path)
        t_robot = np.asarray(report.get("placement", {}).get("T_place"), dtype=float)
        support_path = run_root / STAGE / f"{sequence}_{robot}_support.usda"
        if not support_path.is_file():
            raise ContractError(f"robot {robot}: support surfaces missing at {support_path}")
        model_path = REPO_ROOT / rule.model
        base = f"robots/{robot}"
        _write_row(table, {**motion, **path_updates}, staging / base / "motion.parquet")
        write_json(staging / base / "support_surfaces.json", parse_support_usda(support_path, robot))
        write_json(staging / base / "retarget_report.json", _scrub_paths(report))

        qa_path = run_root / "simulate" / rule.embodiment / "replay_metrics.json"
        qa: dict[str, Any] = {"status": "not_run", "physics_replay": None, "flag_reasons": [], "completion": None}
        if qa_path.is_file():
            metrics = _load_json(qa_path)
            if metrics.get("schema") != "v2d_replay_metrics_v1" or not isinstance(metrics.get("flag"), bool):
                raise ContractError(f"{qa_path}: not a v2d_replay_metrics_v1 report with a boolean flag")
            write_json(staging / "qa" / robot / "physics_replay.json", metrics)
            qa = {
                "status": "flagged" if metrics["flag"] else "pass",
                "physics_replay": f"qa/{robot}/physics_replay.json",
                "flag_reasons": [str(r) for r in metrics.get("flag_reasons", [])],
                "completion": metrics.get("completion"),
            }
        statuses.append(qa["status"])
        reasons += [f"{robot}: {r}" for r in qa["flag_reasons"]] or ([f"{robot}: physics QA not run"] if qa["status"] == "not_run" else [])
        robot_entries[robot] = {
            "model": {"id": robot, "description_file": rule.model, "sha256": sha256_file(model_path)},
            "frame": {
                "description": f"{robot} placement world: robot root at the origin, z up (retarget report T_place)",
                "T_from_world": t_robot.tolist(),
            },
            "motion": {
                "path": f"{base}/motion.parquet",
                "format": "motion_v1/single_robot",
                "fps": float(motion["fps"]),
                "n_frames": len(motion["robot_joint_positions"]),
                "joint_count": len(motion["robot_joint_names"]),
            },
            "support_surfaces": f"{base}/support_surfaces.json",
            "retarget_report": f"{base}/retarget_report.json",
            "qa": qa,
        }

    gates: dict[str, Any] = {}
    for gate in GATES:
        source = run_root / "qa" / f"{gate}.json"
        if not source.is_file():
            gates[gate] = {"status": "not_run", "report": None, "reasons": []}
            reasons.append(f"{gate} gate not run")
            continue
        data = _load_json(source)
        status = "pass" if data.get("verdict") in ("accept", "pass") else "reject"
        write_json(staging / "qa" / f"{gate}.json", _scrub_paths(data))
        gates[gate] = {"status": status, "report": f"qa/{gate}.json", "reasons": [str(r) for r in data.get("reasons", [])]}
        reasons += [f"{gate}: {r}" for r in gates[gate]["reasons"]]
    verdict = package_verdict(statuses, [g["status"] for g in gates.values()])
    if verdict in ("rejected", "flagged") and not allow_flagged:
        raise ContractError(f"QA {verdict} the clip ({'; '.join(reasons)}); pass --allow-flagged to package it anyway")

    task_entry, task_reasons = _task(run_root, staging, sequence, context, first_motion, available, selected,
                                     instruction or workflow.get("instruction"), env_id)
    for reason in task_reasons:  # the package is still valid; --strict (delivery) requires the task
        print(f"warning: {reason}", file=sys.stderr)

    video = context["video"]
    manifest: dict[str, Any] = {
        "schema": PACKAGE_SCHEMA,
        "env_id": env_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sequence_id": sequence,
        "source": {
            "video_sha256": video.get("sha256"),
            "video_bytes": video.get("size"),
            "object_prompt": object_prompt or workflow.get("object_prompt") or bundle_manifest.get("sources", {}).get("object_prompt") or "",
            "instruction": instruction or workflow.get("instruction"),
            "task_profile": workflow.get("task_profile"),
        },
        "provenance": provenance,
        "frames": {
            "world": {
                "description": "reconstruction world: SLAM world rotated so gravity is -z (result.npz *_to_world)",
                "up_axis": "z",
                "units": "m",
            },
            "tracks": {
                "description": "loader frame: world translated so the object is centred in xy with its bottom at z = 0",
                "T_from_world": t_tracks.tolist(),
            },
        },
        "core": {
            "n_frames": n_frames,
            "fps": fps,
            "reconstruction": {
                "result": "core/reconstruction/result.npz",
                "manifest": "core/reconstruction/manifest.json",
                "format": "v2d_result_npz/1",
                "frame": "world",
            },
            "human_tracks": {
                "path": "core/human/tracks.parquet",
                "format": "v2d_loaded_mano/1",
                "fps": fps,
                "n_frames": len(tracks["mano_right_trans"]),
                "frame": "tracks",
                "mano": {
                    "flat_hand_mean": bool(tracks["mano_flat_hand_mean"]),
                    "use_pca": False,
                    "center_idx": tracks["mano_center_idx"],
                },
            },
            "object": {
                "name": first_motion["safe_object_name"] or sequence,
                "mesh": object_rel[mesh_name],
                "materials": [object_rel[m] for m in materials],
                "textures": [object_rel[t] for t in textures],
                "urdf": object_rel[urdf_name],
                "collision": {"mesh": object_rel[mesh_name], "source": "visual_mesh"},
                "physics": _urdf_physics(staging / "core/object" / urdf_name),
                "bounds_m": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
            },
        },
        "robots": robot_entries,
        "task": task_entry,
        "simulators": {},
        "qa": {"verdict": verdict, "reasons": reasons, "gates": gates},
        "files": [],
    }
    files = sorted(p for p in staging.rglob("*") if p.is_file())
    manifest["files"] = [
        {"path": p.relative_to(staging).as_posix(), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in files
    ]
    write_json(staging / "manifest.json", manifest)

    report = validate_package(staging)
    if report.errors:
        raise ContractError("built package failed validation:\n  " + "\n  ".join(report.errors))
    if out.exists():
        if not force:
            raise ContractError(f"{out} exists; pass --force to replace it")
        shutil.rmtree(out)
    staging.rename(out)
    return out
