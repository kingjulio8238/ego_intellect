"""Isaac Lab export (isaaclab_env/v1): the package's ``isaaclab/`` folder.

Writes the layout robotic_grounding's Isaac Lab extension already loads, unmodified:

    isaaclab/env.json                                        isaaclab_env/v1 (below)
    isaaclab/human_motion_data/ego_recon/processed/sequence_id=<seq>/robot_name=<robot>/data.parquet
    isaaclab/human_motion_data/ego_recon/processed/<object>.urdf, .obj, .mtl, textures
    isaaclab/human_motion_data/ego_recon/reconstructed_stage/<seq>_<robot>_support.usda

so a lab runs any robotic_grounding entry point with
``HUMAN_MOTION_DATA_DIR=<package>/isaaclab/human_motion_data`` and
``--motion_file ego_recon/processed/sequence_id=<seq>/robot_name=<robot>``. The motion is the
core motion with its asset paths re-pointed (relative to HUMAN_MOTION_DATA_DIR); the support
USDA is generated from the simulator-neutral support_surfaces.json. The object files are
copies of core/object/: the extension's loader finds object assets by file name beside the
motion, so the copy keeps it unmodified (~2 MB per package).
``tools/envpkg/isaaclab_check.py`` proves the export in a real Isaac Lab process.
Contract: docs/pipeline/ISAACLAB_EXPORT.md.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

from .common import ContractError, write_json

ISAACLAB_SCHEMA = "isaaclab_env/v1"
ROOT = PurePosixPath("isaaclab")
HMD = ROOT / "human_motion_data"
DATASET = "ego_recon"
EXTENSION_PATH = "robotic_grounding/source/robotic_grounding"
# Fields of a motion_v1 row that hold asset paths; everything else is copied verbatim.
PATH_FIELDS = ("raw_motion_file", "object_mesh_paths", "object_urdf_paths")


def partition(sequence: str, robot: str) -> PurePosixPath:
    return PurePosixPath(DATASET) / "processed" / f"sequence_id={sequence}" / f"robot_name={robot}"


def support_usda(surfaces: dict[str, Any]) -> str:
    """The support USDA scripts/reconstruct_support_surfaces.py would write for these disks."""
    lines = ["#usda 1.0", "(", '    defaultPrim = "support_surfaces"', "    metersPerUnit = 1",
             '    upAxis = "Z"', ")", "", 'def Xform "support_surfaces"', "{"]
    for disk in surfaces["surfaces"]:
        if disk["shape"] != "disk" or disk["axis"] != "z":
            raise ContractError(f"support surface {disk['name']}: only z-axis disks are exported")
        x, y, z = (repr(float(v)) for v in disk["center"])
        lines += [
            f'    def Cylinder "{disk["name"]}" (',
            '        prepend apiSchemas = ["PhysicsCollisionAPI"]',
            "    )",
            "    {",
            '        uniform token axis = "Z"',
            f"        double height = {float(disk['thickness'])!r}",
            f"        double radius = {float(disk['radius'])!r}",
            f"        double3 xformOp:translate = ({x}, {y}, {z})",
            '        uniform token[] xformOpOrder = ["xformOp:translate"]',
            "    }",
        ]
    return "\n".join(lines + ["}", ""])


def export_isaaclab(staging: Path, manifest: dict[str, Any], *, extension_revision: str | None) -> dict[str, Any]:
    """Write ``isaaclab/`` into a package being assembled; returns the manifest's
    ``simulators.isaaclab`` entry. ``manifest`` needs core, robots, sequence_id, task."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    sequence = manifest["sequence_id"]
    core_object = manifest["core"]["object"]
    processed = HMD / DATASET / "processed"

    # Object files beside the motion, where the extension's loader finds them by name.
    import shutil

    (staging / processed).mkdir(parents=True, exist_ok=True)
    object_files = [core_object["mesh"], core_object["urdf"], *core_object["materials"], *core_object["textures"]]
    for source in object_files:
        shutil.copy2(staging / source, staging / processed / PurePosixPath(source).name)
    urdf_name = PurePosixPath(core_object["urdf"]).name
    mesh_name = PurePosixPath(core_object["mesh"]).name

    task = None
    if manifest.get("task"):
        task = json.loads((staging / manifest["task"]["spec"]).read_text())
    robots: dict[str, Any] = {}
    for robot, entry in manifest["robots"].items():
        table = pq.read_table(staging / entry["motion"]["path"])
        row = table.to_pylist()[0]
        bodies = len(row.get("object_urdf_paths") or [])
        row.update({  # relative to HUMAN_MOTION_DATA_DIR
            "raw_motion_file": f"../../{manifest['core']['reconstruction']['result']}",
            "object_urdf_paths": [f"{DATASET}/processed/{urdf_name}"] * bodies,
            "object_mesh_paths": [f"{DATASET}/processed/{mesh_name}"] * len(row.get("object_mesh_paths") or []),
        })
        motion_rel = partition(sequence, robot)
        out = staging / HMD / motion_rel / "data.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist([row], schema=table.schema), out)

        surfaces = json.loads((staging / entry["support_surfaces"]).read_text())
        support = HMD / DATASET / "reconstructed_stage" / f"{sequence}_{robot}_support.usda"
        (staging / support).parent.mkdir(parents=True, exist_ok=True)
        (staging / support).write_text(support_usda(surfaces))

        part = (task or {}).get("robots", {}).get(robot)
        if part is None:
            raise ContractError(f"isaaclab export: no task spec for {robot}; the Isaac Lab task id comes from it")
        qa = entry["qa"]
        expected = None
        if qa["physics_replay"]:
            replay = json.loads((staging / qa["physics_replay"]).read_text())
            expected = {"flag": replay["flag"], "completion": replay["completion"], "checks": replay.get("checks", {}),
                        "thresholds": replay.get("thresholds", {})}
        task_id = part["isaaclab_task"]
        motion_arg = str(motion_rel)
        env_prefix = "HUMAN_MOTION_DATA_DIR=<package>/isaaclab/human_motion_data"
        robots[robot] = {
            "task": task_id,
            "motion_file": motion_arg,
            "motion": str(HMD / motion_rel / "data.parquet"),
            "support_usda": str(support),
            "object_urdf": str(processed / urdf_name),
            "robot_description_sha256": entry["model"]["sha256"],
            "env_spec": part["env_spec"],
            "expected_replay": expected,
            "commands": {
                "replay_check": f"{env_prefix} python scripts/rsl_rl/replay_metrics.py --headless --device cpu "
                                f"--task {task_id} --motion_file {motion_arg} --output <out>/replay_metrics.json",
                "train": f"{env_prefix} python scripts/rsl_rl/train.py --headless --task {task_id} "
                         f"--motion_file {motion_arg} --num_envs 4096",
                "play": f"{env_prefix} python scripts/rsl_rl/eval.py --headless --task {task_id} "
                        f"--motion_file {motion_arg} --num_envs 1 --checkpoint <model.pt>",
            },
        }
    env = {
        "schema": ISAACLAB_SCHEMA,
        "human_motion_data_dir": str(HMD),
        "extension": {
            "name": "robotic_grounding",
            "repository_path": EXTENSION_PATH,
            "revision": extension_revision,
            "note": "run the commands from the extension's checkout root (robotic_grounding/) inside its Isaac Lab image",
        },
        "robots": robots,
    }
    write_json(staging / ROOT / "env.json", env)
    return {"env": str(ROOT / "env.json"), "schema": ISAACLAB_SCHEMA}
