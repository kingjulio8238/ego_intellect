"""tools.envpkg: env_package/v1 build + validation, intake/v1 validation.

Synthetic run roots in the layout the e2e stages write; every rejection case mutates one fact.
Run: python3 -m unittest tools.envpkg.tests.test_envpkg
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from tools.envpkg.build import build_package as _build_package
from tools.envpkg.build import parse_support_usda

from tools.envpkg.common import ContractError, schema_errors, sha256_file
from tools.envpkg.intake import validate_intake
from tools.envpkg.success import lift_hold
from tools.envpkg.task import load_task_spec
from tools.envpkg.validate import validate_package

REPO = Path(__file__).resolve().parents[3]


def build_package(*args, **kwargs):
    """The builder, with the MuJoCo export off for the 3-joint synthetic robot (its scene needs
    the real Vega joints; MuJoCoExportTest covers it)."""
    kwargs.setdefault("simulators", ("isaaclab",))
    return _build_package(*args, **kwargs)
SAMPLE_VIDEO = REPO / "reconstruction/assets/tissue_box.mp4"
SEQ = "box"
N, FPS, ROBOT_FPS = 91, 30.0, 50.0
N_ROBOT = int(round((N - 1) / FPS * ROBOT_FPS)) + 1
T_TRACKS = np.array([0.05, -0.4, 0.5])
T_PLACE = np.array([[-1.0, 0, 0, -0.04], [0, -1.0, 0, 0.56], [0, 0, 1.0, 1.28], [0, 0, 0, 1.0]])
SUPPORT = """#usda 1.0
(
    defaultPrim = "support_surfaces"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "support_surfaces"
{
    def Cylinder "box_0" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        uniform token axis = "Z"
        double height = 0.01
        double radius = 0.17
        double3 xformOp:translate = (0.0, 0.09, 0.75)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}
"""
CUBE_OBJ = "mtllib box.mtl\nusemtl m\n" + "".join(
    f"v {x} {y} {z}\n" for x in (-0.1, 0.1) for y in (-0.05, 0.05) for z in (-0.04, 0.04)
) + "f 1 2 3\nf 5 6 7\n"


def _write_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([row]), path)


PROFILE = json.loads((REPO / "robotic_grounding/groot_finetune/task_profiles/tissue_box_lift_hold.json").read_text())
ENV_SPEC = json.loads((Path(__file__).parent / "data/env_spec_vega_sharpa.json").read_text())


def env_spec(control_steps: int, joints: int, names: list[str] | None = None, **edit) -> dict:
    """The real Vega env dump (b3_tbseed), re-dimensioned for the synthetic clip."""
    spec = json.loads(json.dumps(ENV_SPEC))
    spec["runtime"].update(trajectory_end_timestep=control_steps - 1, num_timesteps=control_steps,
                           action_dim=joints, joint_names=names or [f"j{i}" for i in range(joints)])
    for key, value in edit.items():
        spec[key] = value
    return spec


def lift_profile(t: np.ndarray, height: float = 0.15) -> np.ndarray:
    """Rest, lift `height` m over 0.3 s, hold to 1.9 s, put down by 2.4 s (gently), rest."""
    return np.interp(t, [0.0, 0.5, 0.8, 1.9, 2.4, 3.0], [0.0, 0.0, height, height, 0.0, 0.0])


def make_run_root(root: Path, *, flag: bool = False, provenance: bool = False, track_rotation_deg: float = 0.0,
                  gates: dict | None = None, lift_m: float = 0.15, with_env_spec: bool = True,
                  env_spec_edit: dict | None = None) -> Path:
    t = np.arange(N) / FPS
    world = np.tile(np.eye(4), (N, 1, 1))
    world[:, :3, :3] = Rotation.from_euler("z", (20 * t)[:, None], degrees=True).as_matrix()
    world[:, :3, 3] = np.stack([0.1 * t, 0.2 + 0.0 * t, 0.7 + lift_profile(t, lift_m)], axis=1)
    bundle = root / "reconstruction/result_slam_gravity_aligned"
    bundle.mkdir(parents=True)
    hands = {}
    for side in ("left", "right"):
        hands[f"hand_{side}_wrist_to_world_transform"] = np.tile(np.eye(4), (N, 1, 1)).astype(np.float32)
        hands[f"hand_{side}_finger_pose"] = np.zeros((N, 15, 3), np.float32)
        hands[f"hand_{side}_is_valid"] = np.ones(N, bool)
    np.savez(
        bundle / "result.npz",
        camera_to_world_transform=np.tile(np.eye(4), (N, 1, 1)).astype(np.float32),
        camera_is_valid=np.ones(N, bool),
        object_to_world_transform=world.astype(np.float32),
        object_is_valid=np.ones(N, bool),
        **hands,
    )
    (bundle / "manifest.json").write_text(json.dumps({"n_frames": N, "sources": {"object_prompt": "a box", "frames_dir": "/e2e/x"}}))

    world32 = world.astype(np.float32).astype(float)
    rot = Rotation.from_euler("x", track_rotation_deg, degrees=True)
    track_pos = rot.apply(world32[:, :3, 3]) + T_TRACKS
    track_rot = (rot * Rotation.from_matrix(world32[:, :3, :3])).as_rotvec()
    tracks = {
        "raw_motion_file": "/data/result.npz", "fps": FPS, "mano_flat_hand_mean": True, "mano_center_idx": None,
        "object_mesh_paths": ["/data/processed/box.obj"], "object_urdf_paths": ["/data/processed/box_rigid.urdf"],
        "object_root_position": track_pos.tolist(), "object_root_axis_angle": track_rot.tolist(),
    }
    for side in ("left", "right"):
        tracks[f"mano_{side}_trans"] = np.zeros((N, 3)).tolist()
        tracks[f"mano_{side}_finger_pose"] = np.zeros((N, 45)).tolist()
        tracks[f"mano_{side}_joints"] = np.zeros((N, 21, 3)).tolist()
    _write_row(root / f"intermediate/ego_recon/loaded/sequence_id={SEQ}/robot_name=sharpa_wave/part-0.parquet", tracks)

    processed = root / "human_motion_data/ego_recon/processed"
    processed.mkdir(parents=True)
    (processed / "box.obj").write_text(CUBE_OBJ)
    (processed / "box.mtl").write_text("newmtl m\nmap_Kd box_0.png\n")
    (processed / "box_0.png").write_bytes(b"\x89PNG fake")
    (processed / "box.urdf").write_text(
        '<?xml version="1.0"?><robot name="box"><link name="object"><inertial><origin xyz="0 0 0"/>'
        '<mass value="0.3"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.003"/></inertial>'
        '<visual><geometry><mesh filename="box.obj"/></geometry></visual>'
        '<collision><geometry><mesh filename="box.obj"/></geometry></collision></link></robot>'
    )
    t_robot = np.arange(N_ROBOT) / ROBOT_FPS
    robot_obj = np.stack([np.interp(t_robot, t, world32[:, i, 3]) for i in range(3)], axis=1)
    robot_obj = robot_obj @ T_PLACE[:3, :3].T + T_PLACE[:3, 3]
    _write_row(processed / f"sequence_id={SEQ}/robot_name=vega_sharpa/data.parquet", {
        "schema_version": "motion_v1", "motion_kind": "single_robot", "fps": ROBOT_FPS,
        "raw_motion_file": "/work/result.npz", "safe_object_name": "box",
        "robot_joint_names": ["j0", "j1", "j2"], "robot_joint_positions": np.zeros((N_ROBOT, 3)).tolist(),
        "object_mesh_paths": [str(processed / "box.obj")], "object_urdf_paths": [str(processed / "box.urdf")],
        "object_root_position": robot_obj.tolist(),
    })
    stage = root / "human_motion_data/ego_recon/reconstructed_stage"
    stage.mkdir(parents=True)
    (stage / f"{SEQ}_vega_sharpa_support.usda").write_text(SUPPORT)
    (root / "vega_retarget_qa").mkdir()
    (root / f"vega_retarget_qa/{SEQ}_vega_sharpa_report.json").write_text(
        json.dumps({"placement": {"T_place": T_PLACE.tolist()}, "bundle_dir": "/work/x/bundle"}))
    (root / "simulate/vega").mkdir(parents=True)
    (root / "simulate/vega/replay_metrics.json").write_text(json.dumps({
        "schema": "v2d_replay_metrics_v1", "flag": flag, "flag_reasons": ["wrist_pos_p95_m"] if flag else [],
        "completion": 0.33,
        "thresholds": {"settle_object_pos_max_m": 0.025, "rest_object_pos_p95_m": 0.065, "grasp_missed_contact_mean": 0.5,
                       "grasp_object_pos_p95_m": 0.075, "wrist_pos_p95_m": 0.05, "grasp_drift_p95_m": 0.041},
        "checks": {"settle_object_pos_max_m": 0.01, "rest_object_pos_p95_m": 0.02, "grasp_missed_contact_mean": 0.0,
                   "grasp_object_pos_p95_m": 0.03, "wrist_pos_p95_m": 0.02, "grasp_drift_p95_m": 0.02},
    }))
    if with_env_spec:
        steps = int(round((N - 1) / FPS / 0.025)) + 1
        (root / "simulate/vega/env_spec.json").write_text(json.dumps(env_spec(steps, 3, **(env_spec_edit or {}))))
    (root / "qa").mkdir()
    for gate, verdict in (gates or {"triage": "accept", "reconstruction": "pass"}).items():
        if verdict is not None:
            (root / f"qa/{gate}.json").write_text(json.dumps({
                "schema": f"{gate}/v1", "verdict": verdict, "reasons": ["too blurred"] if verdict == "reject" else []}))
    if provenance:
        (root / "e2e_config.json").write_text(json.dumps({"workflow": {
            "sequence_id": SEQ, "instruction": "lift the box", "object_prompt": "a box", "task_profile": "box_lift"},
            "contracts": {"task": PROFILE}}))
        stage_record = {"completed_at": "2026-11-21T00:00:00+00:00", "repository": {"revision": "abc", "runtime_tree_sha256": "def"}}
        (root / "run_manifest.json").write_text(json.dumps({"stages": {
            "reconstruct": {**stage_record, "inputs": {"/v.mp4": {"sha256": "0" * 64, "size": 10}}},
            "retarget": stage_record, "simulate": stage_record}}))
    return root


def rehash(package: Path, mutate_manifest=None) -> None:
    """Recompute file hashes after a deliberate mutation (so only the targeted check fires)."""
    manifest = json.loads((package / "manifest.json").read_text())
    if mutate_manifest:
        mutate_manifest(manifest)
    for entry in manifest["files"]:
        path = package / entry["path"]
        if path.is_file():
            entry["bytes"], entry["sha256"] = path.stat().st_size, sha256_file(path)
    (package / "manifest.json").write_text(json.dumps(manifest))


def rewrite_row(path: Path, **updates) -> None:
    table = pq.read_table(path)
    row = {**table.to_pylist()[0], **updates}
    pq.write_table(pa.Table.from_pylist([row], schema=table.schema), path)


class PackageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.profile = self.tmp / "profile.json"
        self.profile.write_text(json.dumps(PROFILE))
        git = mock.patch("tools.envpkg.build.git_state", return_value=("f" * 40, False))
        git.start()
        self.addCleanup(git.stop)

    def build(self, **run_kwargs) -> Path:
        run = make_run_root(self.tmp / "run", **run_kwargs)
        build_kwargs = {"allow_missing_provenance": not run_kwargs.get("provenance", False)}
        if not run_kwargs.get("provenance", False):
            build_kwargs["task_profile"] = self.profile
        if run_kwargs.get("flag"):
            build_kwargs["allow_flagged"] = True
        return build_package(run, self.tmp / "pkg", **build_kwargs)

    def assertRejected(self, package: Path, fragment: str, strict: bool = False) -> None:
        report = validate_package(package, strict=strict)
        self.assertFalse(report.ok, "package unexpectedly valid")
        self.assertTrue(any(fragment in e for e in report.errors), f"{fragment!r} not in {report.errors}")

    def test_valid_package_and_contents(self) -> None:
        package = self.build()
        report = validate_package(package)
        self.assertTrue(report.ok, report.errors)
        manifest = json.loads((package / "manifest.json").read_text())
        self.assertEqual((manifest["qa"]["verdict"], manifest["qa"]["reasons"]), ("pass", []))
        self.assertEqual(manifest["qa"]["gates"]["triage"], {"status": "pass", "report": "qa/triage.json", "reasons": []})
        self.assertAlmostEqual(manifest["frames"]["tracks"]["T_from_world"][0][3], 0.05, places=6)
        self.assertEqual(manifest["core"]["object"]["physics"]["mass_source"], "default_constant")
        motion = pq.read_table(package / "robots/vega_sharpa/motion.parquet").to_pylist()[0]
        self.assertEqual(motion["object_urdf_paths"], ["core/object/box.urdf"])
        self.assertEqual(motion["raw_motion_file"], "core/reconstruction/result.npz")
        report_json = json.loads((package / "robots/vega_sharpa/retarget_report.json").read_text())
        self.assertEqual(report_json["bundle_dir"], "bundle")  # producer paths scrubbed
        self.assertTrue(any("provenance incomplete" in w for w in report.warnings))
        self.assertRejected(package, "provenance incomplete", strict=True)

    def test_strict_passes_with_full_provenance(self) -> None:
        package = self.build(provenance=True)
        report = validate_package(package, strict=True)
        self.assertTrue(report.ok, report.errors)

    def test_build_refuses_missing_provenance_and_flagged_clips(self) -> None:
        run = make_run_root(self.tmp / "run", flag=True)
        with self.assertRaisesRegex(ContractError, "provenance incomplete"):
            build_package(run, self.tmp / "pkg")
        with self.assertRaisesRegex(ContractError, "QA flagged"):
            build_package(run, self.tmp / "pkg", allow_missing_provenance=True)
        self.assertFalse((self.tmp / "pkg").exists())
        package = build_package(run, self.tmp / "pkg", allow_missing_provenance=True, allow_flagged=True)
        self.assertTrue(validate_package(package).ok)
        self.assertRejected(package, "delivery requires 'pass'", strict=True)

    def test_gate_verdicts(self) -> None:
        run = make_run_root(self.tmp / "run", gates={"triage": "accept", "reconstruction": "reject"})
        with self.assertRaisesRegex(ContractError, "QA rejected"):
            build_package(run, self.tmp / "pkg", allow_missing_provenance=True)
        package = build_package(run, self.tmp / "pkg", allow_missing_provenance=True, allow_flagged=True)
        manifest = json.loads((package / "manifest.json").read_text())
        self.assertEqual(manifest["qa"]["verdict"], "rejected")
        self.assertIn("reconstruction: too blurred", manifest["qa"]["reasons"])
        self.assertTrue(validate_package(package).ok)
        rehash(package, lambda m: m["qa"]["gates"]["reconstruction"].update(status="pass"))
        self.assertRejected(package, "disagrees with the report verdict")
        run2 = make_run_root(self.tmp / "run2", gates={"triage": None, "reconstruction": "pass"})
        package2 = build_package(run2, self.tmp / "pkg2", allow_missing_provenance=True)
        manifest2 = json.loads((package2 / "manifest.json").read_text())
        self.assertEqual((manifest2["qa"]["verdict"], manifest2["qa"]["gates"]["triage"]["status"]), ("incomplete", "not_run"))
        self.assertRejected(package2, "delivery requires 'pass'", strict=True)

    def test_task_spec(self) -> None:
        package = self.build(provenance=True)
        manifest = json.loads((package / "manifest.json").read_text())
        self.assertEqual(manifest["task"], {"spec": "task/task.json", "schema": "task_spec/v1"})
        spec = load_task_spec(package)
        part = spec["robots"]["vega_sharpa"]
        self.assertEqual(spec["task_id"], f"{manifest['env_id']}.tissue_box_lift_hold")
        self.assertEqual((part["reference"]["control_steps"], part["episode"]["evaluation"]["max_steps"]), (121, 121))
        self.assertTrue(part["reference_check"]["success"])
        self.assertEqual({r["name"]: r["weight"] for r in part["rewards"]}["force_closure"], 5.0)
        self.assertEqual(part["action"]["dim"], 3)
        self.assertEqual(part["assist"]["virtual_object_control"]["curriculum"]["virtual_object_control_scale_factor"][0], 1.0)
        self.assertTrue(validate_package(package, strict=True).ok)

        def redefine(m):
            path = package / "task/task.json"
            task = json.loads(path.read_text())
            task["robots"]["vega_sharpa"]["rewards"][0]["definition"] = "something else"
            path.write_text(json.dumps(task))

        rehash(package, redefine)
        self.assertRejected(package, "is not described as catalogued")

    def test_episode_cap_follows_the_terminations(self) -> None:
        self.assertIsNone(load_task_spec(self.build())["robots"]["vega_sharpa"]["episode"]["train"]["max_steps"])
        terms = dict(ENV_SPEC["terminations"])
        terms["time_out"] = {"func": "isaaclab.envs.mdp.terminations:time_out", "time_out": True, "params": {}}
        run = make_run_root(self.tmp / "capped", env_spec_edit={"terminations": terms})
        part = load_task_spec(build_package(run, self.tmp / "pkg2", allow_missing_provenance=True,
                                            task_profile=self.profile))["robots"]["vega_sharpa"]
        self.assertEqual((part["episode"]["train"]["max_steps"], part["notes"]), (400, []))  # 400 > 121 + 50
        runtime = {**env_spec(121, 3)["runtime"], "train_max_episode_steps": 100}
        run = make_run_root(self.tmp / "short", env_spec_edit={"terminations": terms, "runtime": runtime})
        part = load_task_spec(build_package(run, self.tmp / "pkg3", allow_missing_provenance=True,
                                            task_profile=self.profile))["robots"]["vega_sharpa"]
        self.assertIn("capped at 100 steps", part["notes"][0])

    def test_isaaclab_export(self) -> None:
        package = self.build(provenance=True)
        manifest = json.loads((package / "manifest.json").read_text())
        self.assertEqual(manifest["simulators"]["isaaclab"], {"env": "isaaclab/env.json", "schema": "isaaclab_env/v1"})
        env = json.loads((package / "isaaclab/env.json").read_text())
        part = env["robots"]["vega_sharpa"]
        self.assertEqual(part["motion_file"], f"ego_recon/processed/sequence_id={SEQ}/robot_name=vega_sharpa")
        self.assertEqual(part["task"], "VegaSharpa-WholeBody-Manip-v0")
        hmd = package / "isaaclab/human_motion_data"
        motion = pq.read_table(hmd / part["motion_file"] / "data.parquet").to_pylist()[0]
        self.assertEqual(motion["object_urdf_paths"], ["ego_recon/processed/box.urdf"])
        for name in ("box.urdf", "box.obj", "box.mtl", "box_0.png"):  # found by name beside the motion
            self.assertTrue((hmd / "ego_recon/processed" / name).is_file(), name)
        support = parse_support_usda(hmd / f"ego_recon/reconstructed_stage/{SEQ}_vega_sharpa_support.usda", "vega_sharpa")
        self.assertEqual(support, json.loads((package / "robots/vega_sharpa/support_surfaces.json").read_text()))
        self.assertTrue(validate_package(package, strict=True).ok)

        usda = hmd / f"ego_recon/reconstructed_stage/{SEQ}_vega_sharpa_support.usda"
        usda.write_text(usda.read_text().replace("double radius = 0.17", "double radius = 0.2"))
        rehash(package)
        self.assertRejected(package, "support USDA states different disks")

    def test_isaaclab_export_motion_must_be_the_core_motion(self) -> None:
        package = self.build()
        path = package / f"isaaclab/human_motion_data/ego_recon/processed/sequence_id={SEQ}/robot_name=vega_sharpa/data.parquet"
        joints = (np.asarray(pq.read_table(path).to_pylist()[0]["robot_joint_positions"]) + 0.01).tolist()
        rewrite_row(path, robot_joint_positions=joints)
        rehash(package)
        self.assertRejected(package, "exported motion differs from the core motion in ['robot_joint_positions']")

    def test_isaaclab_check_comparisons(self) -> None:
        from tools.envpkg.isaaclab_check import compare_env, compare_replay

        spec = env_spec(121, 3)
        moved = json.loads(json.dumps(spec))
        moved["robot"]["spawn"]["asset_path"] = "/elsewhere/vega_sharpa_reduced.urdf"  # paths do not count
        self.assertEqual(compare_env(spec, moved), [])
        moved["rewards"]["force_closure"]["weight"] = 1.0
        moved["runtime"]["action_dim"] = 4
        self.assertEqual(len(compare_env(spec, moved)), 2)
        expected = {"flag": False, "completion": 0.33, "checks": {"settle_object_pos_max_m": 0.013, "grasp_missed_contact_mean": 0.0}}
        same = {"flag": False, "completion": 0.335, "checks": {"settle_object_pos_max_m": 0.0145, "grasp_missed_contact_mean": 0.02}}
        self.assertEqual(compare_replay(expected, same), [])
        off = {"flag": True, "flag_reasons": ["settle"], "completion": 0.2, "checks": {"settle_object_pos_max_m": 0.04}}
        self.assertEqual(len(compare_replay(expected, off)), 4)

    def test_task_spec_refusals(self) -> None:
        run = make_run_root(self.tmp / "flat", lift_m=0.0)
        with self.assertRaisesRegex(ContractError, "reference motion itself fails the lift_hold"):
            build_package(run, self.tmp / "pkg", allow_missing_provenance=True, task_profile=self.profile)
        rewards = dict(ENV_SPEC["rewards"])
        rewards["novel"] = {"func": "my.module:novel_reward", "weight": 1.0, "params": {}}
        run = make_run_root(self.tmp / "novel", env_spec_edit={"rewards": rewards})
        with self.assertRaisesRegex(ContractError, "novel_reward, which has no definition"):
            build_package(run, self.tmp / "pkg", allow_missing_provenance=True, task_profile=self.profile)
        run = make_run_root(self.tmp / "nospec", with_env_spec=False)
        package = build_package(run, self.tmp / "pkg", allow_missing_provenance=True, task_profile=self.profile)
        self.assertIsNone(json.loads((package / "manifest.json").read_text())["task"])
        self.assertRejected(package, "no task spec attached", strict=True)

    def test_lift_hold_matches_the_reference_implementation(self) -> None:
        sys.path.insert(0, str(REPO / "robotic_grounding"))
        from groot_finetune.closed_loop.evaluation import evaluate_lift_hold
        from groot_finetune.task_profile import LiftHoldEvaluator

        params = {k: v for k, v in PROFILE["evaluator"].items() if k != "id"}
        config = LiftHoldEvaluator(**params)
        rng = np.random.default_rng(0)
        traces = [lift_profile(np.linspace(0, 3, n), h) + rng.normal(0, noise, n)
                  for n, h, noise in zip(rng.integers(2, 160, 150), rng.uniform(0, 0.25, 150), rng.uniform(0, 0.004, 150))]
        traces += [np.cumsum(rng.normal(0, 0.01, n)) for n in rng.integers(1, 120, 50)]
        for trace in traces:
            a, b = evaluate_lift_hold(trace, config=config), lift_hold(trace, params)
            self.assertEqual((a.success, a.lift_hold_success, a.max_hold_steps), (b["success"], b["lift_hold_success"], b["max_hold_steps"]))
            self.assertAlmostEqual(a.max_lift_m, b["max_lift_m"], places=4)
        self.assertGreaterEqual(sum(lift_hold(t, params)["success"] for t in traces), 5)  # both outcomes covered

    def test_build_refuses_tracks_in_a_rotated_frame(self) -> None:
        run = make_run_root(self.tmp / "run", track_rotation_deg=5.0)
        with self.assertRaisesRegex(ContractError, "constant translation"):
            build_package(run, self.tmp / "pkg", allow_missing_provenance=True)

    def test_tampered_extra_and_missing_files(self) -> None:
        package = self.build()
        with open(package / "core/object/box.obj", "a") as stream:
            stream.write("# edited\n")
        self.assertRejected(package, "box.obj: ")
        rehash(package)  # hashes fixed, but the Isaac Lab copy of the mesh no longer matches
        self.assertRejected(package, "processed/box.obj is not a copy of core/object/box.obj")
        shutil.copy(package / "core/object/box.obj", package / "isaaclab/human_motion_data/ego_recon/processed/box.obj")
        rehash(package)
        self.assertTrue(validate_package(package).ok)
        (package / "core/object/stray.txt").write_text("x")
        self.assertRejected(package, "stray.txt: present but not listed")
        (package / "core/object/stray.txt").unlink()
        (package / "core/object/box_0.png").unlink()
        self.assertRejected(package, "box_0.png: listed but missing")

    def test_robot_motion_that_disagrees_with_the_reconstruction(self) -> None:
        package = self.build()
        path = package / "robots/vega_sharpa/motion.parquet"
        moved = (np.asarray(pq.read_table(path).to_pylist()[0]["object_root_position"]) + [0.005, 0, 0]).tolist()
        rewrite_row(path, object_root_position=moved)
        rehash(package)
        self.assertRejected(package, "5.0 mm from frame.T_from_world")

    def test_absolute_asset_path_in_motion(self) -> None:
        package = self.build()
        rewrite_row(package / "robots/vega_sharpa/motion.parquet", object_urdf_paths=["/work/box.urdf"])
        rehash(package)
        self.assertRejected(package, "is not a package file")

    def test_manifest_facts(self) -> None:
        package = self.build()
        package2 = self.tmp / "pkg2"
        shutil.copytree(package, package2)

        def skew(m):
            m["robots"]["vega_sharpa"]["frame"]["T_from_world"][0][0] = 2.0

        rehash(package, skew)
        self.assertRejected(package, "not orthonormal")
        rehash(package2, lambda m: m["robots"]["vega_sharpa"]["qa"].update(status="flagged"))
        self.assertRejected(package2, "disagrees with the replay flag")

    def test_schema_rejects_escaping_paths(self) -> None:
        package = self.build()
        manifest = json.loads((package / "manifest.json").read_text())
        manifest["core"]["object"]["mesh"] = "../outside.obj"
        self.assertTrue(any("core/object/mesh" in e for e in schema_errors("env_package/v1", manifest)))

    def test_support_usda_rejects_unsupported_transforms(self) -> None:
        path = self.tmp / "support.usda"
        path.write_text(SUPPORT.replace('["xformOp:translate"]', '["xformOp:translate", "xformOp:rotateXYZ"]'))
        with self.assertRaisesRegex(ContractError, "only translate"):
            parse_support_usda(path, "vega_sharpa")
        path.write_text(SUPPORT.replace('def Cylinder "box_0"', 'def Cube "box_0"'))
        with self.assertRaisesRegex(ContractError, "Cylinder disks"):
            parse_support_usda(path, "vega_sharpa")


def _file_entry(path: Path, rel: str) -> dict:
    return {"path": rel, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


class MuJoCoExportTest(unittest.TestCase):
    """The MuJoCo export and replay on the real Vega robot (static pose, object on its support)."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        git = mock.patch("tools.envpkg.build.git_state", return_value=("f" * 40, False))
        git.start()
        self.addCleanup(git.stop)

    def vega_run(self) -> Path:
        """The synthetic clip on the real Vega robot: the robot holds its default pose, the object
        starts resting on a support disk placed under it (robot frame) and its reference lifts."""
        import mujoco

        from tools.envpkg.mujoco_export import _robot_spec

        run = make_run_root(self.tmp / "run", provenance=True)
        names = ENV_SPEC["runtime"]["joint_names"]
        pose = ENV_SPEC["robot"]["init_state"]["joint_pos"]
        q = np.array([pose.get(n, 0.0) for n in names])
        urdf = REPO / "robotic_grounding/source/robotic_grounding/robotic_grounding/assets/urdfs/vega_sharpa/vega_sharpa_reduced.urdf"
        spec, _, _ = _robot_spec(urdf, "assets")
        model = spec.compile()
        data = mujoco.MjData(model)
        for name, value in zip(names, q):
            data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
        mujoco.mj_forward(model, data)
        ee = [np.r_[data.xpos[model.body(b).id], data.xquat[model.body(b).id]] for b in ("L_arm_l7", "R_arm_l7")]
        path = run / f"human_motion_data/ego_recon/processed/sequence_id={SEQ}/robot_name=vega_sharpa/data.parquet"
        row = pq.read_table(path).to_pylist()[0]
        n = len(row["object_root_position"])
        obj = np.asarray(row["object_root_position"])
        world_rot = Rotation.from_euler("z", (20 * np.arange(n) / ROBOT_FPS)[:, None], degrees=True)
        wxyz = (Rotation.from_matrix(T_PLACE[:3, :3]) * world_rot).as_quat()[:, [3, 0, 1, 2]]
        row.update(
            robot_joint_names=names, robot_joint_positions=np.tile(q, (n, 1)).tolist(),
            ee_link_names=["L_arm_l7", "R_arm_l7"], ee_pose_w=np.tile(np.stack(ee), (n, 1, 1)).tolist(),
            object_body_position=obj[:, None].tolist(), object_body_wxyz=wxyz[:, None].tolist(),
            hand_contact_active=np.zeros((2, n)).tolist(),
        )
        pq.write_table(pa.Table.from_pylist([row]), path)
        top = obj[0][2] - 0.04  # the cube's bottom at frame 0
        support = run / f"human_motion_data/ego_recon/reconstructed_stage/{SEQ}_vega_sharpa_support.usda"
        support.write_text(SUPPORT.replace("(0.0, 0.09, 0.75)", f"({obj[0][0]}, {obj[0][1]}, {top - 0.005})"))
        steps = int(round((N - 1) / FPS / 0.025)) + 1
        (run / "simulate/vega/env_spec.json").write_text(json.dumps(env_spec(steps, len(names), names)))
        return run

    def test_scene_and_replay(self) -> None:
        import mujoco

        package = _build_package(self.vega_run(), self.tmp / "pkg", allow_flagged=True)
        manifest = json.loads((package / "manifest.json").read_text())
        self.assertEqual(sorted(manifest["simulators"]), ["isaaclab", "mujoco"])
        env = json.loads((package / "mujoco/env.json").read_text())["robots"]["vega_sharpa"]
        model = mujoco.MjModel.from_xml_path(str(package / env["scene"]))
        self.assertEqual((model.nu, model.njnt), (58, 59))  # 58 servos; robot joints + the object's free joint
        self.assertEqual(len(env["hand_contact_bodies"]["left"]), len(env["hand_contact_bodies"]["right"]))
        self.assertEqual(sum(j["delay_physics_steps"] == 4 for j in env["joints"]), 14)  # the arms
        qa = manifest["robots"]["vega_sharpa"]["qa"]["mujoco"]
        report = json.loads((package / qa["report"]).read_text())
        self.assertEqual(report["checks"]["initial_penetration_m"], 0.0)
        self.assertLess(report["checks"]["settle_object_pos_max_m"], 0.03)  # rests; the reference moves 2.5 cm meanwhile
        self.assertLess(report["checks"]["wrist_pos_p95_m"], 0.01)  # the servos hold the pose
        self.assertEqual(report["termination_terms"], ["object_pos_error"])  # the reference lifts away
        self.assertTrue(report["flag"])  # never reaches a reference contact
        self.assertFalse(qa["agree"])  # ... while the fixture's Isaac report passes
        self.assertEqual(manifest["qa"]["verdict"], "flagged")
        self.assertTrue(validate_package(package).ok)

    def test_agreement_limits(self) -> None:
        from tools.envpkg.mujoco_replay import agreement

        isaac = {"flag": False, "completion": 0.33, "checks": {"settle_object_pos_max_m": 0.013, "wrist_pos_p95_m": 0.032}}
        close = {"flag": False, "completion": 0.32, "checks": {"settle_object_pos_max_m": 0.022, "wrist_pos_p95_m": 0.030}}
        self.assertTrue(agreement(close, isaac)["agree"])
        far = {"flag": False, "completion": 0.20, "checks": {"settle_object_pos_max_m": 0.04, "wrist_pos_p95_m": 0.030}}
        self.assertEqual(len(agreement(far, isaac)["reasons"]), 2)
        self.assertFalse(agreement({**close, "flag": True}, isaac)["agree"])


class DeliveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        git = mock.patch("tools.envpkg.build.git_state", return_value=("f" * 40, False))
        git.start()
        self.addCleanup(git.stop)

    def export(self, name: str, **run_kwargs) -> Path:
        from tools.envpkg.delivery import export_run

        run = make_run_root(self.tmp / name, provenance=True, **run_kwargs)
        export_run(run, simulators=("isaaclab",))
        return run

    def test_archive_is_deterministic_and_verifies(self) -> None:
        from tools.envpkg.delivery import archive
        from tools.envpkg.verify_env import main as verify, safe_extract

        run = self.export("a")
        report = json.loads((run / "qa/export.json").read_text())
        self.assertEqual(report["verdict"], "accept")
        package = run / "package/env"
        self.assertTrue((package / "verify/verify_env.py").is_file())
        self.assertTrue((package / "verify/envpkg/schemas/env_package_v1.schema.json").is_file())
        again = archive(package, self.tmp / "again")
        self.assertEqual(sha256_file(again), report["sha256"])  # same package, same bytes
        extracted = safe_extract(again, self.tmp / "x")
        self.assertEqual(extracted.name, report["env_id"])
        self.assertEqual(verify([str(again)]), 0)
        self.assertEqual(verify([str(package)]), 0)
        self.assertFalse(list(package.rglob("__pycache__")))  # verifying never writes into the package

    def test_unsafe_archives_are_refused(self) -> None:
        import io
        import tarfile

        from tools.envpkg.verify_env import safe_extract

        for member in ("../escape.txt", "/abs.txt"):
            bad = self.tmp / "bad.tar.gz"
            with tarfile.open(bad, "w:gz") as tar:
                info = tarfile.TarInfo(member)
                info.size = 1
                tar.addfile(info, io.BytesIO(b"x"))
            with self.assertRaisesRegex(ValueError, "unsafe archive member"):
                safe_extract(bad, self.tmp / "out")

    def test_export_refuses_flagged_clips(self) -> None:
        from tools.envpkg.__main__ import main as cli
        from tools.envpkg.delivery import QARefused, export_run

        run = make_run_root(self.tmp / "f", provenance=True, flag=True)
        with self.assertRaises(QARefused):
            export_run(run, simulators=("isaaclab",))
        self.assertEqual(json.loads((run / "qa/export.json").read_text())["verdict"], "reject")
        self.assertFalse((run / "package/env").exists())
        args = ["export", "--run-root", str(run)]
        with mock.patch("tools.envpkg.delivery.export_run", side_effect=QARefused("QA flagged")):
            self.assertEqual(cli(args), 3)
            self.assertEqual(cli([*args, "--exit-zero"]), 0)

    def test_deliver(self) -> None:
        from tools.envpkg.delivery import deliver
        from tools.envpkg.verify_env import main as verify

        good = self.export("good")
        flagged = make_run_root(self.tmp / "flagged", provenance=True, flag=True)
        try:
            from tools.envpkg.delivery import export_run

            export_run(flagged, simulators=("isaaclab",))
        except ContractError:
            pass
        out = self.tmp / "delivery"
        manifest = deliver([good, flagged, self.tmp / "missing"], out, delivery_id="d1")
        self.assertEqual(len(manifest["environments"]), 1)
        self.assertEqual({s["run"] for s in manifest["skipped"]}, {"flagged", "missing"})
        archive = out / manifest["environments"][0]["archive"]
        self.assertIn(archive.name, (out / "SHA256SUMS").read_text())
        self.assertTrue((out / "verify/verify_env.py").is_file() and (out / "README.md").is_file())
        self.assertEqual(verify([str(archive)]), 0)
        with open(archive, "ab") as stream:
            stream.write(b"tampered")
        self.assertEqual(verify([str(archive)]), 1)  # SHA256SUMS mismatch
        with self.assertRaisesRegex(ContractError, "not empty"):
            deliver([good], out, delivery_id="d1")
        with self.assertRaisesRegex(ContractError, "redistribution of vega_sharpa's assets is not confirmed"):
            deliver([good], self.tmp / "external", delivery_id="d2", external=True)


class IntakeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        (self.tmp / "videos").mkdir()
        video = self.tmp / "videos/c1.mp4"
        if SAMPLE_VIDEO.is_file():
            shutil.copy(SAMPLE_VIDEO, video)
        else:
            video.write_bytes(b"not a video")
        self.manifest = {
            "schema": "intake/v1", "delivery_id": "d1", "created_at": "2026-11-21T00:00:00Z",
            "vendor": {"id": "v1", "name": "Vendor"},
            "license": {"terms": "contract 7", "commercial_use": True, "derivatives_allowed": True,
                        "consent_obtained": True, "pii_reviewed": True},
            "clips": [{
                "clip_id": "c1", "video": _file_entry(video, "videos/c1.mp4"),
                "task": {"object_prompt": "a tissue box", "instruction": "lift the box and hold it", "category": "lift_hold"},
                "capture": {"device": "glasses-x", "mount": "glasses"},
                "content": {"hands_visible": "both", "single_rigid_object": True, "faces_visible": False},
            }],
        }

    def check(self, probe: bool = False):
        path = self.tmp / "intake.json"
        path.write_text(json.dumps(self.manifest))
        return validate_intake(path, probe=probe)

    def test_valid_delivery_without_probe(self) -> None:
        report = self.check()
        self.assertTrue(report.ok, report.errors)

    @unittest.skipUnless(SAMPLE_VIDEO.is_file() and shutil.which("ffprobe"), "needs the sample video and ffprobe")
    def test_probe_accepts_the_sample_clip(self) -> None:
        report = self.check(probe=True)
        self.assertTrue(report.ok, report.errors)
        info = report.clips["c1"]["video"]
        self.assertEqual((info["width"], info["height"], round(info["fps"]), info["frames"]), (1280, 720, 30, 390))

    def test_rejections(self) -> None:
        clip = self.manifest["clips"][0]
        clip["video"]["sha256"] = "0" * 64
        self.assertIn("sha256 mismatch", " ".join(self.check().errors))
        self.setUp()
        clip = self.manifest["clips"][0]
        clip["capture"]["intrinsics"] = {"model": "pinhole", "width": 1280, "height": 720, "fx": 900, "fy": 900,
                                         "cx": 1400, "cy": 360, "distortion": []}
        self.assertIn("principal point outside", " ".join(self.check().errors))
        clip["capture"].pop("intrinsics")
        clip["content"]["faces_visible"] = True
        self.assertTrue(any("faces_visible" in e for e in self.check().errors))
        clip["content"]["faces_visible"] = False
        self.manifest["clips"].append(dict(clip))
        self.assertIn("appears more than once", " ".join(self.check().errors))

    def test_imu_must_cover_the_video(self) -> None:
        imu = self.tmp / "imu.csv"
        imu.write_text("t_s,ax,ay,az,gx,gy,gz\n" + "".join(f"{i / 200},0,0,9.8,0,0,0\n" for i in range(400)))
        clip = self.manifest["clips"][0]
        clip["capture"]["imu"] = {"file": _file_entry(imu, "imu.csv"), "format": "imu_csv/1", "time_offset_s": 0.0}
        self.assertTrue(self.check().ok)  # unprobed: duration unknown, rate and header checked
        if SAMPLE_VIDEO.is_file() and shutil.which("ffprobe"):
            self.assertIn("IMU covers", " ".join(self.check(probe=True).errors))  # 2 s of IMU, 13 s video


if __name__ == "__main__":
    unittest.main()
