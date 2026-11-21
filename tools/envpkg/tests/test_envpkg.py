"""tools.envpkg: env_package/v1 build + validation, intake/v1 validation.

Synthetic run roots in the layout the e2e stages write; every rejection case mutates one fact.
Run: python3 -m unittest tools.envpkg.tests.test_envpkg
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from tools.envpkg.build import build_package, parse_support_usda
from tools.envpkg.common import ContractError, schema_errors, sha256_file
from tools.envpkg.intake import validate_intake
from tools.envpkg.validate import validate_package

REPO = Path(__file__).resolve().parents[3]
SAMPLE_VIDEO = REPO / "reconstruction/assets/tissue_box.mp4"
SEQ = "box"
N, FPS, ROBOT_FPS = 31, 30.0, 50.0
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


def make_run_root(root: Path, *, flag: bool = False, provenance: bool = False, track_rotation_deg: float = 0.0) -> Path:
    t = np.arange(N) / FPS
    world = np.tile(np.eye(4), (N, 1, 1))
    world[:, :3, :3] = Rotation.from_euler("z", (20 * t)[:, None], degrees=True).as_matrix()
    world[:, :3, 3] = np.stack([0.1 * t, 0.2 + 0.0 * t, 0.7 + 0.05 * t], axis=1)
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
    }))
    if provenance:
        (root / "e2e_config.json").write_text(json.dumps({"workflow": {
            "sequence_id": SEQ, "instruction": "lift the box", "object_prompt": "a box", "task_profile": "box_lift"}}))
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
        git = mock.patch("tools.envpkg.build.git_state", return_value=("f" * 40, False))
        git.start()
        self.addCleanup(git.stop)

    def build(self, **run_kwargs) -> Path:
        run = make_run_root(self.tmp / "run", **run_kwargs)
        build_kwargs = {"allow_missing_provenance": not run_kwargs.get("provenance", False)}
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
        self.assertEqual(manifest["qa"], {"verdict": "pass", "reasons": []})
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
        with self.assertRaisesRegex(ContractError, "physics QA flagged"):
            build_package(run, self.tmp / "pkg", allow_missing_provenance=True)
        self.assertFalse((self.tmp / "pkg").exists())
        package = build_package(run, self.tmp / "pkg", allow_missing_provenance=True, allow_flagged=True)
        self.assertTrue(validate_package(package).ok)
        self.assertRejected(package, "delivery requires 'pass'", strict=True)

    def test_build_refuses_tracks_in_a_rotated_frame(self) -> None:
        run = make_run_root(self.tmp / "run", track_rotation_deg=5.0)
        with self.assertRaisesRegex(ContractError, "constant translation"):
            build_package(run, self.tmp / "pkg", allow_missing_provenance=True)

    def test_tampered_extra_and_missing_files(self) -> None:
        package = self.build()
        with open(package / "core/object/box.obj", "a") as stream:
            stream.write("# edited\n")
        self.assertRejected(package, "box.obj: ")
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
