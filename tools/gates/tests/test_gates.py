"""tools.gates: limit evaluation, reconstruction QA and triage on synthetic inputs.

Run: python3 -m unittest tools.gates.tests.test_gates
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from tools.gates import recon_qa, triage
from tools.gates.common import REJECTED_EXIT, evaluate, verdict_of

N = 30
INTR = np.array([300.0, 300.0, 160.0, 120.0], np.float32)
SHAPE = (240, 320)
CUBE_V = np.array([[x, y, z] for x in (-0.05, 0.05) for y in (-0.05, 0.05) for z in (-0.05, 0.05)])
CUBE_F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                   [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])


def _pose(t: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = t
    return pose


def make_run(root: Path) -> Path:
    """A static camera looking at a cube 0.6 m away that slides slowly; wrists on its sides;
    masks rendered from the same geometry (so a correct bundle agrees with them exactly)."""
    bundle = root / "reconstruction/result_slam_gravity_aligned"
    bundle.mkdir(parents=True)
    obj_cam = np.stack([_pose([0.002 * i, 0.0, 0.6]) for i in range(N)])
    left_cam = np.stack([_pose([-0.06 + 0.002 * i, 0.0, 0.6]) for i in range(N)])
    right_cam = np.stack([_pose([0.06 + 0.002 * i, 0.0, 0.6]) for i in range(N)])
    cam = np.tile(np.eye(4), (N, 1, 1))
    arrays = {
        "camera_to_world_transform": cam, "camera_is_valid": np.ones(N, bool), "camera_intrinsics": INTR,
        "object_to_camera_transform": obj_cam, "object_to_world_transform": cam @ obj_cam, "object_is_valid": np.ones(N, bool),
    }
    for side, pose in (("left", left_cam), ("right", right_cam)):
        arrays[f"hand_{side}_wrist_to_camera_transform"] = pose
        arrays[f"hand_{side}_wrist_to_world_transform"] = cam @ pose
        arrays[f"hand_{side}_is_valid"] = np.ones(N, bool)
    np.savez(bundle / "result.npz", **{k: v.astype(np.float32) if v.dtype == float else v for k, v in arrays.items()})
    (bundle / "mesh.obj").write_text(
        "".join(f"v {x} {y} {z}\n" for x, y, z in CUBE_V) + "".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in CUBE_F))
    (bundle / "manifest.json").write_text(json.dumps({"n_frames": N, "sources": {
        "object_track_id": 1, "left_hand_dir": "/x/hamer/3", "right_hand_dir": "/x/hamer/2"}}))
    scale = recon_qa.MASK_SCALE
    try:
        recon_qa.MASK_SCALE = 1.0
        for i in range(N):
            for track, pose in (("1", obj_cam[i]), ("3", left_cam[i]), ("2", right_cam[i])):
                (root / f"reconstruction/masks/{track}").mkdir(parents=True, exist_ok=True)
                if track == "1":
                    mask = recon_qa._silhouette(CUBE_V, CUBE_F, pose, INTR, SHAPE)
                else:  # a 30 px blob around the wrist's projection
                    u = INTR[0] * pose[0, 3] / pose[2, 3] + INTR[2]
                    mask = np.zeros(SHAPE, bool)
                    mask[105:135, int(u) - 15:int(u) + 15] = True
                cv2.imwrite(str(root / f"reconstruction/masks/{track}/{i:06d}.png"), mask.astype(np.uint8) * 255)
    finally:
        recon_qa.MASK_SCALE = scale
    return root


def edit_npz(run: Path, edit) -> None:
    path = run / "reconstruction/result_slam_gravity_aligned/result.npz"
    arrays = {k: np.array(v) for k, v in np.load(path).items()}
    edit(arrays)
    np.savez(path, **arrays)


class EvaluateTest(unittest.TestCase):
    def test_limits(self) -> None:
        limits = {"a": {"min": 0.5, "why": "A low"}, "b": {"max": 1.0, "why": "B high"}, "_note": "x"}
        with mock.patch("tools.gates.common.load_limits", return_value=limits):
            checks, reasons = evaluate({"a": 0.4, "b": 0.9}, "s")
            self.assertEqual(checks["a"]["ok"], False)
            self.assertEqual(len(reasons), 1)
            self.assertIn("needs >= 0.5", reasons[0])
            checks, reasons = evaluate({"a": None, "b": 2.0}, "s")
            self.assertIsNone(checks["a"]["ok"])  # not measurable: skipped, not passed or failed silently
            self.assertIn("B high", reasons[0])
            with self.assertRaises(KeyError):
                evaluate({"a": 1.0}, "s")

    def test_every_limit_names_a_real_metric(self) -> None:
        limits = json.loads((Path(triage.__file__).parent / "limits.json").read_text())
        for section in ("recon_qa/v1", "triage/v1"):
            for name, limit in limits[section].items():
                if not name.startswith("_"):
                    self.assertTrue({"min", "max"} & set(limit) and limit.get("why"), f"{section} {name}")


class ReconQATest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.run_root = make_run(self.tmp / "run")
        self.bundle = self.run_root / "reconstruction/result_slam_gravity_aligned"
        self.masks = self.run_root / "reconstruction/masks"

    def report(self, masks: bool = True) -> dict:
        return recon_qa.run(self.bundle, fps=30.0, masks=self.masks if masks else None, out=None)

    def test_consistent_bundle_passes(self) -> None:
        report = self.report()
        self.assertEqual(report["verdict"], "pass", report["reasons"])
        self.assertGreater(report["metrics"]["object_iou_median"], 0.9)
        self.assertEqual(report["metrics"]["wrist_on_hand_mask_fraction"], 1.0)
        without = self.report(masks=False)
        self.assertEqual(without["verdict"], "pass")
        self.assertIn("object_iou_median", without["skipped"])

    def test_object_off_its_mask_is_rejected(self) -> None:
        def shift(a):
            a["object_to_camera_transform"][:, 0, 3] += 0.04
            a["object_to_world_transform"] = a["camera_to_world_transform"] @ a["object_to_camera_transform"]

        edit_npz(self.run_root, shift)
        self.assertTrue(any(r.startswith("object_iou") for r in self.report()["reasons"]))

    def test_camera_jump_and_swapped_hands_are_rejected(self) -> None:
        def jump(a):
            for key in ("camera_to_world_transform", "object_to_world_transform",
                        "hand_left_wrist_to_world_transform", "hand_right_wrist_to_world_transform"):
                a[key][N // 2:, 0, 3] += 0.3

        edit_npz(self.run_root, jump)
        self.assertTrue(any(r.startswith("camera_speed_max_mps") for r in self.report()["reasons"]))

    def test_cli_exit_codes_and_report(self) -> None:
        cfg = self.run_root / "e2e_config.json"
        cfg.write_text(json.dumps({"inputs": {}}))
        self.assertEqual(recon_qa.main(["--run-root", str(self.run_root)]), 2)  # no video, no --fps: cannot evaluate
        self.assertEqual(recon_qa.main(["--run-root", str(self.run_root), "--fps", "30"]), 0)
        out = self.run_root / "qa/reconstruction.json"
        self.assertEqual(verdict_of(out), (True, []))
        edit_npz(self.run_root, lambda a: a["object_is_valid"].__setitem__(slice(0, 20), False))
        self.assertEqual(recon_qa.main(["--run-root", str(self.run_root), "--fps", "30"]), REJECTED_EXIT)
        self.assertEqual(recon_qa.main(["--run-root", str(self.run_root), "--fps", "30", "--exit-zero"]), 0)
        accepted, reasons = verdict_of(out)
        self.assertFalse(accepted)
        self.assertTrue(reasons[0].startswith("object_valid_fraction"))


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg")
class TriageTest(unittest.TestCase):
    """Hands are mocked (MediaPipe is exercised by tools.gates.calibrate on real clips)."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.model = self.tmp / "hand.task"
        self.model.write_bytes(b"x")
        detector = mock.MagicMock()
        patches = [
            mock.patch.object(triage, "_hand_detector", return_value=detector),
            mock.patch.object(triage, "_hands", return_value=[0.3, 0.25]),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def video(self, name: str, vf: str | None = None, seconds: int = 4) -> Path:
        out = self.tmp / f"{name}.mp4"
        args = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size=854x480:rate=30:duration={seconds}"]
        if vf:
            args += ["-vf", vf]
        subprocess.run([*args, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True)
        return out

    def test_accepts_a_clean_clip_and_rejects_defects(self) -> None:
        good = triage.run(self.video("good"), hand_model=self.model, out=None)
        self.assertEqual(good["verdict"], "accept", good["reasons"])
        frozen = triage.run(self.video("frozen", "fps=3,fps=30"), hand_model=self.model, out=None)
        self.assertTrue(any(r.startswith("frozen_fraction") for r in frozen["reasons"]))
        blurred = triage.run(self.video("blurred", "gblur=sigma=12"), hand_model=self.model, out=None)
        self.assertTrue(any(r.startswith("sharp_frame_fraction") for r in blurred["reasons"]))
        short = triage.run(self.video("short", seconds=1), hand_model=self.model, out=None)
        self.assertTrue(any(r.startswith("format: duration") for r in short["reasons"]))

    def test_no_hands_and_cli(self) -> None:
        video = self.video("good")
        with mock.patch.object(triage, "_hands", return_value=[]):
            out = self.tmp / "triage.json"
            self.assertEqual(triage.main(["--video", str(video), "--out", str(out), "--hand-model", str(self.model)]), REJECTED_EXIT)
            accepted, reasons = verdict_of(out)
            self.assertFalse(accepted)
            self.assertTrue(any(r.startswith("hand_frame_fraction") for r in reasons))
        self.assertEqual(triage.main(["--video", str(self.tmp / "missing.mp4")]), 2)


class PipelineExitTest(unittest.TestCase):
    def test_gate_exit_codes(self) -> None:
        from tools.e2e.cli import _gate_exit
        from tools.modal_v2d import batch as B

        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        report = tmp / "qa/triage.json"
        report.parent.mkdir()
        args = argparse.Namespace(dry_run=False)
        report.write_text(json.dumps({"verdict": "accept", "reasons": []}))
        self.assertEqual(_gate_exit(args, report, "triage"), 0)
        report.write_text(json.dumps({"schema": "triage/v1", "verdict": "reject", "reasons": ["cut_count = 1"]}))
        self.assertEqual(_gate_exit(args, report, "triage"), REJECTED_EXIT)
        self.assertEqual(B.GATE_REJECTED, REJECTED_EXIT)
        files = {f"{B.RUNS}/c1/qa/triage.json": report.read_text()}
        self.assertEqual(B.gate_reasons("c1", lambda p: files[p] if p in files else (_ for _ in ()).throw(OSError())),
                         ["triage/v1: cut_count = 1"])


if __name__ == "__main__":
    unittest.main()
