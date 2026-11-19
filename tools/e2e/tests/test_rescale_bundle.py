"""tools.e2e.rescale_bundle: uniform metric rescale anchored on the hand scale."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]


def make_bundle(root: Path, hand_scale: float) -> Path:
    bundle = root / "bundle"
    bundle.mkdir()
    T = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
    T[:, :3, 3] = [0.1, 0.2, 0.6]
    np.savez(
        bundle / "result.npz",
        camera_to_world_transform=T, object_to_world_transform=T, object_to_camera_transform=T,
        hand_left_wrist_to_world_transform=T, hand_right_wrist_to_world_transform=T,
        hand_left_wrist_to_camera_transform=T, hand_right_wrist_to_camera_transform=T,
        hand_left_wrist_trans_in_camera=T[:, :3, 3], hand_right_wrist_trans_in_camera=T[:, :3, 3],
        hand_left_scale=np.full(3, hand_scale, np.float32), hand_right_scale=np.full(3, hand_scale, np.float32),
        camera_intrinsics=np.array([600, 600, 640, 360], np.float32), object_scale=np.float32(1.0),
    )
    (bundle / "mesh.obj").write_text("v 0.1 0.2 0.3\nv -0.1 0 0\nf 1 2 1\n")
    (bundle / "manifest.json").write_text("{}")
    return bundle


def run(*args):
    return subprocess.run([sys.executable, "-m", "tools.e2e.rescale_bundle", *map(str, args)], cwd=REPO, capture_output=True, text=True)


class RescaleBundleTests(unittest.TestCase):
    def test_oversized_world_is_rescaled_by_inverse_hand_scale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = make_bundle(Path(tmp), 1.5)
            result = run(bundle)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = np.load(bundle / "result.npz")
            np.testing.assert_allclose(data["camera_to_world_transform"][0, :3, 3], np.array([0.1, 0.2, 0.6]) / 1.5, rtol=1e-6)
            np.testing.assert_allclose(data["hand_left_scale"], 1.0, rtol=1e-6)
            np.testing.assert_allclose(data["camera_to_world_transform"][0, :3, :3], np.eye(3))  # rotations untouched
            first_vertex = (bundle / "mesh.obj").read_text().splitlines()[0].split()[1:]
            np.testing.assert_allclose([float(v) for v in first_vertex], np.array([0.1, 0.2, 0.3]) / 1.5, rtol=1e-6)
            self.assertAlmostEqual(json.loads((bundle / "manifest.json").read_text())["metric_rescale"]["factor"], 1 / 1.5)
            self.assertTrue((bundle.parent / "bundle.unscaled" / "result.npz").exists())
            self.assertNotEqual(run(bundle).returncode, 0)  # never rescales twice

    def test_metric_bundle_is_left_alone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = make_bundle(Path(tmp), 1.08)  # a large hand, not an oversized world
            result = run(bundle)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("nothing to do", result.stdout)
            self.assertFalse((bundle.parent / "bundle.unscaled").exists())


if __name__ == "__main__":
    unittest.main()
