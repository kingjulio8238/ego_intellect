"""Local tests for the Modal adapter (no Modal account, network, GPU or docker needed)."""

from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from tools.modal_v2d import checks, config, docker_shim, mounts, sync  # noqa: E402

SHIM = REPO / "tools" / "modal_v2d" / "docker_shim.py"


class FakeRPC:
    """Unix-socket server standing in for the role container's parent process."""

    def __init__(self, path: str, exit_code: int = 0):
        self.path = path
        self.specs: list[dict] = []
        self.exit_code = exit_code
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(path)
        self.server.listen()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while True:
            try:
                connection, _ = self.server.accept()
            except OSError:
                return
            with connection:
                request = json.loads(connection.makefile("r").readline())
                self.specs.append(request["spec"])
                connection.sendall((json.dumps({"log": "remote says hi\n"}) + "\n").encode())
                connection.sendall((json.dumps({"exit": self.exit_code}) + "\n").encode())

    def close(self) -> None:
        self.server.close()


class ShimTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(os.path.realpath(self.tmp.name))
        self.env = {
            **os.environ,
            "V2D_SHIM_STATE": str(self.root / "shim"),
            "V2D_MODAL_RPC": str(self.root / "rpc.sock"),
            "V2D_MODAL_LOCAL_IMAGE": "",
        }

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def shim(self, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SHIM), *args], capture_output=True, text=True, env=env or self.env
        )


class DockerShimTests(ShimTestCase):
    def test_build_and_image_inspect_only_accept_defined_images(self) -> None:
        self.assertEqual(self.shim("build", "-t", "v2d_moge", "-f", "x", ".").returncode, 0)
        self.assertEqual(self.shim("build", "-t", "robotic-grounding:latest", "-f", "x", ".").returncode, 0)
        self.assertNotEqual(self.shim("build", "-t", "v2d_unknown", "-f", "x", ".").returncode, 0)
        self.assertEqual(self.shim("image", "inspect", "v2d_task_library_loader:latest").returncode, 0)
        self.assertEqual(self.shim("image", "inspect", "nope:latest").returncode, 1)

    def test_unsupported_commands_fail_loudly(self) -> None:
        result = self.shim("push", "x")
        self.assertEqual(result.returncode, 2)
        self.assertIn("not supported", result.stderr)
        self.assertNotEqual(self.shim("run", "--mount", "type=bind", "v2d_moge", "true").returncode, 0)

    def test_run_in_container_command_is_forwarded_as_spec(self) -> None:
        rpc = FakeRPC(self.env["V2D_MODAL_RPC"])
        data = self.root / "out"
        data.mkdir()
        try:
            result = self.shim(
                "run", "--rm", "--runtime=nvidia", "--gpus", "all", "--user", "501:20",
                "-e", "HOME=/tmp", "-e", "HF_TOKEN", "-v", f"{data}:/data/depth_folder",
                "v2d_moge", "python", "-m", "v2d.moge.lib.video_to_depth", "--depth_folder", "/data/depth_folder/d",
                env={**self.env, "HF_TOKEN": "secret"},
            )
        finally:
            rpc.close()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("remote says hi", result.stdout)
        spec = rpc.specs[0]
        self.assertEqual(spec["image"], "v2d_moge")
        self.assertTrue(spec["gpu"])
        self.assertEqual(spec["env"], {"HOME": "/tmp", "HF_TOKEN": "secret"})
        self.assertEqual(spec["mounts"], [{"src": str(data), "dst": "/data/depth_folder", "mode": "rw"}])
        self.assertEqual(spec["argv"][:3], ["python", "-m", "v2d.moge.lib.video_to_depth"])

    def test_remote_exit_code_propagates(self) -> None:
        rpc = FakeRPC(self.env["V2D_MODAL_RPC"], exit_code=3)
        try:
            result = self.shim("run", "--rm", "v2d_sam2", "false")
        finally:
            rpc.close()
        self.assertEqual(result.returncode, 3)

    def test_workflow_run_sh_e2e_lifecycle_through_shim(self) -> None:
        """Drive the real robotic_grounding/workflow/run.sh e2e-run against the shim."""
        rpc = FakeRPC(self.env["V2D_MODAL_RPC"])
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        (bin_dir / "docker").write_text(f'#!/bin/sh\nexec "{sys.executable}" "{SHIM}" "$@"\n')
        (bin_dir / "docker").chmod(0o755)
        run_data = self.root / "run"
        mano = self.root / "mano"
        run_data.mkdir()
        mano.mkdir()
        env = {**self.env, "PATH": f"{bin_dir}:{os.environ['PATH']}", "HOME": str(self.root)}
        command = [
            "bash", str(REPO / "robotic_grounding/workflow/run.sh"), "e2e-run", "latest", "0",
            "--run-data", str(run_data), "--mano-dir", str(mano),
            "--workdir", "/workspace/video_to_data/robotic_grounding", "--recreate-on-mount-change",
            "--", "python", "-c", "print('ok')",
        ]
        try:
            first = subprocess.run(command, capture_output=True, text=True, env=env, cwd=REPO / "robotic_grounding")
            second = subprocess.run(command, capture_output=True, text=True, env=env, cwd=REPO / "robotic_grounding")
            inspect = self.shim(
                "inspect", "-f", '{{range .Mounts}}{{printf "%s|%s\\n" .Source .Destination}}{{end}}',
                "robotic-grounding-latest-gpu0",
            )
        finally:
            rpc.close()
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assertIn("Creating and starting new container", first.stdout)
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertIn("Container is already running", second.stdout)  # mounts matched: reused
        self.assertIn(f"{REPO}|/workspace/video_to_data", inspect.stdout)
        self.assertIn(f"{run_data}|/workspace/e2e", inspect.stdout)
        spec = rpc.specs[-1]
        self.assertEqual(spec["image"], "robotic-grounding")
        self.assertEqual(spec["argv"], ["python", "-c", "print('ok')"])
        self.assertEqual(spec["workdir"], "/workspace/video_to_data/robotic_grounding")
        self.assertEqual(spec["env"]["ACCEPT_EULA"], "Y")

    def test_liveness_probe_exec_needs_no_remote_call(self) -> None:
        rpc = FakeRPC(self.env["V2D_MODAL_RPC"])
        try:
            started = self.shim("run", "-d", "--name", "rg", "robotic-grounding:latest", "sleep", "infinity")
            probe = self.shim("exec", "rg", "true")
        finally:
            rpc.close()
        self.assertEqual(started.returncode, 0, started.stderr)
        self.assertEqual(probe.returncode, 0, probe.stderr)
        self.assertEqual(rpc.specs, [])  # no GPU container rented to run `true`

    def test_local_image_runs_in_process_with_symlinked_mounts(self) -> None:
        source = self.root / "src"
        source.mkdir()
        (source / "hello.txt").write_text("hi")
        dst = self.root / "mnt" / "e2e"
        env = {**self.env, "V2D_MODAL_LOCAL_IMAGE": "robotic-grounding"}
        result = self.shim(
            "run", "--rm", "-v", f"{source}:{dst}", "robotic-grounding:latest", "cat", f"{dst}/hello.txt", env=env
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "hi")
        self.assertFalse(dst.exists(), "foreground mounts are reverted")

    def test_inspect_templates(self) -> None:
        record = {"mounts": [{"src": "/a", "dst": "/b"}], "network": "host"}
        self.assertEqual(docker_shim.render_inspect(record, "{{range .Mounts}}{{println .Source}}{{end}}"), "/a\n")
        self.assertEqual(docker_shim.render_inspect(record, "{{.HostConfig.NetworkMode}}"), "host\n")
        with self.assertRaises(docker_shim.ShimError):
            docker_shim.render_inspect(record, "{{.Id}}")


class MountTests(unittest.TestCase):
    def test_apply_shadows_and_reverts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "workspace"
            image_dir.mkdir()
            (image_dir / "baked").write_text("image")
            source = root / "modules"
            source.mkdir()
            undo = mounts.apply([{"src": str(source), "dst": str(image_dir)}])
            self.assertTrue(image_dir.is_symlink())
            mounts.revert(undo)
            self.assertFalse(image_dir.is_symlink())
            self.assertEqual((image_dir / "baked").read_text(), "image")

    def test_system_destinations_are_skipped(self) -> None:
        self.assertEqual(mounts.apply([{"src": "/nonexistent", "dst": "/etc/passwd"}]), [])


class SyncTests(unittest.TestCase):
    def test_push_checkout_restores_mtimes_and_deletions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(os.path.realpath(tmp))
            mirror_root = str(root / "vol" / "fs")
            index = str(root / "vol" / "index.json")
            local = root / "e2e"
            (local / "runs" / "a").mkdir(parents=True)
            keep = local / "runs" / "a" / "keep.txt"
            drop = local / "runs" / "a" / "drop.txt"
            keep.write_text("keep")
            drop.write_text("drop")
            os.utime(keep, ns=(1_000_000_000, 1_000_000_000))
            (local / "runs" / "empty").mkdir()

            first = sync.Mirror(mirror_root, index)
            stats = first.push(str(local))
            self.assertEqual(stats.copied, 2)
            self.assertEqual(first.push(str(local)).copied, 0)  # unchanged -> skipped
            drop.unlink()
            self.assertEqual(first.push(str(local)).deleted, 1)
            first.save()

            # Simulate a volume that does not preserve mtimes.
            mirrored_keep = sync.mirror_path(mirror_root, str(keep))
            os.utime(mirrored_keep, ns=(5, 5))
            for path in local.rglob("*"):
                if path.is_file():
                    path.unlink()
            second = sync.Mirror(mirror_root, index)
            second.load()
            second.checkout(str(local))
            self.assertEqual(keep.read_text(), "keep")
            self.assertEqual(keep.stat().st_mtime_ns, 1_000_000_000)
            self.assertFalse(drop.exists())
            self.assertTrue((local / "runs" / "empty").is_dir())

    def test_worker_diff_and_pull(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(os.path.realpath(tmp))
            mirror_root = str(root / "fs")
            local = root / "work"
            local.mkdir()
            (local / "old.txt").write_text("old")
            mirror = sync.Mirror(mirror_root, str(root / "index.json"))
            mirror.push(str(local))
            remote_root = sync.mirror_path(mirror_root, str(local))
            before = sync.snapshot([remote_root])
            Path(remote_root, "new").mkdir()
            Path(remote_root, "new", "out.txt").write_text("out")
            Path(remote_root, "old.txt").unlink()
            changed, deleted, dirs = sync.diff(before, sync.snapshot([remote_root]))
            mirror.pull_changes(changed, deleted, dirs)
            self.assertEqual((local / "new" / "out.txt").read_text(), "out")
            self.assertFalse((local / "old.txt").exists())


class FakeVolume:
    def __init__(self) -> None:
        self.commits = 0
        self.reloads = 0

    def commit(self) -> None:
        self.commits += 1

    def reload(self) -> None:
        self.reloads += 1


class ParentWorkerRoundTripTests(unittest.TestCase):
    """runtime.StateSync.run_remote -> worker.execute, with volumes faked as local dirs."""

    def test_mirrored_mount_round_trip(self) -> None:
        from unittest import mock

        from tools.modal_v2d import runtime, worker

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(os.path.realpath(tmp))
            vol, weights, repo = root / "vol", root / "weights", root / "repo"
            for path in (vol, weights / "moge", repo):
                path.mkdir(parents=True)
            (weights / "moge" / "model.bin").write_text("w")
            run_dir = root / "e2e" / "runs" / "a" / "reconstruction"
            run_dir.mkdir(parents=True)
            (run_dir / "input.txt").write_text("frames")
            (run_dir / "stale.txt").write_text("stale")
            patches = {
                "STATE_MOUNT": str(vol),
                "WEIGHTS_MOUNT": str(weights),
                "MIRROR_ROOT": str(vol / "fs"),
                "INDEX_PATH": str(vol / "index.json"),
                "CALL_LOG_ROOT": str(vol / "calls"),
                "REPO_ROOT": str(repo),
            }
            state_volume, weights_volume = FakeVolume(), FakeVolume()
            dst = root / "container" / "data" / "out"
            weights_dst = root / "container" / "data" / "weights"

            def remote(spec: dict) -> dict:
                return worker.execute(spec, state_volume=state_volume, weights_volume=weights_volume)

            fake_function = mock.Mock()
            fake_function.remote.side_effect = remote
            emitted: list[str] = []
            with mock.patch.multiple(config, **patches):
                state = runtime.StateSync(state_volume, weights_volume)
                code = state.run_remote(
                    {
                        "image": "v2d_moge",
                        "argv": ["sh", "-c", f"cat input.txt > result.txt && rm stale.txt && ls {weights_dst}"],
                        "env": {},
                        "workdir": str(dst),
                        "mounts": [
                            {"src": str(run_dir), "dst": str(dst)},
                            {"src": str(weights / "moge"), "dst": str(weights_dst)},
                            {"src": str(root / "nope"), "dst": "/etc/passwd"},
                        ],
                        "gpu": True,
                    },
                    {"v2d_moge": fake_function},
                    emitted.append,
                )
            self.assertEqual(code, 0, "".join(emitted))
            self.assertEqual((run_dir / "result.txt").read_text(), "frames")  # pulled back
            self.assertFalse((run_dir / "stale.txt").exists())  # deletion pulled back
            self.assertFalse(dst.exists())  # worker reverted its mount symlinks
            spec = fake_function.remote.call_args.args[0]
            by_dst = {m["dst"]: m for m in spec["mounts"]}
            self.assertTrue(by_dst[str(dst)]["mirrored"])
            self.assertEqual(by_dst[str(dst)]["src"], str(vol / "fs") + str(run_dir))
            self.assertFalse(by_dst[str(weights_dst)]["mirrored"])
            self.assertNotIn("/etc/passwd", by_dst)
            self.assertIn("model.bin", "".join(emitted))  # remote log streamed back
            self.assertGreaterEqual(state_volume.commits, 2)
            # Outputs and log came back inline: only the worker reloaded the volume.
            self.assertEqual(state_volume.reloads, 1)

    def test_large_outputs_and_unchanged_inputs(self) -> None:
        """Outputs above the payload limit are read from the volume (after a reload); a call
        whose inputs are already mirrored does not commit; nested mounts are walked once."""
        from unittest import mock

        from tools.modal_v2d import runtime, worker

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(os.path.realpath(tmp))
            vol, weights, repo = root / "vol", root / "weights", root / "repo"
            for path in (vol, weights, repo):
                path.mkdir(parents=True)
            run_dir = root / "e2e" / "runs" / "a" / "reconstruction"
            (run_dir / "masks").mkdir(parents=True)
            (run_dir / "input.txt").write_text("frames")
            (run_dir / "masks" / "m.png").write_text("mask")
            patches = {
                "STATE_MOUNT": str(vol),
                "WEIGHTS_MOUNT": str(weights),
                "MIRROR_ROOT": str(vol / "fs"),
                "INDEX_PATH": str(vol / "index.json"),
                "CALL_LOG_ROOT": str(vol / "calls"),
                "REPO_ROOT": str(repo),
                "PAYLOAD_MAX_BYTES": 0,
            }
            state_volume, weights_volume = FakeVolume(), FakeVolume()
            dst, masks_dst = root / "c" / "data" / "out", root / "c" / "data" / "masks"

            def remote(spec: dict) -> dict:
                return worker.execute(spec, state_volume=state_volume, weights_volume=weights_volume)

            fake_function = mock.Mock()
            fake_function.remote.side_effect = remote
            spec = {
                "image": "v2d_moge",
                "argv": ["sh", "-c", "cat input.txt > result.txt"],
                "env": {},
                "workdir": str(dst),
                "mounts": [{"src": str(run_dir), "dst": str(dst)}, {"src": str(run_dir / "masks"), "dst": str(masks_dst)}],
                "gpu": True,
            }
            emitted: list[str] = []
            with mock.patch.multiple(config, **patches):
                state = runtime.StateSync(state_volume, weights_volume)
                with mock.patch.object(state.mirror, "push", wraps=state.mirror.push) as push:
                    self.assertEqual(state.run_remote(spec, {"v2d_moge": fake_function}, emitted.append), 0)
                    self.assertEqual([c.args[0] for c in push.call_args_list], [str(run_dir)])  # masks/ covered
                self.assertEqual((run_dir / "result.txt").read_text(), "frames")
                self.assertEqual(state_volume.reloads, 2)  # worker + parent (outputs not inline)
                commits = state_volume.commits
                spec2 = {**spec, "argv": ["sh", "-c", "true"]}
                self.assertEqual(state.run_remote(spec2, {"v2d_moge": fake_function}, emitted.append), 0)
                # Second call: nothing to push, so the parent did not commit (the worker did).
                self.assertEqual(state_volume.commits, commits + 1)

    def test_weight_downloads_route_to_cpu_twin(self) -> None:
        from unittest import mock

        from tools.modal_v2d import runtime

        gpu_function, cpu_function = mock.Mock(), mock.Mock()
        result = {"returncode": 0, "changed": [], "deleted": [], "dirs": [], "log": "/nonexistent", "elapsed_s": 1.0}
        gpu_function.remote.return_value = cpu_function.remote.return_value = result
        with tempfile.TemporaryDirectory() as tmp, mock.patch.multiple(
            config, STATE_MOUNT=tmp, MIRROR_ROOT=f"{tmp}/fs", INDEX_PATH=f"{tmp}/index.json"
        ):
            state = runtime.StateSync(FakeVolume(), FakeVolume())
            state.cpu_functions = {"v2d_sam2": cpu_function}
            base = {"image": "v2d_sam2", "env": {}, "workdir": "", "mounts": [], "gpu": True}
            state.run_remote({**base, "argv": ["python", "-m", "v2d.sam2.lib.download_weights"]}, {"v2d_sam2": gpu_function}, lambda _: None)
            state.run_remote({**base, "argv": ["python", "-m", "v2d.sam2.lib.track"]}, {"v2d_sam2": gpu_function}, lambda _: None)
        self.assertEqual(cpu_function.remote.call_count, 1)
        self.assertEqual(gpu_function.remote.call_count, 1)


class ReviewRegressionTests(unittest.TestCase):
    def test_foundation_pose_dev_command_with_anonymous_volumes(self) -> None:
        """--dev FoundationPose adds anonymous `-v /workspace/...` volumes; they must parse,
        and the --dev modules mount must not be forwarded (the image is built from it)."""
        from unittest import mock

        from tools.modal_v2d import runtime

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(os.path.realpath(tmp))
            repo = root / "repo"
            (repo / "reconstruction" / "modules").mkdir(parents=True)
            out = root / "e2e" / "runs" / "a" / "reconstruction"
            out.mkdir(parents=True)
            volumes = [
                docker_shim.parse_volume(f"{out}:/data/poses_folder"),
                docker_shim.parse_volume(f"{repo / 'reconstruction' / 'modules'}:/workspace"),
                docker_shim.parse_volume("/workspace/v2d_foundation_pose/lib/FoundationPose/mycpp/build"),
            ]
            self.assertEqual(volumes[2]["mode"], "anonymous")
            captured = {}

            def remote(spec: dict) -> dict:
                captured.update(spec)
                return {"returncode": 0, "changed": [], "deleted": [], "dirs": [], "log": str(root / "none.log")}

            function = mock.Mock()
            function.remote.side_effect = remote
            patches = {
                "STATE_MOUNT": str(root / "vol"), "WEIGHTS_MOUNT": str(root / "weights"),
                "MIRROR_ROOT": str(root / "vol" / "fs"), "INDEX_PATH": str(root / "vol" / "i.json"),
                "REPO_ROOT": str(repo),
            }
            with mock.patch.multiple(config, **patches):
                state = runtime.StateSync(FakeVolume(), FakeVolume())
                code = state.run_remote(
                    {"image": "v2d_foundation_pose", "argv": ["true"], "env": {}, "workdir": None,
                     "mounts": volumes, "gpu": True},
                    {"v2d_foundation_pose": function},
                    lambda text: None,
                )
            self.assertEqual(code, 0)
            self.assertEqual([m["dst"] for m in captured["mounts"]], ["/data/poses_folder"])

    def test_run_root_validation_forms(self) -> None:
        from unittest import mock

        from tools.modal_v2d import runtime

        with mock.patch.multiple(config, REPO_ROOT="/workspace/video_to_data"):
            self.assertEqual(
                runtime.runs_touched(["init", "--run-root=/e2e/runs/a", "--sequence-id", "x"]), {"/e2e/runs/a"}
            )
            self.assertEqual(
                runtime.runs_touched(["--config", "/e2e/runs/b/e2e_config.json", "status"]), {"/e2e/runs/b"}
            )
            self.assertEqual(runtime.runs_touched(["use", "/e2e/runs/c/e2e_config.json"]), {"/e2e/runs/c"})
            for bad in (["init", "--run-root=/tmp/x"], ["init", "--run-root", "runs/a"], ["init", "--run-root"]):
                with self.assertRaises(ValueError, msg=bad):
                    runtime.runs_touched(bad)

    def test_failed_checkout_raises_instead_of_deleting_later(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(os.path.realpath(tmp))
            mirror = sync.Mirror(str(root / "fs"), str(root / "index.json"))
            local = root / "e2e"
            local.mkdir()
            (local / "f.txt").write_text("x")
            mirror.push(str(local))
            (local / "f.txt").unlink()
            local.chmod(0o500)  # read-only: the copy back fails
            try:
                stats = mirror.checkout(str(local))
                with self.assertRaises(RuntimeError):
                    stats.raise_errors("checkout")
            finally:
                local.chmod(0o700)

    def test_container_env_drops_role_variables_and_restores_image_pythonpath(self) -> None:
        from unittest import mock

        from tools.modal_v2d import containerenv

        with tempfile.TemporaryDirectory() as tmp:
            recorded = Path(tmp) / "pythonpath"
            recorded.write_text("/opt/hamer")
            base = {"PATH": "/usr/bin", "PYTHONPATH": "/pkg:/root", "MODAL_TASK_ID": "t", "V2D_MODAL_RPC": "s",
                    "PYTHON": "/usr/local/bin/python3", "HOME": "/root"}
            with mock.patch.object(config, "IMAGE_PYTHONPATH_FILE", str(recorded)):
                env = containerenv.container_env({"HOME": "/tmp"}, "robotic-grounding", base=base)
            self.assertEqual(env["PYTHONPATH"], "/opt/hamer")
            self.assertEqual(env["HOME"], "/tmp")
            self.assertTrue(env["PATH"].startswith(config.ISAAC_BIN + ":"))
            for key in ("MODAL_TASK_ID", "V2D_MODAL_RPC", "PYTHON"):
                self.assertNotIn(key, env)


class ConfigTests(unittest.TestCase):
    def test_roles(self) -> None:
        self.assertEqual(config.role_for(["--dry-run", "reconstruct", "--video", "x"]), "control")
        self.assertEqual(config.role_for(["--config", "/e2e/runs/a/e2e_config.json", "train-expert"]), "rg")
        self.assertEqual(config.role_for(["doctor"]), "rg")
        self.assertEqual(config.role_for(["--dry-run", "train-expert"]), "control")
        self.assertEqual(config.role_for(["reconstruct", "--dry-run"]), "control")
        with self.assertRaises(ValueError):
            config.role_for(["bogus"])
        e2e_commands = {
            "init", "use", "setup", "doctor", "triage", "reconstruct", "import-reconstruction", "inspect",
            "retarget", "simulate", "solvability", "render", "train-expert", "collect", "finetune", "evaluate",
            "status",
        }
        self.assertEqual(config.CONTROL_COMMANDS | config.RG_COMMANDS, e2e_commands)

    def test_every_image_has_a_dockerfile(self) -> None:
        modules = REPO / "reconstruction" / "modules"
        for image, module in config.MODULE_IMAGES.items():
            dockerfile = modules / module / "docker" / "Dockerfile"
            self.assertTrue(dockerfile.is_file(), dockerfile)
            names = (modules / module / "docker" / "_config.py").read_text()
            self.assertIn(f'IMAGE_NAME = "{image}"', names)


class ImageTests(unittest.TestCase):
    def test_robotic_grounding_dockerfile_rewrite(self) -> None:
        try:
            from tools.modal_v2d import images
        except ImportError as exc:  # modal SDK not installed locally
            self.skipTest(str(exc))
        rendered = images.render_robotic_grounding_dockerfile(
            (REPO / "robotic_grounding/workflow/Dockerfile").read_text()
        )
        self.assertNotIn("/usr/local/bin/python", rendered)
        self.assertEqual(rendered.count(f"{config.ISAAC_BIN}/python"), 3)
        self.assertTrue(rendered.startswith("FROM nvcr.io/nvidia/isaac-lab:2.3.2\n"))
        self.assertEqual(sum(line.startswith("FROM ") for line in rendered.splitlines()), 1)
        self.assertNotIn("\nCMD ", rendered)


class LfsCheckTests(unittest.TestCase):
    def test_pointer_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / ".gitattributes").write_text("*.mp4 filter=lfs diff=lfs merge=lfs -text\n")
            (repo / "ok.mp4").write_bytes(b"\x00\x00real video")
            self.assertEqual(checks.lfs_materialized(repo), 0)
            (repo / "bad.mp4").write_bytes(checks.LFS_POINTER_PREFIX + b"\noid sha256:abc\n")
            self.assertEqual(checks.lfs_materialized(repo), 1)


if __name__ == "__main__":
    unittest.main()


class BakeTests(unittest.TestCase):
    def test_bake_prepares_repo_and_bootstrap_skips_it(self) -> None:
        from unittest import mock

        from tools.modal_v2d import bake, runtime

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(os.path.realpath(tmp))
            upload, repo = root / "upload", root / "repo"
            (upload / "tools").mkdir(parents=True)
            (upload / "tools" / "a.py").write_text("x = 1\n")
            (upload / "tools" / "modal_v2d").mkdir()
            (upload / "tools" / "modal_v2d" / "plumbing.py").write_text("y = 2\n")
            marker = root / "baked"
            with mock.patch.multiple(config, REPO_UPLOAD=str(upload), REPO_ROOT=str(repo)), mock.patch.object(
                runtime, "BAKED_MARKER", str(marker)
            ), mock.patch.object(runtime, "_install_host_packages") as install:
                bake.main("control")
                self.assertTrue(marker.is_file())
                install.assert_called_once_with("control")
                listed = subprocess.run(["git", "ls-files"], cwd=repo, capture_output=True, text=True).stdout
                self.assertEqual(listed.split(), ["tools/a.py"])  # tools/modal_v2d is excluded
                untracked = subprocess.run(
                    ["git", "ls-files", "--others", "--exclude-standard"], cwd=repo, capture_output=True, text=True
                ).stdout
                self.assertNotIn("modal_v2d", untracked)  # nor listed as untracked by tools/e2e
                mtime = (repo / "tools" / "a.py").stat().st_mtime_ns
                with mock.patch.object(runtime, "_copy_repo") as copy, mock.patch.object(runtime, "_install_shim"), mock.patch.object(
                    runtime, "_place_mano"
                ), mock.patch.multiple(config, LOCAL_POINTER_DIR=str(root / "ptr"), RECON_WEIGHTS=str(root / "w")):
                    install.reset_mock()
                    runtime.bootstrap("control")
                    copy.assert_not_called()
                    install.assert_not_called()
                self.assertEqual((repo / "tools" / "a.py").stat().st_mtime_ns, mtime)


class PrefetchTests(unittest.TestCase):
    def test_prefetch_lists_script_calls_and_runs_them_in_parallel(self) -> None:
        from unittest import mock

        from tools.modal_v2d import progress, runtime

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(os.path.realpath(tmp))
            recon = root / "repo" / "reconstruction"
            (recon / "scripts").mkdir(parents=True)
            (recon / "scripts" / "download_ego_reconstruction_weights.sh").write_text(
                'set -e\nPYTHON_BIN="${PYTHON:-python3}"\nEULA="$3$4"\n'
                '"$PYTHON_BIN" -m v2d.alpha.docker.run_download_weights --output_dir data/weights/alpha\n'
                '"$PYTHON_BIN" -m v2d.beta.docker.run_download_weights --output_dir data/weights/beta "$EULA"\n'
            )
            # Fake wrappers: each sleeps 1 s and records when it ran.
            for name in ("alpha", "beta"):
                package = root / "pkgs" / "v2d" / name / "docker"
                package.mkdir(parents=True)
                for level in (root / "pkgs" / "v2d", root / "pkgs" / "v2d" / name, package):
                    (level / "__init__.py").touch()
                (package / "run_download_weights.py").write_text(
                    "import sys, time\n"
                    f"open({str(root / 'ran.log')!r}, 'a').write(' '.join(sys.argv[1:]) + '\\n')\n"
                    "time.sleep(1)\n"
                )
            env = {**os.environ, "PYTHON": sys.executable, "PYTHONPATH": str(root / "pkgs")}
            tracker = progress.Tracker(
                invocation="p", role="control", argv=["setup"], timeout_s=60, gpu=None, cpu=4,
                memory_gib=16, journal_dir=str(root / "journal"), out=io.StringIO(), heartbeat_s=3600,
            )
            with mock.patch.multiple(config, REPO_ROOT=str(root / "repo")):
                started = time.time()
                runtime.prefetch_weights(["setup", "--accept-nvidia-model-eula"], env, tracker, state=None)
                elapsed = time.time() - started
            ran = (root / "ran.log").read_text().splitlines()
            self.assertEqual(sorted(ran), [
                "--output_dir data/weights/alpha",
                "--output_dir data/weights/beta --accept-nvidia-model-eula",
            ])
            self.assertLess(elapsed, 1.9)  # two 1 s downloads overlapped
            journal = next((root / "journal").glob("*.jsonl")).read_text()
            self.assertIn('"stage": "prefetch.weights"', journal)
            self.assertIn('"side_stage": true', journal)
