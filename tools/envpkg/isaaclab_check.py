"""Prove a package's Isaac Lab export in a real Isaac Lab process.

For each robot in ``isaaclab/env.json``, on a scratch copy of the package (Isaac's URDF
importer may write converted assets beside the URDF), run the export's own ``replay_check``
command, i.e. robotic_grounding's replay_metrics.py against
``HUMAN_MOTION_DATA_DIR=<copy>/isaaclab/human_motion_data``, and require:

1. the robot description Isaac Lab loaded hashes to the package's robot model hash;
2. the resolved training environment equals the one the task spec was built from
   (rewards, terminations, actions, observations, resets, assist, command; runtime dims);
3. the physics-QA replay reaches the same verdict as the clip's ``simulate`` replay:
   same flag, completion within 0.02, each check within its tolerance.

Run inside the robotic_grounding Isaac Lab image, from any directory:

    python -m tools.envpkg.isaaclab_check <package> --rg-root /workspace/video_to_data/robotic_grounding \
        --isaac-python "/workspace/isaaclab/isaaclab.sh -p" [--out report.json]

Exit 0 when every robot matches, 1 otherwise. Contract: docs/pipeline/ISAACLAB_EXPORT.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# CPU PhysX is deterministic per container but not across containers: the same clip replayed
# in two containers differs by up to ~1.5 mm on these checks (docs/pipeline/OPTIMIZATION.md,
# simulate). Tolerances sit well above that and well below the flag thresholds' margins.
POSITION_TOL_M = 0.005
FRACTION_TOL = 0.05
COMPLETION_TOL = 0.02
ENV_SECTIONS = ("actions", "observations", "rewards", "terminations", "events", "curriculum", "command",
                "decimation", "sim_dt", "episode_length_s", "object_solver_iterations")
RUNTIME_KEYS = ("step_dt", "action_dim", "observation_dims", "observation_terms", "reward_terms",
                "termination_terms", "joint_names", "trajectory_end_timestep", "num_timesteps")
PATH_KEYS = {"motion_file", "asset_path", "usd_path", "motion_folder", "scene_config_path"}


def _strip_paths(value: Any) -> Any:
    """Drop machine-specific path fields; keep everything that defines the environment."""
    if isinstance(value, dict):
        return {k: _strip_paths(v) for k, v in value.items() if k not in PATH_KEYS}
    if isinstance(value, list):
        return [_strip_paths(v) for v in value]
    if isinstance(value, str) and value.startswith("/"):
        return os.path.basename(value.rstrip("/"))
    return value


def compare_env(expected: dict[str, Any], fresh: dict[str, Any]) -> list[str]:
    """Differences between the packaged env dump and a fresh one (empty: identical)."""
    diffs = []
    for key in ENV_SECTIONS:
        if _strip_paths(expected.get(key)) != _strip_paths(fresh.get(key)):
            diffs.append(f"env section {key!r} differs")
    for key in RUNTIME_KEYS:
        if expected.get("runtime", {}).get(key) != fresh.get("runtime", {}).get(key):
            diffs.append(f"runtime {key!r}: {expected.get('runtime', {}).get(key)} != {fresh.get('runtime', {}).get(key)}")
    return diffs


def compare_replay(expected: dict[str, Any], fresh: dict[str, Any]) -> list[str]:
    """Differences between the packaged physics-QA verdict and a fresh replay's."""
    diffs = []
    if fresh.get("flag") != expected["flag"]:
        diffs.append(f"flag {fresh.get('flag')} != {expected['flag']} ({fresh.get('flag_reasons')})")
    if abs(float(fresh.get("completion", -1)) - float(expected["completion"])) > COMPLETION_TOL:
        diffs.append(f"completion {fresh.get('completion')} != {expected['completion']}")
    for name, value in expected.get("checks", {}).items():
        got = fresh.get("checks", {}).get(name)
        tol = POSITION_TOL_M if name.endswith("_m") else FRACTION_TOL
        if got is None or abs(float(got) - float(value)) > tol:
            diffs.append(f"check {name}: {got} != {value} (tolerance {tol})")
    return diffs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_robot(copy: Path, robot: str, part: dict[str, Any], *, rg_root: Path, isaac_python: str,
                work: Path) -> dict[str, Any]:
    out = work / robot
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "HUMAN_MOTION_DATA_DIR": str(copy / "isaaclab/human_motion_data"),
           "ROBOTIC_GROUNDING_INTERMEDIATE_DIR": str(work / "intermediate"),
           "ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y", "OMNI_KIT_ACCEPT_EULA": "YES"}
    argv = [*shlex.split(isaac_python), "scripts/rsl_rl/replay_metrics.py", "--headless", "--device", "cpu",
            "--task", part["task"], "--motion_file", part["motion_file"],
            "--output", str(out / "replay_metrics.json"), "--env_spec_out", str(out / "env_spec.json")]
    start = time.time()
    with open(out / "run.log", "w") as log:
        code = subprocess.run(argv, cwd=rg_root, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    result: dict[str, Any] = {"exit_code": code, "seconds": round(time.time() - start, 1), "log": str(out / "run.log")}
    if code != 0 or not (out / "replay_metrics.json").is_file() or not (out / "env_spec.json").is_file():
        result["errors"] = [f"replay failed (exit {code}); see {out / 'run.log'}"]
        return result
    fresh_replay = json.loads((out / "replay_metrics.json").read_text())
    fresh_env = json.loads((out / "env_spec.json").read_text())
    errors = []
    asset = (fresh_env.get("robot") or {}).get("spawn", {}).get("asset_path")
    if not asset or not Path(asset).is_file():
        errors.append(f"robot description not found at {asset}")
    elif _sha256(Path(asset)) != part["robot_description_sha256"]:
        errors.append(f"robot description {asset} differs from the package's (sha256)")
    packaged_env = json.loads((copy / part["env_spec"]).read_text())
    errors += compare_env(packaged_env, fresh_env)
    if part["expected_replay"] is not None:
        errors += compare_replay(part["expected_replay"], fresh_replay)
    result.update({
        "errors": errors,
        "replay": {k: fresh_replay.get(k) for k in ("flag", "flag_reasons", "completion", "checks", "ended", "termination_terms")},
        "expected": part["expected_replay"],
    })
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("package", type=Path)
    parser.add_argument("--rg-root", type=Path, required=True, help="robotic_grounding checkout (the extension)")
    parser.add_argument("--isaac-python", default="python", help="command that runs Isaac Lab's python")
    parser.add_argument("--robot", action="append", help="robots to check (default: all exported)")
    parser.add_argument("--out", type=Path, help="report JSON")
    args = parser.parse_args(argv)
    env_json = json.loads((args.package / "isaaclab/env.json").read_text())
    with tempfile.TemporaryDirectory(prefix="isaaclab_check_") as tmp:
        work = Path(tmp)
        copy = work / "package"
        shutil.copytree(args.package, copy)
        robots = {name: part for name, part in env_json["robots"].items() if not args.robot or name in args.robot}
        results = {name: check_robot(copy, name, part, rg_root=args.rg_root, isaac_python=args.isaac_python, work=work)
                   for name, part in robots.items()}
        written = sorted(str(p.relative_to(copy)) for p in copy.rglob("*")
                         if p.is_file() and not (args.package / p.relative_to(copy)).exists())
        for name, result in results.items():
            log = Path(result["log"])
            result["log_tail"] = log.read_text(errors="replace").splitlines()[-15:] if result["errors"] and log.is_file() else []
    report = {"schema": "isaaclab_check/v1", "package": str(args.package), "ok": all(not r["errors"] for r in results.values()),
              "robots": results, "files_written_into_package": written}
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    print(f"ISAACLAB_CHECK {'OK' if report['ok'] else 'FAILED'}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
