# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Camera-free physics replay of a retargeted clip, with numeric QA metrics.

Plays the motion from frame 0 with zero residual actions (the robot PD-tracks the retargeted
reference), the virtual object control (VOC) assist OFF and the curriculum off, so the object
moves only through contact physics. Per control step it records the tracking errors the
environment's own terminations use (object position / orientation vs the reference, each
wrist vs the reference) and stops at the first termination or at the end of the motion.

This is what a reviewer checks by eye in the environment video: the object sits on its
support (does not float or sink), the hands reach it, and it follows the reference once
grasped. It needs no RTX cameras, so it runs wherever Isaac physics runs (Modal included).

The output JSON holds the raw series summaries, the completion ratio (reference frames
reached / last reference frame, as in ``record_dataset.py``) and a ``flag`` with reasons
when a metric exceeds its threshold. A flag is data, not a failure: the exit code is 0
unless the replay itself cannot run or a metric cannot be read.

Usage:
    python scripts/rsl_rl/replay_metrics.py --headless --task VegaSharpa-WholeBody-Manip-v0 \
        --motion_file ego_recon/processed/sequence_id=<seq>/robot_name=vega_sharpa \
        --output /tmp/replay_metrics.json
"""

import argparse
import json
import sys
import time

from isaaclab.app import AppLauncher

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# The open-loop replay cannot hold the object through the whole clip even when the clip is
# good (zero residual actions; the trained policy supplies the corrections), so the flags
# use the phases an open-loop replay does judge: the object at rest before the first
# reference contact (float / sink / drift), the first second of reference contact (the
# hands reach and touch the object; it follows the reference), and the robot's own wrist
# tracking. Completion (open-loop hold) is reported, not flagged.
#
# Flag thresholds, calibrated 2026-11-20 on three clips judged good on review
# (tissue_box_simple, and the fast-pipeline tissue_box clips b3_tbseed and b3_tb2) and four
# controlled defects of tissue_box_simple: object raised 4 cm, lowered 2 cm, shifted 5 cm,
# and played 0.5 s behind the hands. Every good clip passes and every defect is flagged;
# values and margins are in docs/pipeline/OPTIMIZATION.md in the monorepo. The settle check
# and the replay ending before contact catch placement defects with wide margins; the
# grasp-drift check catches the timing defect with a thin one. Re-calibrate as reviewed
# clips accumulate. The rest and grasp object errors are loose backstops: real
# reconstructions drift up to ~5 cm before contact and still look right. The environment's
# own terminations (object 0.10 m / 0.5 rad, wrist 0.15 m) are RL-episode limits.
DEFAULT_THRESHOLDS = {
    "settle_object_pos_max_m": 0.025,
    "rest_object_pos_p95_m": 0.065,
    "grasp_missed_contact_mean": 0.5,
    "grasp_object_pos_p95_m": 0.075,
    "wrist_pos_p95_m": 0.05,
    "grasp_drift_p95_m": 0.041,
}
SETTLE_WINDOW_S = 0.5  # the object's first moments on its support: floats, sinks, overlaps
GRASP_WINDOW_S = 1.0  # reference-contact time after the first contact that is checked

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--motion_file", type=str, required=True)
parser.add_argument("--output", type=str, required=True, help="metrics JSON path")
parser.add_argument(
    "--states_out",
    type=str,
    default="",
    help="also save the replayed physics states (robot joints, object and reference poses) "
    "to this .npz for scripts/render_replay_overlay.py; default: replay_states.npz next to "
    "--output; 'none' to skip",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="step cap; 0 = the motion's length plus a margin",
)
parser.add_argument(
    "--env_spec_out",
    type=str,
    default="",
    help="also write the TRAINING environment's resolved configuration (rewards, terminations, "
    "actions, observations, resets, curriculum, robot, timing; before the QA replay's overrides) "
    "plus its runtime dimensions to this JSON, the source of the package's task spec "
    "(tools/envpkg/task.py in the monorepo); default: env_spec.json next to --output; 'none' to skip",
)
parser.add_argument(
    "--voc_scale",
    type=float,
    default=0.0,
    help="virtual object control assist (0 = physics only; the QA setting)",
)
for _name, _value in DEFAULT_THRESHOLDS.items():
    parser.add_argument(f"--{_name}", type=float, default=_value)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Isaac imports after the app starts."""

import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.utils.math import quat_error_magnitude  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from robotic_grounding.tasks import *  # noqa: F401, F403, E402
from robotic_grounding.tasks.scene_utils import (  # noqa: E402
    SceneConfig,
    apply_scene_config,
)


def _command_name(env_cfg) -> str:
    commands = env_cfg.commands
    for name in ("motion", "dual_hands_object_tracking_command"):
        if getattr(commands, name, None) is not None:
            return name
    raise SystemExit(f"replay_metrics: no tracking command on {type(env_cfg).__name__}")


def _apply_qa_defaults(env_cfg, command_name: str, voc_scale: float) -> None:
    """Frame-0 reset, no settle freeze, VOC at `voc_scale`, curriculum off, markers off."""
    cmd = getattr(env_cfg.commands, command_name)
    for attr, value in (
        ("always_reset_to_first_frame", True),
        ("reset_to_first_frame_prob", 1.0),
        ("reset_freeze_steps", 0),
        ("debug_vis", False),
        ("voc_decay_steps", 0),
        ("virtual_object_control_decay_steps", 0),
    ):
        if hasattr(cmd, attr):
            setattr(cmd, attr, value)
    cmd.initial_virtual_object_control_curriculum_scale = voc_scale
    if hasattr(cmd, "voc_reset_scale"):
        cmd.voc_reset_scale = voc_scale
    if hasattr(env_cfg, "curriculum"):
        env_cfg.curriculum = None


ENV_SPEC_SCHEMA = "v2d_env_spec_v1"


def _cfg_dict(obj) -> object:
    """A configclass as plain data (functions become "module:name"), or the error."""
    if obj is None:
        return None
    try:
        return obj.to_dict()
    except Exception as exc:  # noqa: BLE001 - recorded, and the task-spec builder refuses it
        return {"_unserializable": f"{type(exc).__name__}: {exc}"}


def _training_env_spec(env_cfg, command_name: str) -> dict:
    """The training configuration as resolved for this clip, before _apply_qa_defaults edits it."""
    robot = env_cfg.scene.robot
    return {
        "schema": ENV_SPEC_SCHEMA,
        "task": args_cli.task,
        "motion_file": args_cli.motion_file,
        "command_name": command_name,
        "decimation": int(env_cfg.decimation),
        "sim_dt": float(env_cfg.sim.dt),
        "episode_length_s": float(env_cfg.episode_length_s),
        "actions": _cfg_dict(env_cfg.actions),
        "observations": _cfg_dict(env_cfg.observations),
        "rewards": _cfg_dict(env_cfg.rewards),
        "terminations": _cfg_dict(env_cfg.terminations),
        "events": _cfg_dict(env_cfg.events),
        "curriculum": _cfg_dict(getattr(env_cfg, "curriculum", None)),
        "command": _cfg_dict(getattr(env_cfg.commands, command_name)),
        "robot": _cfg_dict(robot),
        "object_solver_iterations": [
            getattr(env_cfg, "object_solver_position_iteration_count", None),
            getattr(env_cfg, "object_solver_velocity_iteration_count", None),
        ],
    }


def _runtime_spec(unwrapped, command, train_episode_length_s: float) -> dict:
    """Dimensions and term lists of the built environment (unchanged by the QA overrides)."""
    obs = unwrapped.observation_manager
    try:
        joint_names = list(unwrapped.scene["robot"].data.joint_names)
    except KeyError:
        joint_names = []
    return {
        "step_dt": float(unwrapped.step_dt),
        "action_dim": int(unwrapped.action_manager.total_action_dim),
        "observation_dims": {g: list(np.atleast_1d(d).tolist()) for g, d in obs.group_obs_dim.items()},
        "observation_terms": {
            g: dict(zip(obs.active_terms[g], [list(np.atleast_1d(d).tolist()) for d in obs.group_obs_term_dim[g]], strict=True))
            for g in obs.active_terms
        },
        "reward_terms": list(unwrapped.reward_manager.active_terms),
        "termination_terms": list(unwrapped.termination_manager.active_terms),
        "joint_names": joint_names,
        "trajectory_end_timestep": int(_first(getattr(command, "trajectory_end_timestep", command.num_timesteps - 1))),
        "num_timesteps": int(command.num_timesteps),
        "train_max_episode_steps": int(np.ceil(train_episode_length_s / float(unwrapped.step_dt) - 1e-9)),
    }


def _first(x) -> float:
    return float(x.reshape(-1)[0]) if torch.is_tensor(x) else float(x)


def _object_errors(command) -> tuple[float, float]:
    """Worst object body position (m) and orientation (rad) error vs the reference, env 0."""
    if hasattr(command, "object_position_e") and hasattr(
        command, "object_body_position_command_e"
    ):
        pos = torch.norm(
            command.object_position_e - command.object_body_position_command_e, dim=-1
        )[0].max()
        rot = quat_error_magnitude(
            command.object_orientation_e.reshape(-1, 4),
            command.object_body_wxyz_command_e.reshape(-1, 4),
        ).reshape(command.num_envs, -1)[0].max()
    else:
        pos = torch.norm(command.object_pos_w - command.command_object_pos_w, dim=-1)[0]
        rot = quat_error_magnitude(command.object_quat_w, command.command_object_quat_w)[0]
    return float(pos), float(rot)


def _wrist_errors(command) -> list[float]:
    """Per-wrist (end-effector) position error vs the reference (m), env 0."""
    error = torch.norm(command.command_ee_pos_w - command.robot_ee_pos_w, dim=-1)
    return [float(v) for v in error[0].reshape(-1)]


def _contact(command) -> tuple[float, float]:
    """Reference contact labels (left, right) at the current timestep, env 0."""
    left = getattr(command, "left_hand_contact_active_command", None)
    right = getattr(command, "right_hand_contact_active_command", None)
    return (_first(left) if left is not None else 0.0, _first(right) if right is not None else 0.0)


def _missed(env) -> float:
    """Fraction of expected contact wrench directions missing in sim (0 when none expected)."""
    from robotic_grounding.tasks.v2d_whole_body.mdp.rewards.contact_rewards import (
        missed_contact_penalty,
    )

    return _first(missed_contact_penalty(env, command_name="motion"))


def _grasp_drift(motion_dir: str) -> dict:
    """Kinematic wrist-in-object drift over reference contact (scripts/data_quality_checks)."""
    import glob

    import pyarrow.parquet as pq

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from data_quality_checks.grasp_drift import grasp_drift

    files = sorted(glob.glob(os.path.join(motion_dir, "*.parquet")))
    if not files:
        raise SystemExit(f"replay_metrics: no parquet in {motion_dir}")
    return grasp_drift(pq.read_table(files[0]).to_pydict())


def _object_mesh_path(motion_dir: str) -> str:
    """The object's visual mesh as recorded in the motion parquet (first body)."""
    import glob

    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(motion_dir, "*.parquet")))
    paths = pq.read_table(files[0], columns=["object_mesh_paths"]).column(0)[0].as_py() if files else []
    return str(paths[0]) if paths else ""


def _summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 5),
        "p95": round(float(np.percentile(a, 95)), 5),
        "max": round(float(a.max()), 5),
        "first": round(float(a[0]), 5),
    }


def main() -> int:
    t0 = time.time()
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.motion_file = args_cli.motion_file
    scene_config = SceneConfig.from_motion_file(env_cfg.motion_file)
    apply_scene_config(env_cfg, scene_config)
    drift = _grasp_drift(str(scene_config.motion_file))
    command_name = _command_name(env_cfg)
    env_spec = _training_env_spec(env_cfg, command_name)
    _apply_qa_defaults(env_cfg, command_name, args_cli.voc_scale)

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    command = unwrapped.command_manager.get_term(command_name)
    env.reset()
    t_ready = time.time()
    env_spec_out = args_cli.env_spec_out or os.path.join(
        os.path.dirname(os.path.abspath(args_cli.output)), "env_spec.json"
    )
    if env_spec_out != "none":
        try:
            env_spec["runtime"] = _runtime_spec(unwrapped, command, env_spec["episode_length_s"])
        except Exception as exc:  # noqa: BLE001 - recorded; the task-spec builder refuses the file
            env_spec["runtime_error"] = f"{type(exc).__name__}: {exc}"
            print(f"[WARN] replay_metrics: env spec runtime section failed: {exc}")
        os.makedirs(os.path.dirname(os.path.abspath(env_spec_out)), exist_ok=True)
        with open(env_spec_out, "w") as f:
            json.dump(env_spec, f, indent=1, default=str)

    end = int(_first(getattr(command, "trajectory_end_timestep", command.num_timesteps - 1)))
    start = int(_first(command.timestep))
    max_steps = args_cli.max_steps or (end - start + 10)
    actions = torch.zeros(env.action_space.shape, device=unwrapped.device)

    object_pos: list[float] = []
    object_rot: list[float] = []
    wrists: list[list[float]] = []
    contacts: list[tuple[float, float]] = []
    missed: list[float] = []
    timesteps: list[int] = []
    whole_body = command_name == "motion"

    def record() -> None:
        p, r = _object_errors(command)
        object_pos.append(p)
        object_rot.append(r)
        wrists.append(_wrist_errors(command))
        contacts.append(_contact(command))
        missed.append(_missed(unwrapped) if whole_body else float("nan"))
        timesteps.append(int(_first(command.timestep)))

    states: dict[str, list] = {
        key: []
        for key in ("joint_pos", "object_pos", "object_wxyz", "ref_object_pos",
                    "ref_object_wxyz", "ee_pos", "ref_ee_pos", "timestep")
    }
    states_error: list[str] = []
    robot = unwrapped.scene["robot"] if whole_body else None
    origin = unwrapped.scene.env_origins[0]

    def _vec(x, width: int) -> np.ndarray:
        return x[0].reshape(-1, width)[0].detach().cpu().numpy().astype(np.float32)

    def record_state() -> None:
        if robot is None or states_error:
            return
        try:
            states["joint_pos"].append(robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32))
            states["object_pos"].append(_vec(command.object_pos_w - origin, 3))
            states["object_wxyz"].append(_vec(command.object_quat_w, 4))
            states["ref_object_pos"].append(_vec(command.command_object_pos_w - origin, 3))
            states["ref_object_wxyz"].append(_vec(command.command_object_quat_w, 4))
            states["ee_pos"].append((command.robot_ee_pos_w[0] - origin).reshape(-1, 3).detach().cpu().numpy())
            states["ref_ee_pos"].append((command.command_ee_pos_w[0] - origin).reshape(-1, 3).detach().cpu().numpy())
            states["timestep"].append(int(_first(command.timestep)))
        except Exception as exc:  # noqa: BLE001 - the video is optional; the metrics are not
            states_error.append(f"{type(exc).__name__}: {exc}")
            print(f"[WARN] replay_metrics: not recording states for the overlay video: {exc}")

    reached, reasons, ended = start, [], "step_cap"
    # Errors at frame 0, before any physics step: catches an object placed inside or above
    # its support (it moves on the first steps) and hands far from their reference.
    record()
    record_state()
    for _ in range(max_steps):
        before = int(_first(command.timestep))
        with torch.inference_mode():
            _, _, terminated, truncated, _ = env.step(actions)
        if bool(_first(terminated)) or bool(_first(truncated)):
            # The env resets inside step(); the command now points at frame 0 again.
            reached = max(reached, before + 1)
            manager = unwrapped.termination_manager
            reasons = [n for n in manager.active_terms if bool(_first(manager.get_term(n)))]
            ended = "timeout" if bool(_first(truncated)) and not bool(_first(terminated)) else "terminated"
            break
        reached = int(_first(command.timestep))
        record()
        record_state()
    steps = len(object_pos) - 1
    t_done = time.time()
    joint_names = list(robot.joint_names) if robot is not None else []
    urdf_path = str(getattr(getattr(env_cfg.scene.robot, "spawn", None), "asset_path", "") or "")
    env.close()

    span = max(end - start, 1)
    completion = 1.0 if ended == "timeout" else min(max(reached - start, 0), span) / span
    if not all(np.isfinite(object_pos)) or not object_pos:
        raise SystemExit("replay_metrics: object error is not finite (no object in the scene?)")
    wrist_arr = np.asarray(wrists, dtype=np.float64)
    in_contact = [max(c) > 0.5 for c in contacts]
    first_contact = in_contact.index(True) if any(in_contact) else None
    grasp_steps = max(1, int(round(GRASP_WINDOW_S / float(unwrapped.step_dt))))
    settle_steps = max(1, int(round(SETTLE_WINDOW_S / float(unwrapped.step_dt))))
    settle = object_pos[: settle_steps + 1]
    rest = range(0, first_contact if first_contact is not None else len(object_pos))
    grasp = (
        [i for i in range(first_contact, len(object_pos)) if in_contact[i]][:grasp_steps]
        if first_contact is not None
        else []
    )
    metrics = {
        "object_pos_m": _summary(object_pos),
        "object_rot_rad": _summary(object_rot),
        "wrist_pos_m": {
            f"ee{i}": _summary(wrist_arr[:, i].tolist()) for i in range(wrist_arr.shape[1])
        },
        "rest_object_pos_m": _summary([object_pos[i] for i in rest]),
        "grasp_object_pos_m": _summary([object_pos[i] for i in grasp]),
        "grasp_missed_contact": _summary([missed[i] for i in grasp]),
        "settle_object_pos_m": _summary(settle),
        "grasp_drift_m": drift,
        "first_contact_step": first_contact,
        "grasp_steps_observed": len(grasp),
    }
    th = {name: getattr(args_cli, name) for name in DEFAULT_THRESHOLDS}
    wrist_p95 = max(v["p95"] for v in metrics["wrist_pos_m"].values())
    checks = {
        "settle_object_pos_max_m": metrics["settle_object_pos_m"].get("max"),
        "rest_object_pos_p95_m": metrics["rest_object_pos_m"].get("p95"),
        "grasp_missed_contact_mean": metrics["grasp_missed_contact"].get("mean"),
        "grasp_object_pos_p95_m": metrics["grasp_object_pos_m"].get("p95"),
        "wrist_pos_p95_m": wrist_p95,
        "grasp_drift_p95_m": drift["p95_m"],
    }
    flags = [
        f"{name} {value:.4f} (limit {th[name]})"
        for name, value in checks.items()
        if value is not None and value > th[name]
    ]
    if first_contact is None:
        flags.append(
            f"replay ended ({ended}: {', '.join(reasons) or 'none'}) before the first "
            f"reference contact, {steps} steps in"
        )
    elif len(grasp) < grasp_steps:
        flags.append(f"object lost {len(grasp)} steps into contact (window {grasp_steps})")
    series = {
        "timestep": timesteps,
        "object_pos_m": [round(v, 4) for v in object_pos],
        "object_rot_rad": [round(v, 4) for v in object_rot],
        "wrist_pos_m": [[round(v, 4) for v in w] for w in wrists],
        "contact_ref": [[int(c[0] > 0.5), int(c[1] > 0.5)] for c in contacts],
        "missed_contact": [round(v, 3) for v in missed],
    }
    report = {
        "schema": "v2d_replay_metrics_v1",
        "task": args_cli.task,
        "motion_file": args_cli.motion_file,
        "voc_scale": args_cli.voc_scale,
        "control_dt_s": float(unwrapped.step_dt),
        "start_timestep": start,
        "end_timestep": end,
        "reached_timestep": reached,
        "steps": steps,
        "ended": ended,
        "termination_terms": reasons,
        "completion": round(completion, 4),
        "metrics": metrics,
        "thresholds": th,
        "flag": bool(flags),
        "flag_reasons": flags,
        "checks": checks,
        "series": series,
        "timing_s": {"setup": round(t_ready - t0, 1), "replay": round(t_done - t_ready, 1)},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.output)), exist_ok=True)
    states_out = args_cli.states_out or os.path.join(
        os.path.dirname(os.path.abspath(args_cli.output)), "replay_states.npz"
    )
    if states_out != "none" and states["joint_pos"] and not states_error:
        np.savez_compressed(
            states_out,
            **{key: np.asarray(value) for key, value in states.items()},
            joint_names=np.asarray(joint_names),
            urdf_path=np.asarray(urdf_path),
            motion_dir=np.asarray(str(scene_config.motion_file)),
            object_mesh_path=np.asarray(_object_mesh_path(str(scene_config.motion_file))),
            control_dt_s=np.asarray(float(unwrapped.step_dt)),
            flag=np.asarray(bool(flags)),
            flag_reasons=np.asarray(json.dumps(flags)),
        )
        report["states_file"] = states_out
    elif states_out != "none":
        report["states_file"] = None
        report["states_error"] = states_error or ["no robot states (floating-hand task)"]
    with open(args_cli.output, "w") as f:
        json.dump(report, f, indent=1)
    print("REPLAY_METRICS " + json.dumps({k: report[k] for k in (
        "completion", "ended", "termination_terms", "flag", "flag_reasons", "timing_s")}))
    print("REPLAY_CHECKS " + json.dumps(checks) + f" first_contact={first_contact}")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except BaseException:  # noqa: BLE001 - Kit's launcher exits 0 on an uncaught exception
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    simulation_app.close()
    sys.exit(code)
