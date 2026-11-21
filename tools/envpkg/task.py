"""Task/reward spec (task_spec/v1): what a lab's RL run optimises and how success is judged.

Built per clip from two sources, never typed by hand:

- the TRAINING environment exactly as Isaac Lab resolved it for this clip
  (``RUN_ROOT/simulate/<embodiment>/env_spec.json``, written by
  robotic_grounding/scripts/rsl_rl/replay_metrics.py before its QA overrides): rewards,
  terminations, action, observations, resets, assist curriculum, robot, timing;
- the clip's task profile (the success evaluator, e.g. ``lift_hold``).

The spec is simulator-neutral: every term names its reference implementation (``function``)
and states what it computes (``definition``), so the Isaac Lab exporter wires the same
terms and the MuJoCo exporter re-implements them. A term this module has no definition for
is an error, so a recipe change cannot slip into a package undescribed. The success
criterion is checked on the clip's own reference motion: a task the reference cannot pass
is refused. Contract: docs/pipeline/TASK_SPEC.md.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .common import ContractError

TASK_SCHEMA = "task_spec/v1"
ENV_SPEC_SCHEMA = "v2d_env_spec_v1"

# What each term computes. Keys are the function names Isaac Lab records ("module:name").
REWARDS = {
    "is_terminated": "1 on a step where a non-timeout termination fired, else 0",
    "motion_object_keypoints_tracking_exp": (
        "mean over object bodies and 6 keypoints (unit offsets +-x, +-y, +-z in the object frame) of "
        "exp(-||k_sim - k_ref||^2 / var)"
    ),
    "motion_hand_keypoints_gaussian_exp": (
        "sum over hands of exp(-sum_k ||k_sim - k_ref||^2 / std^2), keypoints = wrist position and "
        "fingertip positions"
    ),
    "force_closure_reward": (
        "for each hand whose reference contact label is active: fraction of wrench-basis directions "
        "whose simulated contact-wrench support (max over the hand's bodies) exceeds min_support; "
        "averaged over active hands"
    ),
    "action_rate_l2": "||a_t - a_{t-1}||^2",
    "action_l2": "||a_t||^2",
    "motion_finger_joint_pos_gaussian_exp": "sum over hands of exp(-||q_finger_ref - q_finger||^2 / std^2)",
    "motion_contact_tracking_gaussian_exp": (
        "sum over hands of exp(-d^2 / std^2), d = chamfer distance between simulated and reference "
        "hand-object contact points (0 when neither has contacts)"
    ),
    "contact_wrench_support_reward": (
        "per wrench-basis direction where both reference and simulation have contact support: "
        "exp(-loss / var) (loss beyond `tolerance`), averaged over the reference's active directions "
        "per body, then over bodies and hands; in [0, 1]"
    ),
    "unintended_contact_penalty": (
        "simulated hand-object contact on bodies where the reference expects none: an indicator plus a "
        "term proportional to that contact's wrench support"
    ),
    "missed_contact_penalty": (
        "per body with expected contact, the fraction of expected wrench-basis directions the simulation "
        "does not support; averaged over those bodies, then hands; in [0, 1]"
    ),
}
TERMINATIONS = {
    "timestep_termination": "the reference trajectory's last frame is reached (time-out, not a failure)",
    "time_out": "the episode reached its maximum length (time-out, not a failure)",
    "hand_wrist_away_from_trajectory": (
        "either wrist farther than `threshold` m from its reference (not checked during the reset freeze)"
    ),
    "object_pos_error": "object position farther than `threshold` m from the reference (not during the reset freeze)",
    "object_quat_error": "object orientation more than `threshold` rad from the reference (not during the reset freeze)",
    "robot_state_diverged": "robot state non-finite or any joint speed above `max_joint_vel` rad/s",
}
OBSERVATIONS = {
    "joint_pos_rel": "joint positions minus their defaults",
    "joint_vel_rel": "joint velocities",
    "motion_joint_pos_delta": "reference joint positions at the next `num_future_frames` frames minus current joint positions",
    "motion_ee_pos_delta": "reference wrist positions at the next `num_future_frames` frames minus current wrist positions",
    "motion_ee_quat_delta": (
        "rotation from each current wrist orientation to its reference at the next `num_future_frames` "
        "frames, as a 6D rotation representation"
    ),
    "hand_object_reference_transform": (
        "the `side` wrist's pose (position, quaternion w-x-y-z) in the frame of the reference object pose; "
        "zero when the wrist is farther than `threshold` m from the object"
    ),
    "object_pose_delta": "reference object pose (position, quaternion w-x-y-z) expressed in the current object frame",
    "wrist_position_e": "wrist positions in the environment frame, [right, left]",
    "wrist_wxyz_e": "wrist orientations (w, x, y, z) in the environment frame, [right, left]",
    "object_position_e": "object position in the environment frame",
    "object_wxyz_e": "object orientation (w, x, y, z) in the environment frame",
    "command_trajectory_progress": "current reference frame / last reference frame",
    "last_action": "the previous action",
}
ASSIST_ACTIONS = {
    "VirtualRigidObjectControl": (
        "not a policy output: a PD wrench on the object toward its reference pose (linear/angular "
        "stiffness and damping, capped at max_force / max_torque), multiplied by the assist scale"
    ),
}
ACTIONS = {
    "ReferenceResidualJointPositionAction": (
        "joint position target = reference joint position at the current frame + scale * EMA(clip(a, +-clip), ema_factor), "
        "clipped to joint limits when clip_to_joint_limits"
    ),
}
SUCCESS = {
    "lift_hold": (
        "object height z relative to its first sample, sampled each control step: success when the "
        "max lift >= lift_threshold_m, the lift stays >= hold_threshold_m for >= min_hold_steps "
        "consecutive steps, the final lift >= final_min_lift_m, the last placement_window_steps "
        "samples are placed (all <= placement_threshold_m, step changes <= max_terminal_step_m, "
        "range <= max_terminal_range_m), and no step-to-step descent after holding exceeds "
        "max_descent_step_m"
    ),
}
COMMAND_KEYS = (
    "dt", "motion_speed", "reset_freeze_steps", "reset_to_first_frame_prob", "reset_finger_openness",
    "reset_shoulder_spread", "initial_virtual_object_control_curriculum_scale", "voc_decay_steps",
    "voc_reset_scale", "anchor_body_name", "ee_link_names", "left_wrist_body_name", "right_wrist_body_name",
    "fingertip_body_name", "hand_contact_bodies", "num_future_frames", "dt_future_frames",
)


def _name(function: Any) -> str:
    if not isinstance(function, str) or ":" not in function:
        raise ContractError(f"env spec: term function {function!r} is not a recorded 'module:name'")
    return function.rsplit(":", 1)[1]


def _terms(section: dict[str, Any] | None, catalog: dict[str, str], kind: str) -> list[tuple[str, dict, str, str]]:
    """(name, term, function id, definition) for every configured term of a manager section."""
    if not isinstance(section, dict) or "_unserializable" in (section or {}):
        raise ContractError(f"env spec: {kind} section missing or unserializable: {section!r:.200}")
    out = []
    for name, term in section.items():
        if not isinstance(term, dict) or "func" not in term:
            continue  # group flags (concatenate_terms, ...) and non-term fields
        function = term["func"]
        short = _name(function)
        if short not in catalog:
            raise ContractError(f"env spec: {kind} term {name!r} uses {function}, which has no definition in "
                                f"tools/envpkg/task.py; describe it before packaging")
        out.append((name, term, function, catalog[short]))
    return out


def _noise(noise: Any) -> dict[str, Any] | None:
    if not isinstance(noise, dict):
        return None
    if "n_min" in noise and "n_max" in noise and noise.get("operation", "add") == "add":
        return {"kind": "additive_uniform", "min": noise["n_min"], "max": noise["n_max"]}
    raise ContractError(f"env spec: unsupported observation noise {noise}")


def reference_object_heights(motion: dict[str, Any], *, control_dt: float, motion_speed: float) -> np.ndarray:
    """The reference object's height at each control step of a frame-0 episode (the command
    consumes the reference at ``motion_speed`` x real time, linearly interpolated)."""
    z = np.asarray(motion["object_root_position"], dtype=float)[:, 2]
    times = np.arange(len(z)) / float(motion["fps"])
    samples = np.arange(0.0, times[-1] + 1e-9, control_dt * motion_speed)
    return np.interp(samples, times, z)


def _evaluate(evaluator: dict[str, Any], heights: np.ndarray) -> dict[str, Any]:
    from .success import EVALUATORS

    kind = evaluator.get("id")
    if kind not in EVALUATORS:
        raise ContractError(f"no implementation of success evaluator {kind!r}")
    return EVALUATORS[kind](heights, {k: v for k, v in evaluator.items() if k != "id"})


def robot_task(env_spec: dict[str, Any], motion: dict[str, Any], motion_path: str, evaluator: dict[str, Any],
               env_spec_path: str) -> dict[str, Any]:
    """The per-robot part of the task spec from one resolved Isaac Lab environment."""
    if env_spec.get("schema") != ENV_SPEC_SCHEMA:
        raise ContractError(f"env spec schema {env_spec.get('schema')!r}, expected {ENV_SPEC_SCHEMA}")
    if "runtime" not in env_spec:
        raise ContractError(f"env spec has no runtime section ({env_spec.get('runtime_error', 'missing')})")
    runtime = env_spec["runtime"]
    command = env_spec.get("command") or {}
    missing = [k for k in ("dt", "motion_speed", "reset_freeze_steps") if k not in command]
    if missing:
        raise ContractError(f"env spec command lacks {missing}")
    step_dt = float(runtime["step_dt"])
    if not math.isclose(step_dt, env_spec["sim_dt"] * env_spec["decimation"], rel_tol=1e-6):
        raise ContractError(f"env spec: step_dt {step_dt} != sim_dt x decimation")

    actions, assists = [], []
    for name, term in (env_spec.get("actions") or {}).items():
        kind = str(term.get("class_type", "")).rsplit(":", 1)[-1].rsplit(".", 1)[-1]
        if kind in ASSIST_ACTIONS:
            assists.append({"name": name, "kind": kind, "definition": ASSIST_ACTIONS[kind],
                            **{k: v for k, v in term.items() if k not in ("class_type", "debug_vis", "clip")}})
            continue
        if kind not in ACTIONS:
            raise ContractError(f"env spec: action {name!r} of type {term.get('class_type')} has no definition")
        actions.append({
            "name": name, "kind": kind, "definition": ACTIONS[kind],
            **{k: term.get(k) for k in ("joint_names", "scale", "clip", "ema_factor", "clip_to_joint_limits")},
        })

    policy = (env_spec.get("observations") or {}).get("policy")
    obs_dims = runtime["observation_terms"].get("policy", {})
    observations = [
        {"name": name, "function": function, "definition": definition, "params": term.get("params") or {},
         "dim": obs_dims.get(name), "noise": _noise(term.get("noise")), "clip": term.get("clip"), "scale": term.get("scale")}
        for name, term, function, definition in _terms(policy, OBSERVATIONS, "observation")
    ]
    rewards = [
        {"name": name, "function": function, "definition": definition, "weight": float(term["weight"]),
         "params": term.get("params") or {}}
        for name, term, function, definition in _terms(env_spec.get("rewards"), REWARDS, "reward")
    ]
    terminations = [
        {"name": name, "function": function, "definition": definition, "time_out": bool(term.get("time_out", False)),
         "params": term.get("params") or {}}
        for name, term, function, definition in _terms(env_spec.get("terminations"), TERMINATIONS, "termination")
    ]

    events = env_spec.get("events") or {}
    reset_event = (events.get("reset_to_trajectory_frame") or {}).get("params", {})
    material = (events.get("hand_physics_material") or {}).get("params", {})
    groups = (events.get("setup_collision_groups") or {}).get("params", {})
    curriculum = ((env_spec.get("curriculum") or {}).get("voc_curriculum") or {}).get("params")
    robot = env_spec.get("robot") or {}
    spawn = robot.get("spawn") or {}

    reference_steps = int(runtime["trajectory_end_timestep"]) + 1
    heights = reference_object_heights(motion, control_dt=step_dt, motion_speed=float(command["motion_speed"]))
    if abs(len(heights) - reference_steps) > 1:
        raise ContractError(f"reference resampled to {len(heights)} control steps, Isaac Lab plays {reference_steps}")
    check = _evaluate(evaluator, heights)
    # Isaac Lab ends an episode only through termination terms: max_episode_length is enforced
    # only when a `time_out` term uses it (isaaclab.envs.mdp.time_out). Without one, training
    # episodes run to the reference end (timestep_termination) or a failure.
    capped = any(_name(t["func"]) == "time_out" for t in (env_spec.get("terminations") or {}).values()
                 if isinstance(t, dict) and "func" in t)
    train_steps = int(runtime["train_max_episode_steps"]) if capped else None
    notes = []
    if capped and train_steps < reference_steps + int(command["reset_freeze_steps"]):
        notes.append(
            f"training episodes are capped at {train_steps} steps, shorter than a frame-0 pass over the "
            f"reference ({reference_steps} steps + {command['reset_freeze_steps']} freeze)"
        )
    return {
        "isaaclab_task": env_spec["task"],
        "env_spec": env_spec_path,
        "reference": {"motion": motion_path, "fps": float(motion["fps"]), "frames": len(motion["robot_joint_positions"]),
                      "playback_speed": float(command["motion_speed"]), "control_steps": reference_steps},
        "timing": {"physics_dt_s": float(env_spec["sim_dt"]), "decimation": int(env_spec["decimation"]), "control_dt_s": step_dt},
        "action": {"dim": int(runtime["action_dim"]), "joint_names": runtime.get("joint_names", []), "terms": actions},
        "observations": {"dim": runtime["observation_dims"].get("policy"),
                         "concatenated": bool((policy or {}).get("concatenate_terms", True)),
                         "corruption": bool((policy or {}).get("enable_corruption", False)), "terms": observations},
        "rewards": rewards,
        "terminations": terminations,
        "reset": {
            "start_frame": "uniform over the reference" if float(command.get("reset_to_first_frame_prob", 0.0)) < 1.0 else "first",
            "first_frame_probability": float(command.get("reset_to_first_frame_prob", 0.0)),
            "freeze_steps": int(command["reset_freeze_steps"]),
            "finger_openness": command.get("reset_finger_openness"),
            "event_params": reset_event,
        },
        "assist": {
            "virtual_object_control": {
                "description": "a virtual controller pulls the object toward its reference pose, scaled by "
                               "`scale`; the curriculum anneals it to 0 so the final policy manipulates through "
                               "contact alone; evaluation runs with it off",
                "initial_scale": command.get("initial_virtual_object_control_curriculum_scale"),
                "controllers": assists,
                "curriculum": curriculum,
            },
        },
        "physics": {
            "robot_description": spawn.get("asset_path"),
            "robot_gravity": not bool((spawn.get("rigid_props") or {}).get("disable_gravity", False)),
            "hand_contact_material": {"bodies": (material.get("asset_cfg") or {}).get("body_names"),
                                      "static_friction": material.get("static_friction_range"),
                                      "dynamic_friction": material.get("dynamic_friction_range"),
                                      "restitution": material.get("restitution_range")},
            "object_material": "core.object.physics.friction (the simulator default; not set by the task)",
            "collisions": {"robot_object": not bool(groups.get("disable_robot_to_object_collisions", False)),
                           "robot_support": not bool(groups.get("disable_robot_to_fixed_object_collisions", False))},
            "object_solver_iterations": env_spec.get("object_solver_iterations"),
            "actuators": robot.get("actuators"),
        },
        "command": {k: command.get(k) for k in COMMAND_KEYS if k in command},
        "episode": {
            "train": {"max_steps": train_steps, "ends": "reference end (time-out) or a failure termination"
                      + ("" if capped else "; no episode-length cap (no time_out term)"),
                      "start": "reset distribution above", "assist": "curriculum",
                      "configured_episode_length_s": env_spec.get("episode_length_s")},
            "evaluation": {"start_frame": 0, "assist_scale": 0.0, "max_steps": reference_steps,
                           "pass": "no non-timeout termination before the reference ends, and the success criterion holds"},
        },
        "reference_check": check,
        "notes": notes,
    }


def build_task_spec(*, env_id: str, sequence: str, instruction: str | None, object_name: str,
                    profile: dict[str, Any], robots: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """task_spec/v1 from the clip's task profile and the per-robot parts (robot_task)."""
    evaluator = dict(profile.get("evaluator") or {})
    kind = evaluator.get("id")
    if kind not in SUCCESS:
        raise ContractError(f"task profile evaluator {kind!r} has no definition (known: {sorted(SUCCESS)})")
    failed = [name for name, part in robots.items() if not part["reference_check"]["success"]]
    if failed:
        raise ContractError(f"the reference motion itself fails the {kind} criterion for {failed}: "
                            f"{ {n: robots[n]['reference_check'] for n in failed} }")
    return {
        "schema": TASK_SCHEMA,
        "task_id": f"{env_id}.{profile.get('task_id', kind)}",
        "instruction": instruction or profile.get("instruction"),
        "object": object_name,
        "profile": profile.get("task_id"),
        "success": {"evaluator": kind, "params": {k: v for k, v in evaluator.items() if k != "id"},
                    "definition": SUCCESS[kind]},
        "robots": robots,
    }


def load_task_spec(package: Path) -> dict[str, Any]:
    """The task spec of a package (exporters read it through this)."""
    manifest = json.loads((Path(package) / "manifest.json").read_text())
    task = manifest.get("task")
    if not task:
        raise ContractError(f"{package}: no task spec attached")
    return json.loads((Path(package) / task["spec"]).read_text())
