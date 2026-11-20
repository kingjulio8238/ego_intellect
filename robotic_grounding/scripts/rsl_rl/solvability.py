# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Solvability check: can a policy track this clip's reference in physics?

The constructive answer: train the clip's own expert task (same rewards, terminations,
actions and observations as ``train.py``) for a small budget, and evaluate the
deterministic policy from frame 0 with the virtual object control (VOC) assist off. A clip
is SOLVABLE when, inside the budget, a policy survives the whole reference (no wrist,
object-position, object-orientation or divergence termination before the timeout) in at
least ``--success_survival`` of the evaluation episodes. The policy that passes is saved:
it is the proof, and a baseline the lab can start from.

STATUS (2026-11-20): EXPERIMENTAL. The evaluation is validated (the bundled tissue-box
expert passes it 8/8), but the budget does not yet solve a known-good clip: 400 iterations
at 1024 or 4096 envs never pass the lift (the recipe needs ~20k). Results and the open
directions are in docs/pipeline/OPTIMIZATION.md ("Solvability check") in the monorepo.

The levers meant to make it cheap compared with the 20k-iteration expert recipe:
  * many environments on one GPU (``--num_envs``; the recipe's 4096 by default);
  * the VOC curriculum (1.0 -> 0.0 in 11 linear stages) compressed into
    ``--voc_anneal_iterations`` instead of 12,500 iterations;
  * early stop: the frame-0 evaluation runs every ``--eval_interval`` iterations once the
    assist is off, and the run stops at the first pass.

A run that exhausts ``--max_iterations`` without a pass is NOT SOLVED within the budget
(unsolvable, or harder than the budget), with the best evaluation reported.

The evaluation also applies the task's own evaluator (``--task_profile``, e.g.
``lift_hold``) to the simulated object height and to the reference object height. The
second one is free and settles whether the task as specified is reachable at all by
tracking this reference (a reference that never lifts 10 cm cannot pass a 10 cm lift).

Usage:
    python scripts/rsl_rl/solvability.py --headless --task VegaSharpa-WholeBody-Manip-v0 \
        --motion_file ego_recon/processed/sequence_id=<seq>/robot_name=vega_sharpa \
        --output <run>/solvability/vega/solvability.json \
        [--task_profile groot_finetune/task_profiles/tissue_box_lift_hold.json]
"""

import argparse
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--task", type=str, default="VegaSharpa-WholeBody-Manip-v0")
parser.add_argument("--motion_file", type=str, required=True)
parser.add_argument("--output", type=str, required=True, help="verdict JSON path")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=1500, help="PPO iteration budget")
parser.add_argument(
    "--voc_anneal_iterations",
    type=int,
    default=500,
    help="iterations over which the VOC assist decays 1.0 -> 0.0 (recipe: 12,500)",
)
parser.add_argument(
    "--eval_interval", type=int, default=100, help="iterations between frame-0 evaluations"
)
parser.add_argument(
    "--eval_before_anneal",
    action="store_true",
    help="also evaluate (assist off) while the curriculum still has assist on; diagnostic",
)
parser.add_argument(
    "--success_survival",
    type=float,
    default=0.8,
    help="fraction of frame-0 evaluation episodes that must reach the timeout",
)
parser.add_argument(
    "--eval_envs",
    type=int,
    default=0,
    help="environments counted in an evaluation (0 = all); the rest still step",
)
parser.add_argument("--task_profile", type=str, default="", help="task profile JSON (evaluator)")
parser.add_argument(
    "--no_stop",
    action="store_true",
    help="keep training to --max_iterations after a pass (learning-curve calibration)",
)
parser.add_argument(
    "--expert_onnx",
    type=str,
    default="",
    help="validation of the evaluation itself: run the frame-0 evaluation on this ONNX "
    "expert (e.g. the bundled tissue-box expert) and exit; no RL",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--checkpoint_out",
    type=str,
    default="",
    help="where to save the passing policy (default: policy.pt next to --output; 'none')",
)
parser.add_argument(
    "--qa_report",
    type=str,
    default="",
    help="simulate's replay_metrics.json: a flagged clip is skipped before Isaac starts "
    "(no RL spend on a clip that already failed the physics QA)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False


def _skip_flagged() -> bool:
    """Write a SKIPPED verdict for a clip the physics QA flagged; True when skipped."""
    if not args_cli.qa_report:
        return False
    with open(args_cli.qa_report) as f:
        qa = json.load(f)
    if not qa.get("flag"):
        return False
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.output)), exist_ok=True)
    report = {
        "schema": "v2d_solvability_v1",
        "task": args_cli.task,
        "motion_file": args_cli.motion_file,
        "solvable": None,
        "verdict": "SKIPPED: physics QA flagged the clip",
        "qa_report": args_cli.qa_report,
        "qa_flag_reasons": qa.get("flag_reasons", []),
    }
    with open(args_cli.output, "w") as f:
        json.dump(report, f, indent=1)
    print("SOLVABILITY " + json.dumps({k: report[k] for k in ("solvable", "verdict")}))
    return True


if _skip_flagged():
    sys.exit(0)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Isaac imports after the app starts."""

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from robotic_grounding.tasks import *  # noqa: F401, F403, E402
from robotic_grounding.tasks.scene_utils import (  # noqa: E402
    SceneConfig,
    apply_scene_config,
)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

VOC_STAGES = 10  # the recipe's 11-value linear ladder 1.0, 0.9, ..., 0.0


def _compress_curriculum(env_cfg, anneal_iterations: int) -> list[int]:
    """Same 11-stage linear VOC ladder, ending at `anneal_iterations` instead of 12,500."""
    term = getattr(getattr(env_cfg, "curriculum", None), "voc_curriculum", None)
    if term is None:
        raise SystemExit("solvability: the task has no voc_curriculum term to compress")
    schedule = [
        max(1, round(anneal_iterations * (k + 1) / VOC_STAGES)) for k in range(VOC_STAGES)
    ]
    term.params["timestep_schedule"] = schedule
    term.params["virtual_object_control_scale_factor"] = [
        round(1.0 - k / VOC_STAGES, 3) for k in range(VOC_STAGES + 1)
    ]
    return schedule


def _lift_hold(task_profile: str):
    """The task's own evaluator (lift_hold), or None without a profile."""
    if not task_profile:
        return None
    from groot_finetune.task_profile import load_task_profile

    return load_task_profile(task_profile).evaluator


def _reference_evaluation(command, evaluator) -> dict | None:
    """Apply the task evaluator to the reference object height over the evaluated window."""
    if evaluator is None:
        return None
    from groot_finetune.closed_loop.evaluation import evaluate_lift_hold

    pos = getattr(command, "_object_body_pos_w", None)  # (T, bodies, 3), env frame
    if pos is None:
        return {"error": "no reference object trajectory on the command"}
    trace = pos[:, 0, 2]
    start = int(command.reset_timestep[0])
    end = int(command.trajectory_end_timestep[0])
    trace = trace[start : end + 1].detach().cpu().numpy().astype(np.float64)
    result = evaluate_lift_hold(trace, config=evaluator)
    return {
        "success": result.success,
        "lift_hold_success": result.lift_hold_success,
        "max_lift_m": round(result.max_lift_m, 4),
        "max_hold_steps": result.max_hold_steps,
        "frames": int(trace.size),
    }


class FrameZeroEvaluation:
    """Deterministic policy from frame 0, assist off, until every counted env finishes once."""

    def __init__(self, env: RslRlVecEnvWrapper, command, evaluator, eval_envs: int):
        self.env = env
        self.unwrapped = env.unwrapped
        self.command = command
        self.evaluator = evaluator
        self.count = eval_envs if 0 < eval_envs <= env.num_envs else env.num_envs
        self.curriculum = None
        manager = self.unwrapped.curriculum_manager
        if manager is not None and "voc_curriculum" in manager.active_terms:
            index = manager.active_terms.index("voc_curriculum")
            self.curriculum = manager._term_cfgs[index].func

    def __call__(self, policy, max_steps: int) -> dict:
        # Everything that touches the env runs in inference_mode, like the runner's own
        # rollouts: the env's state tensors are created there and may only be updated in
        # place there.
        with torch.inference_mode():
            return self._evaluate(policy, max_steps)

    def _evaluate(self, policy, max_steps: int) -> dict:
        env, unwrapped, command = self.env, self.unwrapped, self.command
        saved = {
            "counter": unwrapped.common_step_counter,
            "first_frame": command.cfg.always_reset_to_first_frame,
            "voc": command.virtual_object_controller_scale_factor.clone(),
            "voc_reset": command.cfg.voc_reset_scale,
            "disabled": getattr(self.curriculum, "_disabled", None),
        }
        t0 = time.time()
        try:
            command.cfg.always_reset_to_first_frame = True
            command.virtual_object_controller_scale_factor = torch.zeros_like(saved["voc"])
            if self.curriculum is not None:
                self.curriculum._disabled = True  # keep the assist at 0 through resets
            return self._run(policy, max_steps, t0)
        finally:
            command.cfg.always_reset_to_first_frame = saved["first_frame"]
            command.virtual_object_controller_scale_factor = saved["voc"]
            command.cfg.voc_reset_scale = saved["voc_reset"]
            if self.curriculum is not None:
                self.curriculum._disabled = saved["disabled"]
            unwrapped.common_step_counter = saved["counter"]
            env.reset()

    def _run(self, policy, max_steps: int, t0: float) -> dict:
        env, unwrapped, command = self.env, self.unwrapped, self.command
        n, device = self.count, unwrapped.device
        obs, _ = env.reset()
        manager = unwrapped.termination_manager
        terms = [t for t in manager.active_terms]
        done = torch.zeros(n, dtype=torch.bool, device=device)
        survived = torch.zeros(n, dtype=torch.bool, device=device)
        length = torch.zeros(n, dtype=torch.long, device=device)
        reached = torch.zeros(n, dtype=torch.long, device=device)
        start = command.reset_timestep[:n].clone()
        end = command.trajectory_end_timestep[:n].clone()
        reasons = {t: 0 for t in terms}
        obj_err_sum = torch.zeros(n, device=device)
        z0 = command.object_position_e[:n, 0, 2].clone()
        z_trace = [z0.detach().cpu().numpy()]
        steps = 0
        while not bool(done.all()) and steps < max_steps:
            actions = policy(obs)
            before = command.timestep[:n].clone()
            obs, _, dones, extras = env.step(actions)
            steps += 1
            live = ~done
            d = dones[:n].bool()
            timeouts = extras.get("time_outs")
            timeouts = timeouts[:n].bool() if timeouts is not None else torch.zeros_like(d)
            newly = live & d
            survived |= newly & timeouts
            for t in terms:
                fired = manager.get_term(t)[:n].bool() & newly & ~timeouts
                reasons[t] += int(fired.sum())
            reached = torch.where(newly, before + 1, torch.where(live, command.timestep[:n], reached))
            length += live.long()
            err = torch.norm(
                command.object_position_e[:n] - command.object_body_position_command_e[:n], dim=-1
            ).max(dim=-1).values
            obj_err_sum += torch.where(live & ~d, err, torch.zeros_like(err))
            # Height of the episode still running (an env that finished is held at its last value).
            z_now = command.object_position_e[:n, 0, 2]
            z_trace.append(torch.where(live & ~d, z_now, torch.as_tensor(z_trace[-1], device=device)).detach().cpu().numpy())
            done |= d
        span = (end - start).clamp(min=1).float()
        completion = ((reached - start).clamp(min=0).float() / span).clamp(max=1.0)
        completion = torch.where(survived, torch.ones_like(completion), completion)
        result = {
            "episodes": n,
            "survival_rate": round(float(survived.float().mean()), 4),
            "completion_mean": round(float(completion.mean()), 4),
            "completion_p10": round(float(torch.quantile(completion, 0.1)), 4),
            "termination_counts": {k: v for k, v in reasons.items() if v},
            "object_pos_err_mean_m": round(float((obj_err_sum / length.clamp(min=1)).mean()), 4),
            "unfinished": int((~done).sum()),
            "steps": steps,
            "wall_s": round(time.time() - t0, 1),
        }
        if self.evaluator is not None:
            from groot_finetune.closed_loop.evaluation import evaluate_lift_hold

            traces = np.stack(z_trace, axis=1)  # (n, steps + 1)
            lift = [evaluate_lift_hold(trace, config=self.evaluator) for trace in traces]
            result["task_success_rate"] = round(float(np.mean([r.success for r in lift])), 4)
            result["lift_hold_rate"] = round(float(np.mean([r.lift_hold_success for r in lift])), 4)
            result["max_lift_m_mean"] = round(float(np.mean([r.max_lift_m for r in lift])), 4)
        return result


def _write_report(env, agent_cfg, t0, t_ready, reference, schedule, *, history, solved_at, best,
                  iteration, train_s) -> int:
    out_dir = os.path.dirname(os.path.abspath(args_cli.output))
    solved = solved_at is not None
    verdict = (
        f"SOLVED by budgeted RL at iteration {solved_at['iteration']}" if solved
        else f"NOT SOLVED within {args_cli.max_iterations} RL iterations"
    )
    t_done = time.time()
    fps = env.num_envs * agent_cfg.num_steps_per_env * iteration / max(train_s, 1e-6)
    report = {
        "schema": "v2d_solvability_v1",
        "task": args_cli.task,
        "motion_file": args_cli.motion_file,
        "solvable": solved,
        "verdict": verdict,
        "solved": solved_at,
        "best": best,
        "reference_task_evaluation": reference,
        "criteria": {
            "success_survival": args_cli.success_survival,
            "evaluation": "frame-0 reset, VOC assist 0, the recipe's terminations; a pass "
            "completes the whole reference",
        },
        "budget": {
            "num_envs": env.num_envs,
            "max_iterations": args_cli.max_iterations,
            "voc_anneal_iterations": args_cli.voc_anneal_iterations,
            "voc_schedule": schedule,
            "eval_interval": args_cli.eval_interval,
            "iterations_run": iteration,
        },
        "history": history,
        "timing_s": {
            "boot": round(t_ready - t0, 1),
            "train": round(train_s, 1),
            "eval": round(sum(h["wall_s"] for h in history), 1),
            "total": round(t_done - t0, 1),
        },
        "env_steps_per_s": round(fps) if iteration else None,
        "checkpoint": (
            (args_cli.checkpoint_out or os.path.join(out_dir, "policy.pt")) if solved_at else None
        ),
    }
    with open(args_cli.output, "w") as f:
        json.dump(report, f, indent=1)
    print("SOLVABILITY " + json.dumps(
        {k: report[k] for k in ("solvable", "verdict", "timing_s", "env_steps_per_s")}))
    env.close()
    return 0

def main() -> int:
    t0 = time.time()
    torch.manual_seed(args_cli.seed)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.motion_file = args_cli.motion_file
    apply_scene_config(env_cfg, SceneConfig.from_motion_file(env_cfg.motion_file))
    schedule = _compress_curriculum(env_cfg, args_cli.voc_anneal_iterations)

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args_cli.seed
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.logger = "tensorboard"
    agent_cfg.save_interval = 10**9  # checkpoints are written here, not by the runner
    if agent_cfg.num_steps_per_env != env_cfg.curriculum.voc_curriculum.params["num_steps_per_env"]:
        raise SystemExit("solvability: curriculum num_steps_per_env does not match the PPO cfg")

    out_dir = os.path.dirname(os.path.abspath(args_cli.output))
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    unwrapped = env.unwrapped
    command = unwrapped.command_manager.get_term("motion")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # The recipe's starting point: zero residual (pure reference tracking).
    torch.nn.init.zeros_(runner.alg.policy.actor[-1].weight)
    torch.nn.init.zeros_(runner.alg.policy.actor[-1].bias)
    t_ready = time.time()

    evaluator = _lift_hold(args_cli.task_profile)
    evaluate = FrameZeroEvaluation(env, command, evaluator, args_cli.eval_envs)
    # Reference evaluator at frame 0 (free): reset to frame 0 once and read the window.
    command.cfg.always_reset_to_first_frame = True
    with torch.inference_mode():
        env.reset()
    reference = _reference_evaluation(command, evaluator)
    # A frame-0 episode ends at the reference end (timestep_termination), not at
    # max_episode_length: frames + the reset freeze + a margin.
    freeze = max(int(command.cfg.reset_freeze_steps), int(getattr(command.cfg, "voc_decay_steps", 0)))
    horizon = int(command.trajectory_end_timestep[0] - command.reset_timestep[0]) + freeze + 20
    command.cfg.always_reset_to_first_frame = False
    with torch.inference_mode():
        env.reset()

    history = []
    print(
        f"SOLVABILITY start num_envs={env.num_envs} budget={args_cli.max_iterations} "
        f"voc_schedule={schedule} horizon={horizon} boot_s={t_ready - t0:.1f} reference={reference}"
    )
    if args_cli.expert_onnx:
        from groot_finetune.source_policy import OnnxSourcePolicy

        expert = OnnxSourcePolicy.from_checkpoint(args_cli.expert_onnx, device=unwrapped.device)
        result = evaluate(expert, horizon)
        print("SOLVABILITY_EXPERT " + json.dumps(result))
        with open(args_cli.output, "w") as f:
            json.dump({"schema": "v2d_solvability_expert_check_v1", "expert": args_cli.expert_onnx,
                       "reference_task_evaluation": reference, "evaluation": result}, f, indent=1)
        env.close()
        return 0
    # Iteration 0: the zero-residual policy (the physics replay's open loop, assist off).
    first = evaluate(runner.get_inference_policy(device=unwrapped.device), horizon)
    history.append({"iteration": 0, "train_s": 0.0, **first})
    print("SOLVABILITY_EVAL " + json.dumps(history[-1]))

    iteration, train_s, solved_at, best = 0, 0.0, None, history[-1]
    while iteration < args_cli.max_iterations:
        chunk = min(args_cli.eval_interval, args_cli.max_iterations - iteration)
        t = time.time()
        runner.learn(num_learning_iterations=chunk, init_at_random_ep_len=iteration == 0)
        train_s += time.time() - t
        iteration += chunk
        if iteration < args_cli.voc_anneal_iterations and not args_cli.eval_before_anneal:
            print(f"SOLVABILITY_TRAIN iteration={iteration} train_s={train_s:.0f} (assist still on)")
            continue
        result = evaluate(runner.get_inference_policy(device=unwrapped.device), horizon)
        assist = float(command.virtual_object_controller_scale_factor.reshape(-1)[0])
        history.append({"iteration": iteration, "train_s": round(train_s, 1), "train_assist": assist, **result})
        print("SOLVABILITY_EVAL " + json.dumps(history[-1]))
        if result["survival_rate"] > best["survival_rate"]:
            best = history[-1]
        if (
            solved_at is None
            and assist == 0.0
            and result["survival_rate"] >= args_cli.success_survival
        ):
            solved_at = history[-1]
            checkpoint = args_cli.checkpoint_out or os.path.join(out_dir, "policy.pt")
            if checkpoint != "none":
                runner.save(checkpoint)
            if not args_cli.no_stop:
                break

    return _write_report(
        env, agent_cfg, t0, t_ready, reference, schedule, history=history, solved_at=solved_at,
        best=best, iteration=iteration, train_s=train_s,
    )


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
