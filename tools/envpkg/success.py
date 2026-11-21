"""Success criteria of task_spec/v1, self-contained so a lab can score rollouts without our repo.

``lift_hold`` reproduces groot_finetune.closed_loop.evaluation.evaluate_lift_hold exactly
(tools/envpkg/tests checks the two agree); keep them in step.
"""

from __future__ import annotations

from typing import Any

import numpy as np

LIFT_HOLD_DEFAULTS = {
    "final_min_lift_m": -0.01,
    "placement_threshold_m": 0.03,
    "placement_window_steps": 20,
    "max_descent_step_m": 0.01,
    "max_terminal_step_m": 0.005,
    "max_terminal_range_m": 0.01,
}


def lift_hold(object_z: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
    """Score one episode's object-height trace (one sample per control step, the first being the
    episode start). Returns success plus the quantities it is decided on."""
    p = {**LIFT_HOLD_DEFAULTS, **params}
    trace = np.asarray(object_z, dtype=np.float64)
    if trace.ndim != 1 or trace.size == 0 or not np.isfinite(trace).all():
        raise ValueError("object_z must be a non-empty, finite 1-D array")
    lift = trace[1:] - trace[0]
    max_lift = latest = max_descent = 0.0
    run = max_hold = 0
    for value in lift:
        max_lift = max(max_lift, value)
        if max_lift >= p["hold_threshold_m"]:
            max_descent = max(max_descent, latest - value)
        run = run + 1 if value >= p["hold_threshold_m"] else 0
        max_hold = max(max_hold, run)
        latest = value
    window = int(p["placement_window_steps"])
    terminal = lift[-window:]
    lift_hold_success = max_lift >= p["lift_threshold_m"] and max_hold >= p["min_hold_steps"]
    stable_placement = bool(
        lift.size >= window
        and terminal.max() <= p["placement_threshold_m"]
        and np.abs(np.diff(terminal)).max() <= p["max_terminal_step_m"]
        and np.ptp(terminal) <= p["max_terminal_range_m"]
    )
    success = bool(lift_hold_success and latest >= p["final_min_lift_m"] and stable_placement
                   and max_descent <= p["max_descent_step_m"])
    return {
        "success": success,
        "lift_hold_success": bool(lift_hold_success),
        "stable_placement": stable_placement,
        "max_lift_m": round(float(max_lift), 4),
        "max_hold_steps": int(max_hold),
        "final_lift_m": round(float(latest), 4),
        "max_descent_m": round(float(max_descent), 4),
        "steps": int(trace.size),
    }


EVALUATORS = {"lift_hold": lift_hold}
