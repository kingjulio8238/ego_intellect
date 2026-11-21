"""Second physics check: the physics-QA replay, run in MuJoCo on the package's MJCF scene.

The same experiment as robotic_grounding's replay_metrics.py in Isaac Lab: start at reference
frame 0, drive every joint to the reference (zero residual action; the arm's command delay
applied), no virtual object assist, and record per control step the object's position and
orientation error and each wrist's position error against the reference, stopping at the
first termination the task spec defines. The checks and flag thresholds are the Isaac
replay's own (from the package), so the two verdicts compare one to one.

One difference: Isaac's ``grasp_missed_contact`` is the fraction of expected contact-wrench
directions the simulation does not support; MuJoCo here reports the fraction of hands whose
reference expects contact but that touch the object nowhere (a contact-presence version of
the same check). The kinematic grasp-drift check does not depend on the simulator and is
taken from the package.

    python3 -m tools.envpkg.mujoco_replay <package> [--robot vega_sharpa] [--out report.json]

Contract: docs/pipeline/MUJOCO_EXPORT.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

SETTLE_WINDOW_S = 0.5  # as in replay_metrics.py
GRASP_WINDOW_S = 1.0
# MuJoCo-only check: the object's start pose must not sit inside the support or the hands.
# Isaac (PhysX) pops an interpenetrating object out, which its settle check then catches;
# MuJoCo's soft contacts push it out gently, so the overlap is measured directly instead.
# Measured: 0.0 mm on good clips, 20 mm with the object lowered 2 cm, 27 mm shifted 5 cm.
INITIAL_PENETRATION_MAX_M = 0.01
# Cross-simulator agreement (docs/pipeline/MUJOCO_EXPORT.md): same verdict, completion and each
# position check within these of Isaac's. Measured on b3_tbseed / b3_tb2: completion 0.008-0.011,
# checks <= 9 mm (settle; MuJoCo settles the object a little more on its support).
AGREE_COMPLETION = 0.05
AGREE_POSITION_M = 0.015
POSITION_CHECKS = ("settle_object_pos_max_m", "rest_object_pos_p95_m", "grasp_object_pos_p95_m", "wrist_pos_p95_m")


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    if np.dot(q0, q1) < 0:
        q1 = -q1
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
    else:
        theta = np.arccos(dot)
        q = (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)
    return q / np.linalg.norm(q)


def _quat_angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(2.0 * np.arccos(np.clip(abs(np.dot(a, b)), -1.0, 1.0)))


def reference(motion: dict[str, Any], *, control_dt: float, speed: float, steps: int) -> dict[str, np.ndarray]:
    """The reference at each control step (linear in position, slerp in orientation)."""
    fps = float(motion["fps"])
    q = np.asarray(motion["robot_joint_positions"], dtype=float)
    obj_p = np.asarray(motion["object_root_position"], dtype=float)
    obj_q = np.asarray(motion["object_body_wxyz"], dtype=float)[:, 0]
    ee = np.asarray(motion["ee_pose_w"], dtype=float)[:, :, :3]
    contact = np.asarray(motion["hand_contact_active"], dtype=float).T  # (T, 2) left, right
    out = {k: [] for k in ("q", "obj_p", "obj_q", "ee", "contact")}
    for k in range(steps):
        f = min(k * control_dt * speed * fps, len(q) - 1)
        i = int(np.floor(f))
        j, t = min(i + 1, len(q) - 1), f - i
        out["q"].append((1 - t) * q[i] + t * q[j])
        out["obj_p"].append((1 - t) * obj_p[i] + t * obj_p[j])
        out["obj_q"].append(_slerp(obj_q[i], obj_q[j], t))
        out["ee"].append((1 - t) * ee[i] + t * ee[j])
        out["contact"].append(contact[int(round(f))])
    return {k: np.asarray(v) for k, v in out.items()}


def _grasp_drift(motion: dict[str, Any], isaac: dict[str, Any]) -> float:
    """robotic_grounding's kinematic grasp-drift check on this motion (the Isaac replay uses the
    same function); the package's value when the repository is not at hand."""
    from .common import REPO_ROOT

    scripts = REPO_ROOT / "robotic_grounding/scripts"
    if not (scripts / "data_quality_checks/grasp_drift.py").is_file():
        return isaac["checks"]["grasp_drift_p95_m"]
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from data_quality_checks.grasp_drift import grasp_drift

    return grasp_drift({k: [v] for k, v in motion.items()})["p95_m"]


def agreement(mujoco_report: dict[str, Any], isaac_report: dict[str, Any]) -> dict[str, Any]:
    """Do the two simulators reach the same physics-QA verdict, within the measured limits?"""
    diffs = {"flag": [mujoco_report["flag"], isaac_report["flag"]],
             "completion": round(abs(mujoco_report["completion"] - isaac_report["completion"]), 4)}
    reasons = []
    if mujoco_report["flag"] != isaac_report["flag"]:
        reasons.append(f"verdicts differ: MuJoCo flag {mujoco_report['flag']}, Isaac flag {isaac_report['flag']}")
    if diffs["completion"] > AGREE_COMPLETION:
        reasons.append(f"completion differs by {diffs['completion']} (limit {AGREE_COMPLETION})")
    for name in POSITION_CHECKS:
        a, b = mujoco_report["checks"].get(name), isaac_report["checks"].get(name)
        if a is None or b is None:
            continue
        diffs[name] = round(abs(a - b), 5)
        if diffs[name] > AGREE_POSITION_M:
            reasons.append(f"{name} differs by {diffs[name]} m (limit {AGREE_POSITION_M})")
    return {"agree": not reasons, "reasons": reasons, "differences": diffs,
            "limits": {"completion": AGREE_COMPLETION, "position_m": AGREE_POSITION_M}}


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    a = np.asarray(values, dtype=float)
    return {"n": int(a.size), "mean": round(float(a.mean()), 5), "p95": round(float(np.percentile(a, 95)), 5),
            "max": round(float(a.max()), 5), "first": round(float(a[0]), 5)}


def replay(package: Path, robot: str, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    import mujoco
    import pyarrow.parquet as pq

    t0 = time.time()
    manifest = manifest or json.loads((package / "manifest.json").read_text())
    env = json.loads((package / "mujoco/env.json").read_text())["robots"][robot]
    task = json.loads((package / manifest["task"]["spec"]).read_text())["robots"][robot]
    isaac = json.loads((package / manifest["robots"][robot]["qa"]["physics_replay"]).read_text())
    motion = pq.read_table(package / manifest["robots"][robot]["motion"]["path"]).to_pylist()[0]
    model = mujoco.MjModel.from_xml_path(str(package / env["scene"]))
    data = mujoco.MjData(model)

    ctrl = env["control"]
    steps = int(task["reference"]["control_steps"])
    ref = reference(motion, control_dt=ctrl["control_dt_s"], speed=ctrl["playback_speed"], steps=steps)
    terms = {t["name"]: t for t in task["terminations"]}
    limits = {
        "object_pos": terms.get("object_pos_error", {}).get("params", {}).get("threshold", 0.10),
        "object_rot": terms.get("object_quat_error", {}).get("params", {}).get("threshold", 0.50),
        "wrist": terms.get("hand_wrist_away", {}).get("params", {}).get("threshold", 0.15),
    }

    joints = [j["joint"] for j in env["joints"]]
    qadr = np.array([model.jnt_qposadr[model.joint(j).id] for j in joints])
    act = np.array([model.actuator(j).id for j in joints])
    delays = np.array([j["delay_physics_steps"] for j in env["joints"]])
    obj_body = model.body("object").id
    obj_qadr = model.jnt_qposadr[model.joint("object_free").id]
    obj_geom = model.geom(env["object"]["geom"]).id
    ee_ids = [model.body(b).id for b in env["ee_bodies"]]
    hand_bodies = {side: {model.body(b).id for b in names} for side, names in env["hand_contact_bodies"].items()}

    data.qpos[qadr] = ref["q"][0]
    data.qpos[obj_qadr:obj_qadr + 3] = ref["obj_p"][0]
    data.qpos[obj_qadr + 3:obj_qadr + 7] = ref["obj_q"][0]
    mujoco.mj_forward(model, data)
    overlaps = [-c.dist for c in data.contact[: data.ncon] if obj_geom in (c.geom1, c.geom2)]
    initial_penetration = round(max([0.0, *overlaps]), 5)
    history: deque[np.ndarray] = deque([ref["q"][0].copy()] * (int(delays.max()) + 1), maxlen=int(delays.max()) + 1)

    def errors(k: int) -> tuple[float, float, list[float]]:
        p = float(np.linalg.norm(data.xpos[obj_body] - ref["obj_p"][k]))
        r = _quat_angle(data.xquat[obj_body], ref["obj_q"][k])
        w = [float(np.linalg.norm(data.xpos[b] - ref["ee"][k][i])) for i, b in enumerate(ee_ids)]
        return p, r, w

    def hand_touch() -> dict[str, bool]:
        touch = {side: False for side in hand_bodies}
        for c in data.contact[: data.ncon]:
            if obj_geom not in (c.geom1, c.geom2):
                continue
            other = model.geom_bodyid[c.geom2 if c.geom1 == obj_geom else c.geom1]
            for side, ids in hand_bodies.items():
                touch[side] |= int(other) in ids
        return touch

    object_pos, object_rot, wrists, missed, contact_ref = [], [], [], [], []

    def record(k: int) -> None:
        p, r, w = errors(k)
        object_pos.append(p)
        object_rot.append(r)
        wrists.append(w)
        labels = ref["contact"][k] > 0.5
        touch = hand_touch()
        active = [side for side, on in zip(("left", "right"), labels) if on]
        missed.append(float(np.mean([not touch[s] for s in active])) if active else 0.0)
        contact_ref.append([int(v) for v in labels])

    record(0)
    reached, ended, reasons = 0, "timeout", []
    for k in range(steps - 1):
        target = ref["q"][k]
        for _ in range(int(ctrl["decimation"])):
            history.append(target)
            delayed = np.array([history[-1 - d][i] for i, d in enumerate(delays)])
            data.ctrl[act] = delayed
            mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            ended, reasons = "terminated", ["robot_state_diverged"]
            break
        p, r, w = errors(k + 1)
        hit = [name for name, bad in (("object_pos_error", p > limits["object_pos"]),
                                      ("object_quat_error", r > limits["object_rot"]),
                                      ("hand_wrist_away", max(w) > limits["wrist"])) if bad]
        reached = k + 1
        if hit:
            ended, reasons = "terminated", hit
            break
        record(k + 1)

    end = steps - 1
    completion = 1.0 if ended == "timeout" else min(reached, end) / max(end, 1)
    dt = ctrl["control_dt_s"]
    in_contact = [max(c) > 0 for c in contact_ref]
    first = in_contact.index(True) if any(in_contact) else None
    grasp_steps = max(1, round(GRASP_WINDOW_S / dt))
    settle = object_pos[: max(1, round(SETTLE_WINDOW_S / dt)) + 1]
    rest = object_pos[: first if first is not None else len(object_pos)]
    grasp = [i for i in range(first, len(object_pos)) if in_contact[i]][:grasp_steps] if first is not None else []
    wrist_arr = np.asarray(wrists)
    checks = {
        "settle_object_pos_max_m": _summary(settle).get("max"),
        "rest_object_pos_p95_m": _summary(rest).get("p95"),
        "grasp_missed_contact_mean": _summary([missed[i] for i in grasp]).get("mean"),
        "grasp_object_pos_p95_m": _summary([object_pos[i] for i in grasp]).get("p95"),
        "wrist_pos_p95_m": max(_summary(wrist_arr[:, i].tolist())["p95"] for i in range(wrist_arr.shape[1])),
        "grasp_drift_p95_m": _grasp_drift(motion, isaac),  # kinematic: simulator-independent
        "initial_penetration_m": initial_penetration,
    }
    thresholds = {**isaac["thresholds"], "initial_penetration_m": INITIAL_PENETRATION_MAX_M}
    flags = [f"{n} {v:.4f} (limit {thresholds[n]})" for n, v in checks.items() if v is not None and v > thresholds[n]]
    if first is None:
        flags.append(f"replay ended ({ended}: {', '.join(reasons) or 'none'}) before the first reference contact, "
                     f"{len(object_pos) - 1} steps in")
    elif len(grasp) < grasp_steps:
        flags.append(f"object lost {len(grasp)} steps into contact (window {grasp_steps})")
    return {
        "schema": "mujoco_replay_metrics/v1",
        "robot": robot,
        "scene": env["scene"],
        "mujoco_version": mujoco.__version__,
        "steps": len(object_pos) - 1,
        "reached_timestep": reached,
        "end_timestep": end,
        "ended": ended,
        "termination_terms": reasons,
        "completion": round(completion, 4),
        "first_contact_step": first,
        "checks": checks,
        "thresholds": thresholds,
        "flag": bool(flags),
        "flag_reasons": flags,
        "series": {"object_pos_m": [round(v, 4) for v in object_pos], "object_rot_rad": [round(v, 4) for v in object_rot],
                   "wrist_pos_m": [[round(v, 4) for v in w] for w in wrists], "missed_contact": [round(v, 3) for v in missed]},
        "seconds": round(time.time() - t0, 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("package", type=Path)
    parser.add_argument("--robot", default="vega_sharpa")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", action="store_true",
                        help="compare with the package's recorded MuJoCo replay (exit 1 on a difference)")
    args = parser.parse_args(argv)
    report = replay(args.package, args.robot)
    if args.check:
        manifest = json.loads((args.package / "manifest.json").read_text())
        recorded = json.loads((args.package / manifest["robots"][args.robot]["qa"]["mujoco"]["report"]).read_text())
        diffs = [f"flag {report['flag']} != {recorded['flag']}"] if report["flag"] != recorded["flag"] else []
        if abs(report["completion"] - recorded["completion"]) > 0.01:
            diffs.append(f"completion {report['completion']} != {recorded['completion']}")
        for name, value in recorded["checks"].items():
            got = report["checks"].get(name)
            if (got is None) != (value is None) or (got is not None and abs(got - value) > 0.002):
                diffs.append(f"{name} {got} != {value}")
        print("MUJOCO_CHECK " + ("OK" if not diffs else "FAILED " + "; ".join(diffs)))
        return 1 if diffs else 0
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("flag", "flag_reasons", "completion", "ended", "termination_terms",
                                             "checks", "seconds")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
