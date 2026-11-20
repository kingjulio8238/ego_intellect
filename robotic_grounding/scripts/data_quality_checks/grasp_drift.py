# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject sequences whose hands slide over the object while the reference says they hold it.

During a rigid grasp each wrist is fixed in the object's frame. For every contact segment
(``hand_contact_active``) this takes the wrist position in the object frame and measures how
far it strays from its median over the segment (the median, not the first frame: contact
onset is where reconstructions are noisiest, so a first-frame reference charges a good clip
for its first frame's error). A large drift means
the hand and object trajectories disagree in time or space (e.g. the object lags the hands),
which a reviewer sees in the environment video as markers that do not move together.

score = 95th-percentile drift in metres over both hands' contact frames.
Calibrated 2026-11-20 on three clips judged good on review (tissue_box_simple 0.015 m,
fast-pipeline tissue_box b3_tbseed 0.036 m, b3_tb2 0.037 m) and the object played 0.5 s
behind the hands (0.046 m). The margins are thin (about 0.4 cm on each side); see
docs/pipeline/OPTIMIZATION.md in the monorepo.
"""

from __future__ import annotations

import numpy as np

THRESHOLD_M = 0.041


def _rotate_inverse(wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v (N, 3) by the inverse of unit quaternions wxyz (N, 4)."""
    w, xyz = wxyz[:, :1], -wxyz[:, 1:]  # conjugate
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def grasp_drift(data: dict) -> dict:
    """Per-hand and overall wrist-in-object drift over reference contact frames (metres)."""
    ee = np.asarray(data["ee_pose_w"][0], dtype=np.float64)  # (T, hands, 7): xyz + wxyz
    obj_pos = np.asarray(data["object_body_position"][0], dtype=np.float64)[:, 0]
    obj_wxyz = np.asarray(data["object_body_wxyz"][0], dtype=np.float64)[:, 0]
    obj_wxyz = obj_wxyz / np.linalg.norm(obj_wxyz, axis=1, keepdims=True)
    contact = data.get("hand_contact_active", [None])[0] or []
    drifts: list[float] = []
    per_hand = []
    for side, active in enumerate(contact[: ee.shape[1]]):
        frames = np.flatnonzero(np.asarray(active, dtype=np.float64) > 0.5)
        side_drift: list[float] = []
        if frames.size:
            local = _rotate_inverse(obj_wxyz, ee[:, side, :3] - obj_pos)
            for segment in np.split(frames, np.flatnonzero(np.diff(frames) > 1) + 1):
                center = np.median(local[segment], axis=0)
                side_drift += np.linalg.norm(local[segment] - center, axis=1).tolist()
        per_hand.append(round(float(np.percentile(side_drift, 95)), 4) if side_drift else None)
        drifts += side_drift
    p95 = float(np.percentile(drifts, 95)) if drifts else None
    return {"p95_m": None if p95 is None else round(p95, 4), "per_hand_p95_m": per_hand,
            "contact_frames": len(drifts)}


def check(data: dict, threshold: float = THRESHOLD_M) -> dict:
    """Pass when the 95th-percentile wrist drift in the object frame is at most ``threshold``."""
    result = grasp_drift(data)
    if result["p95_m"] is None:
        return {"pass": True, "score": 0.0, "reason": "no reference contact frames"}
    ok = result["p95_m"] <= threshold
    return {
        "pass": ok,
        "score": result["p95_m"],
        "reason": "" if ok else f"wrist drifts {result['p95_m']:.3f} m in the object frame "
        f"during contact (limit {threshold} m)",
    }
