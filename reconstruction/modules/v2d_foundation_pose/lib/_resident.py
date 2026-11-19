"""Resident preload for v2d_foundation_pose (see v2d_common/resident.py).

Builds the scorer/refiner predictor backend (TensorRT engines on the GPU) with the
call's own --backend/--weights_dir; FoundationPoseTracker takes it instead of building it.
"""

from __future__ import annotations

ENTRIES = {
    "v2d.foundation_pose.lib.run_estimate_mesh_scale",
    "v2d.foundation_pose.lib.run_video_to_poses",
}


def _arg(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None


def preload(module: str, argv: list[str]) -> None:
    if module not in ENTRIES:
        return
    weights_dir = _arg(argv, "--weights_dir")
    if not weights_dir:
        return
    from v2d.common import resident
    from v2d.foundation_pose.lib.backends import DEFAULT_BACKEND, create_predictor_backend
    from v2d.foundation_pose.lib.foundation_pose_tracker import predictor_backend_key

    backend = _arg(argv, "--backend") or DEFAULT_BACKEND
    resident.put(predictor_backend_key(backend, weights_dir), create_predictor_backend(backend, weights_dir))
