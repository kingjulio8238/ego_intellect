# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Docker wrapper for lib/run_scale_then_track.py (scale search + tracking, one container)."""

import os

from v2d.docker.container import run_in_container
from v2d.foundation_pose.docker._config import DEV_PRESERVE_VOLUMES, IMAGE_NAME, MODULES_DIR


def run_scale_then_track(
    mesh_path: str,
    rgb_path: str,
    depth_path: str,
    mask_path: str,
    intrinsics_path: str,
    weights_dir: str,
    scale_path: str,
    rescaled_mesh_path: str,
    video_path: str,
    depth_folder: str,
    masks_folder: str,
    camera_intrinsics_path: str,
    poses_dir: str,
    backend: str = "nvidia_tensorrt",
    lo: float = 0.5,
    hi: float = 2.0,
    n_samples: int = 7,
    n_levels: int = 3,
    iou_weight: float = 1.0,
    depth_weight: float = 1.0,
    chamfer_weight: float = 0.0,
    registration_iterations: int = 5,
    reference_frame: int = 0,
    reregister_iou_thresh: float | None = None,
    mask_depth: bool = False,
    dev: bool = False,
) -> None:
    weights_container = f"/data/weights_dir/{os.path.basename(os.path.abspath(weights_dir))}"
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.foundation_pose.lib.run_scale_then_track",
        inputs={
            "mesh_path": mesh_path,
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "mask_path": mask_path,
            "intrinsics_path": intrinsics_path,
            "weights_dir": weights_dir,
            "video_path": video_path,
            "depth_folder": depth_folder,
            "masks_folder": masks_folder,
            "camera_intrinsics_path": camera_intrinsics_path,
        },
        outputs={"scale_path": scale_path, "rescaled_mesh_path": rescaled_mesh_path, "poses_dir": poses_dir},
        extra_args={
            "backend": backend,
            "lo": lo,
            "hi": hi,
            "n_samples": n_samples,
            "n_levels": n_levels,
            "iou_weight": iou_weight,
            "depth_weight": depth_weight,
            "chamfer_weight": chamfer_weight,
            "registration_iterations": registration_iterations,
            "reference_frame": reference_frame,
            "reregister_iou_thresh": reregister_iou_thresh,
            "mask_depth": mask_depth,
        },
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
        env={
            "FOUNDATIONPOSE_WEIGHTS_DIR": weights_container,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
        extra_volumes=DEV_PRESERVE_VOLUMES if dev else None,
    )
