# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mesh-scale search followed by video tracking in one process (reconstruct --fast).

Same two functions, same arguments and same files as the separate
run_estimate_mesh_scale and run_video_to_poses calls; tracking reads the rescaled mesh
the scale search just wrote. One process saves one container round trip and one
FoundationPose import (~15 s). FoundationPose re-seeds every RNG at each register(), so
the tracking pass sees the same random state it would in a fresh process.
"""

from __future__ import annotations

import argparse

from v2d.foundation_pose.lib.run_estimate_mesh_scale import run_estimate_mesh_scale
from v2d.foundation_pose.lib.run_video_to_poses import run_video_to_poses


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    # scale search (run_estimate_mesh_scale)
    p.add_argument("--mesh_path", required=True)
    p.add_argument("--rgb_path", required=True)
    p.add_argument("--depth_path", required=True)
    p.add_argument("--mask_path", required=True)
    p.add_argument("--intrinsics_path", required=True)
    p.add_argument("--weights_dir", required=True)
    p.add_argument("--scale_path", required=True)
    p.add_argument("--rescaled_mesh_path", required=True)
    p.add_argument("--backend", default="nvidia_tensorrt")
    p.add_argument("--lo", type=float, default=0.5)
    p.add_argument("--hi", type=float, default=2.0)
    p.add_argument("--n_samples", type=int, default=7)
    p.add_argument("--n_levels", type=int, default=3)
    p.add_argument("--iou_weight", type=float, default=1.0)
    p.add_argument("--depth_weight", type=float, default=1.0)
    p.add_argument("--chamfer_weight", type=float, default=0.0)
    p.add_argument("--registration_iterations", type=int, default=5)
    # tracking (run_video_to_poses)
    p.add_argument("--video_path", required=True)
    p.add_argument("--depth_folder", required=True)
    p.add_argument("--masks_folder", required=True)
    p.add_argument("--camera_intrinsics_path", required=True)
    p.add_argument("--poses_dir", required=True)
    p.add_argument("--reference_frame", type=int, default=0)
    p.add_argument("--reregister_iou_thresh", type=float, default=None)
    p.add_argument("--mask_depth", action="store_true")
    p.add_argument("--target_width", type=int, default=None)
    p.add_argument("--target_height", type=int, default=None)
    p.add_argument("--register_iteration", type=int, default=10)
    p.add_argument("--track_iteration", type=int, default=5)
    p.add_argument("--n_particles", type=int, default=1)
    p.add_argument("--particle_process_noise_t", type=float, default=0.005)
    p.add_argument("--particle_process_noise_r", type=float, default=0.02)
    p.add_argument("--particle_iteration", type=int, default=3)
    p.add_argument("--particle_mask_iou_weight", type=float, default=1.0)
    a = p.parse_args()

    run_estimate_mesh_scale(
        mesh_path=a.mesh_path, rgb_path=a.rgb_path, depth_path=a.depth_path, mask_path=a.mask_path,
        intrinsics_path=a.intrinsics_path, weights_dir=a.weights_dir, scale_path=a.scale_path,
        backend=a.backend, rescaled_mesh_path=a.rescaled_mesh_path,
        lo=a.lo, hi=a.hi, n_samples=a.n_samples, n_levels=a.n_levels,
        iou_weight=a.iou_weight, depth_weight=a.depth_weight, chamfer_weight=a.chamfer_weight,
        registration_iterations=a.registration_iterations,
    )
    run_video_to_poses(
        a.video_path, a.depth_folder, a.masks_folder, a.camera_intrinsics_path,
        a.rescaled_mesh_path, a.poses_dir, a.weights_dir,
        backend=a.backend, reference_frame=a.reference_frame,
        target_width=a.target_width, target_height=a.target_height,
        reregister_iou_thresh=a.reregister_iou_thresh,
        register_iteration=a.register_iteration, track_iteration=a.track_iteration,
        n_particles=a.n_particles, particle_process_noise_t=a.particle_process_noise_t,
        particle_process_noise_r=a.particle_process_noise_r, particle_iteration=a.particle_iteration,
        particle_mask_iou_weight=a.particle_mask_iou_weight, mask_depth=a.mask_depth,
    )


if __name__ == "__main__":
    main()
