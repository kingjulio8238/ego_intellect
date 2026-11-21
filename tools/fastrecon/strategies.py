"""Smoke catalog (record: docs/pipeline/OPTIMIZATION.md). One entry = one isolated call replay.

patch = {"set": {flag: value | True | None(remove)}, "add": [...], "env": {...}}
tag: a = output-identical by construction, b = numerically equivalent expected,
     c = changes the algorithm (kept only if inside the call's noise floor).
time_factor only feeds the cost projection (expected exec time vs production).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Strategy:
    name: str
    image: str
    module: str  # suffix of the `python -m` module, e.g. "lib.refine_simple"
    nth: int = 0  # which recorded call of that module (0 = first)
    patch: dict = field(default_factory=dict)
    gpu: str | None = "L40S"
    tag: str = "a"
    time_factor: float = 1.0
    note: str = ""


GS = ("v2d_gsplat_refinement", "lib.refine_simple")
FUSED = {"add": ["--fused_checks"]}
NODIAG = {"set": {"--render_every": "0", "--overlay_path": None}}
FUSED_NODIAG = {"set": {"--render_every": "0", "--overlay_path": None}, "add": ["--fused_checks"]}

FP_ENV = {"V2D_FP_FAST_DIAMETER": "1", "V2D_FP_QUIET_LOGS": "1", "V2D_FP_SEQUENTIAL_FRAMES": "1"}
GS_FAST = {**FUSED_NODIAG, "env": {"V2D_GSPLAT_SHARED_MANO": "1", "V2D_GSPLAT_VGG_CACHE": "1", "V2D_GSPLAT_MICRO": "1",
                                   "V2D_GSPLAT_COMPILE": "1", "V2D_GSPLAT_LOAD_WORKERS": "8"}}

STRATEGIES: list[Strategy] = [
    # A. remove work
    Strategy("A2-wilor-ref-frame-only", "v2d_wilor", "lib.video_to_hands",
             patch={"add": ["--frame_indices", "0"]}, tag="a", time_factor=0.2),
    Strategy("A1-gsplat-no-diagnostics", *GS, patch=NODIAG, tag="a", time_factor=0.8),
    Strategy("A1-sam3d-no-texture-bake", "v2d_sam3d", "lib.image_to_mesh",
             patch={"set": {"--with_texture_baking": None}}, tag="c", time_factor=0.8,
             note="texture only colours the viewer mesh; geometry must match"),
    # C. executor: these calls barely use the GPU -> CPU containers
    Strategy("C-interp-on-cpu", "v2d_wilor", "lib.tracks_interpolate", gpu=None, tag="a", time_factor=0.1),
    Strategy("C-align-hands-on-cpu", "v2d_hamer", "lib.align_hands", gpu=None, tag="b", time_factor=2.0),
    Strategy("C-gdino-on-cpu", "v2d_grounding_dino", "", gpu=None, tag="b", time_factor=2.0),
    Strategy("C-geocalib-on-cpu", "v2d_geocalib", "", gpu=None, tag="b", time_factor=3.0),
    Strategy("C-anycalib-on-cpu", "v2d_anycalib", "", gpu=None, tag="b", time_factor=3.0),
    # D. per-call compute
    Strategy("D1-gsplat-fused-checks", *GS, patch=FUSED, tag="a", time_factor=0.8),
    Strategy("D1-gsplat-fused+nodiag", *GS, patch=FUSED_NODIAG, tag="a", time_factor=0.7),
    Strategy("D1-gsplat-fused+nodiag+tf32", *GS,
             patch={**FUSED_NODIAG, "env": {"TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1", "NVIDIA_TF32_OVERRIDE": "1"}},
             tag="b", time_factor=0.7),
    Strategy("D5-moge-batch16", "v2d_moge", "lib.video_to_depth", patch={"set": {"--batch_size": "16"}}, tag="b"),
    Strategy("D5-moge-batch32", "v2d_moge", "lib.video_to_depth", patch={"set": {"--batch_size": "32"}}, tag="b"),
    Strategy("D5-moge-pipelined-io", "v2d_moge", "lib.video_to_depth",
             patch={"env": {"V2D_MOGE_PIPELINE": "1", "V2D_MOGE_WRITERS": "8"}}, tag="a", time_factor=0.6,
             note="decode thread + writer pool around unchanged inference"),
    Strategy("D5-moge-pipelined-io-H100", "v2d_moge", "lib.video_to_depth",
             patch={"env": {"V2D_MOGE_PIPELINE": "1", "V2D_MOGE_WRITERS": "8"}}, gpu="H100", tag="b", time_factor=0.5),
    Strategy("D9-sam3d-seed0", "v2d_sam3d", "lib.image_to_mesh", patch={"set": {"--seed": "0"}}, tag="c",
             note="pins the generator: makes the call reproducible"),
    Strategy("D8-fp-track-sequential-frames", "v2d_foundation_pose", "video_to_poses",
             patch={"env": {"V2D_FP_SEQUENTIAL_FRAMES": "1"}}, tag="a", time_factor=0.7,
             note="profile: cap.set() seek per frame = 20.8 s of 77 s"),
    Strategy("D7-fp-scale-hull-diameter+quiet-logs", "v2d_foundation_pose", "estimate_mesh_scale",
             patch={"env": {"V2D_FP_FAST_DIAMETER": "1", "V2D_FP_QUIET_LOGS": "1"}}, tag="a", time_factor=0.5,
             note="profile: compute_mesh_diameter 58.6 s + tensor log formatting 11 s of 134 s"),
    Strategy("D12-sam3d-prebuilt-gsplat", "v2d_sam3d", "lib.image_to_mesh", patch={}, tag="a", time_factor=0.6,
             note="image now ships the compiled gsplat extension (profile: 65 s JIT per call)"),
    Strategy("D12b-sam3d-prebuilt-gsplat-seed0", "v2d_sam3d", "lib.image_to_mesh", patch={"set": {"--seed": "0"}},
             tag="a", time_factor=0.6, note="seeded: must reproduce D9-sam3d-seed0's diff exactly"),
    Strategy("M1-fp-scale+track-one-call", "v2d_foundation_pose", "estimate_mesh_scale",
             patch={"merge": {"image": "v2d_foundation_pose", "module": "video_to_poses",
                              "new_module": "v2d.foundation_pose.lib.run_scale_then_track",
                              "drop_second": ["--mesh_path"]},
                    "env": {"V2D_FP_FAST_DIAMETER": "1", "V2D_FP_QUIET_LOGS": "1", "V2D_FP_SEQUENTIAL_FRAMES": "1"}},
             tag="a", time_factor=0.6, note="vs D7 (73 s) + D8 (44 s) as separate calls"),
    # D11. hardware sweep (H100 vs L40S control)
    Strategy("D11-gsplat-fused+nodiag-H100", *GS, patch=FUSED_NODIAG, gpu="H100", tag="b", time_factor=0.5),
    Strategy("D11-moge-H100", "v2d_moge", "lib.video_to_depth", gpu="H100", tag="b", time_factor=0.7),
    Strategy("D11-sam3d-H100", "v2d_sam3d", "lib.image_to_mesh", gpu="H100", tag="b", time_factor=0.7),
    Strategy("D11-fp-scale-H100", "v2d_foundation_pose", "estimate_mesh_scale", gpu="H100", tag="b", time_factor=0.7),
    Strategy("D11-fp-track-H100", "v2d_foundation_pose", "video_to_poses", gpu="H100", tag="b", time_factor=0.8),
    Strategy("D11-sam2-H100", "v2d_sam2", "lib.video_to_masks", gpu="H100", tag="b", time_factor=0.7),
    Strategy("D11-hamer-H100", "v2d_hamer", "lib.masks_to_hands", gpu="H100", tag="b", time_factor=0.8),
    Strategy("D11-droid-H100", "v2d_droid_slam", "lib.video_to_slam", gpu="H100", tag="b", time_factor=0.8),
    Strategy("D11-wilor-H100", "v2d_wilor", "lib.video_to_hands", gpu="H100", tag="b", time_factor=0.8),
    # P. profiles (cProfile, host side): where the critical-path calls spend wall time
    Strategy("P-sam3d", "v2d_sam3d", "lib.image_to_mesh", patch={"profile": True}, tag="profile", time_factor=1.2),
    Strategy("P-fp-scale", "v2d_foundation_pose", "estimate_mesh_scale", patch={"profile": True}, tag="profile", time_factor=1.2),
    Strategy("P-fp-track", "v2d_foundation_pose", "video_to_poses", patch={"profile": True}, tag="profile", time_factor=1.2),
    Strategy("P-gsplat", *GS, patch={"profile": True, **FUSED_NODIAG}, tag="profile", time_factor=1.2),
    Strategy("P-hamer", "v2d_hamer", "lib.masks_to_hands", patch={"profile": True}, tag="profile", time_factor=1.2),
    # R2. gsplat step restructuring (round 2; all on top of fused checks + no diagnostics)
    Strategy("R2-gsplat-exact-caches", *GS,
             patch={**FUSED_NODIAG, "env": {"V2D_GSPLAT_VGG_CACHE": "1", "V2D_GSPLAT_MICRO": "1", "V2D_GSPLAT_LOAD_WORKERS": "8"}},
             tag="a", time_factor=0.7, note="VGG target features once per frame; constant label/index tensors"),
    Strategy("R2-gsplat-shared-mano", *GS,
             patch={**FUSED_NODIAG, "env": {"V2D_GSPLAT_SHARED_MANO": "1", "V2D_GSPLAT_LOAD_WORKERS": "8"}},
             tag="b", time_factor=0.6, note="2 batched MANO calls per step instead of ~22"),
    Strategy("R2-gsplat-all-exact+mano", *GS,
             patch={**FUSED_NODIAG, "env": {"V2D_GSPLAT_SHARED_MANO": "1", "V2D_GSPLAT_VGG_CACHE": "1",
                                            "V2D_GSPLAT_MICRO": "1", "V2D_GSPLAT_LOAD_WORKERS": "8"}},
             tag="b", time_factor=0.5),
    Strategy("R2-gsplat-all+compile", *GS,
             patch={**FUSED_NODIAG, "env": {"V2D_GSPLAT_SHARED_MANO": "1", "V2D_GSPLAT_VGG_CACHE": "1",
                                            "V2D_GSPLAT_MICRO": "1", "V2D_GSPLAT_COMPILE": "1",
                                            "V2D_GSPLAT_LOAD_WORKERS": "8"}},
             tag="b", time_factor=0.5),
    Strategy("R2-gsplat-all-exact+mano-repeat", *GS,
             patch={**FUSED_NODIAG, "env": {"V2D_GSPLAT_SHARED_MANO": "1", "V2D_GSPLAT_VGG_CACHE": "1",
                                            "V2D_GSPLAT_MICRO": "1", "V2D_GSPLAT_LOAD_WORKERS": "8"}},
             tag="b", time_factor=0.5, note="repeat: replay timing varies +-80 s"),
    # R3. resident execution (tools/modal_v2d/zygote.py): imports + model preloaded before the
    # call, untimed (in a run it overlaps upstream work). Each pair = same fast patch with and
    # without residency, so the time difference is exactly what residency removes.
    Strategy("R3-sam3d-fast", "v2d_sam3d", "lib.image_to_mesh", patch={}, tag="a", time_factor=1.0),
    Strategy("R3-sam3d-fast-resident", "v2d_sam3d", "lib.image_to_mesh", patch={**{}, "resident": True}, tag="a", time_factor=1.0),
    Strategy("R3-fp-scale-fast", "v2d_foundation_pose", "estimate_mesh_scale", patch={"env": FP_ENV}, tag="a", time_factor=1.0),
    Strategy("R3-fp-scale-fast-resident", "v2d_foundation_pose", "estimate_mesh_scale", patch={**{"env": FP_ENV}, "resident": True}, tag="a", time_factor=1.0),
    Strategy("R3-fp-track-fast", "v2d_foundation_pose", "video_to_poses", patch={"env": FP_ENV}, tag="a", time_factor=1.0),
    Strategy("R3-fp-track-fast-resident", "v2d_foundation_pose", "video_to_poses", patch={**{"env": FP_ENV}, "resident": True}, tag="a", time_factor=1.0),
    Strategy("R3-gsplat-fast", *GS, patch=GS_FAST, tag="a", time_factor=1.0),
    Strategy("R3-gsplat-fast-resident", *GS, patch={**GS_FAST, "resident": True}, tag="a", time_factor=1.0),
    Strategy("R3-sam2-fast", "v2d_sam2", "lib.video_to_masks", patch={}, tag="a", time_factor=1.0),
    Strategy("R3-sam2-fast-resident", "v2d_sam2", "lib.video_to_masks", patch={**{}, "resident": True}, tag="a", time_factor=1.0),
    Strategy("R3-gdino-fast", "v2d_grounding_dino", "", patch={}, tag="a", time_factor=1.0),
    Strategy("R3-gdino-fast-resident", "v2d_grounding_dino", "", patch={**{}, "resident": True}, tag="a", time_factor=1.0),
    Strategy("R3-wilor-ref-fast", "v2d_wilor", "lib.video_to_hands", patch={"add": ["--frame_indices", "0"]}, tag="a", time_factor=1.0),
    Strategy("R3-wilor-ref-fast-resident", "v2d_wilor", "lib.video_to_hands", patch={**{"add": ["--frame_indices", "0"]}, "resident": True}, tag="a", time_factor=1.0),
    Strategy("R3-gsplat-fast-H100", *GS, patch=GS_FAST, gpu="H100", tag="b", time_factor=0.8,
             note="launch-bound step loop: faster host CPU + GPU"),
    Strategy("P-gsplat-fast", *GS, patch={**GS_FAST, "profile": True}, tag="profile", time_factor=1.2),
    Strategy("P-sam3d-fast", "v2d_sam3d", "lib.image_to_mesh", patch={"profile": True}, tag="profile", time_factor=1.2),
    # R4 (reconstruct round 2, 2026-11-21). B: the object branch starts on a reference-frame-only
    # SAM2 call; its 3 masks must be byte-identical to the full propagation's frame 0.
    Strategy("R4B-sam2-ref-frame-only", "v2d_sam2", "lib.video_to_masks",
             patch={"add": ["--ref_frame_only"]}, tag="a", time_factor=0.3,
             note="expect identical=3 (masks/{1,2,3}/000000.png), the other frames missing"),
    # D: repeated grid scales reuse their score (exact): expect R3-fp-scale-fast-resident's diff.
    Strategy("R4D-fp-scale-reuse-resident", "v2d_foundation_pose", "estimate_mesh_scale",
             patch={"env": FP_ENV, "resident": True}, tag="a", time_factor=0.7),
    # C: fewer gsplat epochs (changes the optimisation; judged by tools.fastrecon.variance
    # against V-gsplat-fast-1..3, kept on the same volume).
    Strategy("R4C-gsplat-ep8", *GS, patch={**GS_FAST, "set": {**GS_FAST["set"], "--n_epochs": "8"}, "keep": True},
             tag="c", time_factor=0.8),
    Strategy("R4C-gsplat-ep6", *GS, patch={**GS_FAST, "set": {**GS_FAST["set"], "--n_epochs": "6"}, "keep": True},
             tag="c", time_factor=0.6),
    # E: SAM3D seeded (library default 42; the pipeline passed none, so every run differed),
    # fewer diffusion steps, scatter-average texture baking. Resident, as in a run.
    Strategy("R4E-sam3d-seed42-resident", "v2d_sam3d", "lib.image_to_mesh",
             patch={"set": {"--seed": "42"}, "resident": True}, tag="c", time_factor=0.7),
    Strategy("R4E-sam3d-seed42-repeat-resident", "v2d_sam3d", "lib.image_to_mesh",
             patch={"set": {"--seed": "42"}, "resident": True}, tag="c", time_factor=0.7,
             note="must reproduce R4E-sam3d-seed42-resident's diff exactly"),
    Strategy("R4E-sam3d-seed42-steps12-resident", "v2d_sam3d", "lib.image_to_mesh",
             patch={"set": {"--seed": "42", "--stage1_inference_steps": "12", "--stage2_inference_steps": "12"},
                    "resident": True}, tag="c", time_factor=0.6),
    Strategy("R4E-sam3d-seed42-fastbake-resident", "v2d_sam3d", "lib.image_to_mesh",
             patch={"set": {"--seed": "42", "--texture_bake_mode": "fast"}, "resident": True}, tag="c", time_factor=0.5),
    # E. environment checks: does a call reproduce stock3 on this workspace's images?
    Strategy("E-hamer-hands-repeat", "v2d_hamer", "lib.masks_to_hands", tag="a"),
    Strategy("E-hamer-align-repeat", "v2d_hamer", "lib.align_hands", tag="a"),
    # V. gsplat run-to-run variance, stock vs --fast, same inputs (outputs kept for stats)
    Strategy("V-gsplat-stock-1", *GS, patch={"keep": True}, tag="variance"),
    Strategy("V-gsplat-fast-1", *GS, patch={**GS_FAST, "keep": True}, tag="variance"),
    Strategy("V-gsplat-stock-2", *GS, patch={"keep": True}, tag="variance"),
    Strategy("V-gsplat-fast-2", *GS, patch={**GS_FAST, "keep": True}, tag="variance"),
    Strategy("V-gsplat-stock-3", *GS, patch={"keep": True}, tag="variance"),
    Strategy("V-gsplat-fast-3", *GS, patch={**GS_FAST, "keep": True}, tag="variance"),
]
