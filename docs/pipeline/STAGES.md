# Pipeline stages (egocentric video → sellable RL environment)

Product: per-clip **environment factory**. Input is egocentric video from our own capture or from
vendors; output is a validated, sim-ready RL environment that frontier robotics labs buy and train
on. Labs own RL training, data collection, fine-tuning and evaluation; we own everything up to a
delivered, solvable environment.

Times are one 13 s / 390-frame clip (`tissue_box`) on Modal L40S, measured 2026-11-18.

## Ours, per clip (chronological)

| # | Stage | `run_e2e.sh` / code | Status | Time now |
|---|---|---|---|---|
| 1 | Vendor intake spec (format, intrinsics/IMU, metadata) | `python3 -m tools.envpkg intake` | **Built** (A): `intake/v1` contract + checker ([INTAKE.md](INTAKE.md)) | seconds |
| 2 | Ingest | `init` (+ per-clip `setup --skip-reconstruction`) | Optimized in batch mode | 7–26 s |
| 3 | Triage gate (reject bad clips before GPU spend) | `triage` (batch default) | **Built** (B): format, sharpness, exposure, frozen frames, cuts, close-up hands; exit 3 = rejected ([GATES.md](GATES.md)) | ~4 s (CPU) |
| 4 | Reconstruct (hands, object, camera bundle) | `reconstruct --fast` | **Optimized** (round 2: −26%, gate PASS; −80% vs stock e2e) | ~7.2 min (was 9.75) |
| 5 | Reconstruction QA (pass/fail thresholds) | `inspect` (batch default) | **Built** (B): 15 limits incl. mesh-vs-mask IoU, SLAM jumps, jitter, scale; caught both airplane reconstructions ([GATES.md](GATES.md)) | 3–5 s (CPU) |
| 6 | Retarget (robot-agnostic human-hand motion + robot retargets) | `retarget` | **Optimized** (Vega; −60%): IK 100 → 30 iterations, verification video moved to `render` | ~1.3 min (was 3.0–3.5) |
| 7 | Environment build + physics QA replay (assets, collision, support surfaces; numeric flags) | `simulate` | **Optimized** (−50%): CPU physics for the replay, Isaac caches baked into the image | ~0.5–0.6 min (was ~1.1) |
| 8 | Render overlay (visual check) | `render` (overlay of the physics replay; flagged + 1 in 20, after delivery) | **Optimized** | 0 on the path (was 5.3 min); 20 s per rendered clip |
| 9 | Solvability check (replaces 13.4 h RL as our QA) | `solvability` (experimental) | **Paused**: test validated, no cheap proof yet ([OPTIMIZATION.md](OPTIMIZATION.md)) | — |
| 10 | Task / reward spec | package builder (`task/task.json`), from `simulate`'s `env_spec.json` | **Built** (C): `task_spec/v1`: rewards, terminations, action, observations, resets, assist, physics, train/eval episodes, `lift_hold` success checked on the reference ([TASK_SPEC.md](TASK_SPEC.md)) | seconds (CPU) |
| 11 | Export / package (Isaac Lab + MuJoCo) | `python3 -m tools.envpkg build / validate` | **Contract + Isaac Lab export built** (A, D): `env_package/v1` core + validator ([ENV_PACKAGE.md](ENV_PACKAGE.md)); `isaaclab/` proven in a fresh Isaac Lab process ([ISAACLAB_EXPORT.md](ISAACLAB_EXPORT.md)); MuJoCo next (E) | seconds (core) |
| 12 | Delivery to lab | — | **To add** | — |

## Ours, not per clip

| Stage | Status | Time now |
|---|---|---|
| Setup (images, weights; once per workspace) | Partly optimized (weights cached) | 24 min cold → ~2 min cached |
| Batch orchestration / GPU scheduling / $ per environment (cross-cutting) | **Round 1 done** — see [OPTIMIZATION.md](OPTIMIZATION.md) | 13.8 clips/h per workspace, $1.01–1.20/clip |

## Labs' (out of our per-clip scope)

| Stage | `run_e2e.sh` | As intended | Our use |
|---|---|---|---|
| Expert RL training | `train-expert` | ~13.4 h, ~$38 | Reference demo on a sample of environments only |
| Expert render | (`collect` render) | 2.9 min | Reference demo |
| Demonstration collection | `collect` | 11.8 min | Reference demo |
| GR00T fine-tune | `finetune` | 24 min | Reference demo |
| Closed-loop evaluation | `evaluate` | 1.6–8 h (20–100 episodes) | Reference demo |

Assumptions that set this scope (change any and the scope changes): labs buy environments, not
demonstrations; deliverable is robot-agnostic plus a few common robots (Vega + Sharpa today);
Isaac Lab is the primary simulator.

## Build plan (stages not yet built), 2026-11-21

Decided: build before more speed work; package for Isaac Lab and MuJoCo. In dependency order:

| Step | Builds | Status |
|---|---|---|
| A | Package contract + vendor intake spec (stages 1, 11) | **Done**: `tools/envpkg`, [ENV_PACKAGE.md](ENV_PACKAGE.md), [INTAKE.md](INTAKE.md) |
| B | Triage gate + reconstruction QA thresholds (stages 3, 5) | **Done**: `tools/gates`, [GATES.md](GATES.md); calibrated 52/52 (recon) and 32/32 (triage) |
| C | Task/reward spec per clip (stage 10), written into `manifest.task` | **Done**: `tools/envpkg/task.py`, [TASK_SPEC.md](TASK_SPEC.md) |
| D | Isaac Lab exporter (`isaaclab/`), replay check equal to `simulate` | **Done**: `tools/envpkg/isaaclab.py` + `isaaclab_check.py`, [ISAACLAB_EXPORT.md](ISAACLAB_EXPORT.md) |
| E | MuJoCo exporter (`mujoco/`, starting from `robotic_grounding/flash_chord`), second physics check | Todo |
| F | Delivery: archive, lab-side verify, `run_e2e.sh export` / `deliver` gated on QA | Todo |
| G | One end-to-end run on a fresh workspace (founders-78536), with cold setup measured | Todo |
