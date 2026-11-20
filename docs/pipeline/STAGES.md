# Pipeline stages (egocentric video → sellable RL environment)

Product: per-clip **environment factory**. Input is egocentric video from our own capture or from
vendors; output is a validated, sim-ready RL environment that frontier robotics labs buy and train
on. Labs own RL training, data collection, fine-tuning and evaluation; we own everything up to a
delivered, solvable environment.

Times are one 13 s / 390-frame clip (`tissue_box`) on Modal L40S, measured 2026-11-18.

## Ours, per clip (chronological)

| # | Stage | `run_e2e.sh` / code | Status | Time now |
|---|---|---|---|---|
| 1 | Vendor intake spec (format, intrinsics/IMU, metadata) | — | **To add** | — |
| 2 | Ingest | `init` (+ per-clip `setup --skip-reconstruction`) | Optimized in batch mode | 7–26 s |
| 3 | Triage gate (reject bad clips before GPU spend) | — | **To add** | — |
| 4 | Reconstruct (hands, object, camera bundle) | `reconstruct --fast` | **Optimized** (−73%; −79% vs stock re-run) | 9.75 min |
| 5 | Reconstruction QA (pass/fail thresholds) | `inspect` | Exists; needs thresholds | 0.1 min |
| 6 | Retarget (robot-agnostic human-hand motion + robot retargets) | `retarget` | Not optimized | 3.0 min |
| 7 | Environment build + physics QA replay (assets, collision, support surfaces; numeric flags) | `simulate` | Not optimized; QA added in the same boot | 2.0 min |
| 8 | Render overlay (visual check) | `render` (optional; flagged + 1 in 20, after delivery) | **Off the critical path** | 0 on the path (was 5.3 min) |
| 9 | Solvability check (replaces 13.4 h RL as our QA) | — | **To add** | — |
| 10 | Task / reward spec | — | **To add** | — |
| 11 | Export / package (Isaac Lab / USD; possibly MuJoCo) | — | **To add** | — |
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
