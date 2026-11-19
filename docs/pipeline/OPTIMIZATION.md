# Optimization order and record

Stages are listed in [STAGES.md](STAGES.md). Rank = time on our per-clip path × how much of it can
be cut. Quality bar for every change: outputs within the stock pipeline's own run-to-run noise
(reconstruction: `tools/fastrecon/compare_bundles.py`), equal success.

## Order

| Rank | Stage | Why here | Status |
|---|---|---|---|
| 1 | Batch orchestration / GPU scheduling | One GPU wait (up to 15 min) exceeds any stage; dominates throughput and $/environment at scale | **Done (round 1)**: 2.4–6 → 13.8 clips/h per workspace, $2.76 → $1.01–1.20/clip |
| 2 | Render overlay (5.3 min) | Largest unoptimized stage; likely removable from the critical path (optional / async / low-res) | Todo |
| 3 | Solvability check (new) | Must be cheap by design — the naive version is 13.4 h of RL | Todo |
| 4 | Retarget (3.0 min) | Next-largest existing stage | Todo |
| 5 | Environment build (2.0 min) | | Todo |
| 6 | Triage gate (new) | Saves the full per-clip cost of every rejected clip | Todo |
| 7 | Ingest (1.4 min) | Mostly container-start overhead | Todo |
| 8 | Setup | One-time per workspace; cold image loads 5–10 min per module on a new workspace | Todo |
| 9 | Task/reward spec, export, delivery (new) | Small; build correct first, then fast | Todo |
| 10 | Reconstruct, remaining | gsplat backward (~100 s, launch-bound); seed SAM3D (50 s; needs approval: changes stock) | Mostly done |

## Done: reconstruct (2026-11-18)

Stock 45:47 (re-run) / 36.5 min (e2e run) → ~9:45 (`reconstruct --fast`), gate PASS (fast2b,
final1, final2). Method: record a stock run, replay any single module call in isolation with a
strategy applied (`tools/fastrecon`), keep a strategy only if it saves time and its outputs stay
inside stock-vs-stock noise, then stack winners into one gated e2e run.

| Item | Stock | Fast | Cut | How |
|---|---|---|---|---|
| Diagnostic renders | 656 s | 0 | −100% | not produced under `--fast` (nothing downstream reads them) |
| EKF smoothing | 15 s | 0 | −100% | skipped when gsplat refinement runs |
| Track interpolation | 66 s | 1 s | −98% | I/O fix |
| WiLoR (reference frame) | 71 s | 3 s | −96% | reference frame only + resident preload |
| gsplat refinement | 410 s | 139 s | −66% | one host sync per step, torch.compile, persistent inductor cache |
| MoGe depth | 112 s | 41 s | −63% | overlapped I/O |
| FoundationPose tracking | 74 s | 35 s | −53% | sequential decode + resident predictor |
| Grounding DINO | 11 s | 5 s | −53% | resident preload |
| FoundationPose scale | 146 s | 79 s | −46% | hull diameter, quiet logging, resident predictor |
| SAM3D | 162 s | 50–91 s | −44…−69% | prebuilt extension + resident pipeline |
| HaMeR regress / align | | | −11% / −28% | |
| Whole run | 45:47 | ~9:45 | −79% | + DAG concurrency (object ∥ hand ∥ SLAM ∥ GeoCalib), warm-ups |

Run-to-run: gsplat fast-vs-stock spread equals stock-vs-stock on all 7 metrics. SAM3D is unseeded
(nondeterministic in stock too); fast3 failed betas on that alone.

## Batch orchestration / GPU scheduling (2026-11-19)

### Results

Same 13-step `reconstruct --fast` per clip; clips are tissue_box (390 frames) and airplane
(780 frames, first time through the pipeline: ok in every batch).

| Run | Mode | Clips | Makespan | Clips/h per workspace | GPU-start wait p50 / p95 / max | Billed $/clip |
|---|---|---|---|---|---|---|
| final2 (before) | one clip at a time, latency mode (`--fast` residents + warm-ups) | 1 | 24:37 (9:45 without its stall) | 2.4 (6.2) | 15 s / 142 s / 924 s | 2.76 |
| b1 | one control container per clip, 8 GPU slots, exit after each call | 4 | 19.9 min | 12.0 | 32 s / 93 s / 149 s | 1.05 |
| b2 | one shared control container, GPU ledger (9), 60 s scaledown | 4 | 18.6 min | 12.9 | 7 s / 53 s / 200 s | 1.20 |
| b3 | b2 + dirty-only index saves, hedges inside the budget | 6 | 26.0 min | **13.8** | 14 s / 183 s / 231 s | **1.01** |
| b4 | b3 + keep models warm across clips, 240 s scaledown for model images | 6 | — | — | — | — (control container preempted at 10 min; restart failed on `init`: fixed) |

Quality: the batch changes how calls are scheduled, not what they compute. A b3 clip seeded
with the stock run's calibration (as final2 was) **passes** the stock gate. Unseeded clips
differ from stock3 only through AnyCalib's x264 re-encode, whose bitstream depends on the
encoder thread count (stock: host core count; ours: pinned to 8): every thread count gives a
different `undistorted.mp4` (md5 probe), so stock itself varies this much across hosts. Clips
encoded the same way agree with each other within (or at the edge of) the stock noise floor.

### Findings

| # | Finding | Evidence |
|---|---|---|
| F1 | Workspace GPU cap: 10 concurrent GPUs on Modal Starter, idle (scaledown) containers included; a call past it waits silently and starts ~65 s after a slot frees | 11 concurrent probes: 10 started in 6–18 s, the 11th waited 81 s |
| F2 | **L40S capacity is the binding constraint and varies by the hour.** At times only ~4 L40S containers start promptly per workspace and the rest wait minutes, far below the cap | newacc 13:51: 10 probes, ≤4 running, DROID image waited 5 min. Tiny-image starts at 13:57 on both workspaces: **L40S 10–530 s**, A100-80GB 11–277 s, A10G 3–323 s, A100-40GB 3 s, **H100 3.3–4.5 s (8/8)**; at 12:12 L40S was 3 s |
| F3 | Cold start of a loaded module image is 6–18 s; a container reused for the next call of its image starts in <1 s | `--probe`; b2: 26 of 60 calls reused a container, p50 start 7 s |
| F4 | Single-clip latency mode exceeds the cap by itself (5 resident preloads + ~5 warm-ups + calls); idle speculative GPUs were ~half of its bill | final2 billed $2.76 vs $1.45 of call time |
| F5 | Per-clip ingest (`init` + `setup`) was 4.4 min, almost all re-checking per-workspace weights | batch: `setup --skip-reconstruction`, 7–20 s per clip |
| F6 | One run per workspace (global lock, single sync index) | per-run-root locks and index files; up to 6 clips concurrently |
| F7 | Shared-container overheads: every call rewrote the sync index of every clip; journal writes failed during another clip's volume reload | dirty-only partition saves; journal write retries |
| F8 | Long-running control containers get preempted, and Modal restarts the same input | b4: preempted at 10 min. Fixed: `init` skipped when the run exists, lock taken over by the same input; the pipeline resumes through its skip-if-done steps |
| F9 | Keeping the next call's model preloaded in a reused container works | same container, second call: SAM3D exec 98.6→78.7 s, FoundationPose scale 96.7→90.6 s |
| F10 | A local `__pycache__` under `reconstruction/modules` changes all 12 module images' build context (`.dockerignore` only ignores one level) → full rebuild + cold loads | launch from a clean snapshot; fix is in the todo list |

### What was built (`tools/modal_v2d`)

- `run_e2e_modal.sh --batch manifest.json`: many clips in one Modal app; `mode: shared` (default)
  runs every clip's init → setup → reconstruct in one control container (`runtime.run_batch_role`),
  `mode: per_clip` gives each clip its own. Remote driver: survives a laptop disconnect.
- GPU ledger (`ledger.py`): models every GPU container the batch holds (busy, idle until its
  scaledown, hedges); calls reuse idle containers of their image, new containers start only
  under the budget, speculation never takes the last slots.
- Hedged dispatch: a call with no container after 30 s is also sent to a twin function (same
  image and GPU, separate pool), only with budget left; first container to claim it runs it.
- Keep-warm models (`keep_models`), per-image scaledown windows (`options.scaledown`).
- Per-run-root locks, per-root sync index partitions, preemption-safe restart.
- Tools: `--probe` (cold start per image), `cap_probe`, `hedge_check`, `warm_check`,
  `batch_report.py` (makespan, clips/h, start waits, $/clip estimated and billed),
  `batch_shard.py` (split a manifest across workspaces by credit headroom).
- 51 unit tests (`python -m pytest tools/modal_v2d/tests`).

### Where the ceiling is now

GPU execution is ~640 s per tissue clip. b3 kept its 9-GPU budget only ~30 % busy with
execution: the rest was idle scaledown containers counted against the cap, container starts
(L40S scarcity), and model loads. The next gains, in order: GPU-type flexibility for the
arch-portable modules, hedging to H100 (F2; needs a numerics gate per module), shorter scaledown for
non-model images in throughput mode, keep-warm models (F9), and more workspaces in parallel
(each adds its own cap and credit).

## Next: todo for the remaining stages (for review)

Same method as reconstruct: profile → isolate each call → smoke levers in parallel against
recorded inputs → gate against stock → one e2e. Ordered by the ranking above.

**1b. Batch / GPU scheduling, round 2** (the remaining throughput is here: F2)
- [ ] GPU-type flexibility: send hedges to an H100 twin instead of a second L40S pool (H100
      started in 3–5 s, 8/8, while L40S waited up to 9 min; H100 is $3.95/h vs $1.95 but faster). Module images already compile
      for sm 8.0/8.6/8.9/9.0; FoundationPose TRT engines are already per-sm files; change
      `GSPLAT_PREBUILD` from arch 8.9 to `8.0 8.6 8.9 9.0` (rebuilds SAM3D + gsplat images).
- [ ] Numerics gate per module on A100 vs L40S (replay harness: each call on both, compare
      with stock-vs-stock spread), then enable the fallback only where it passes.
- [ ] Rerun b4 (2 clips) to validate preemption resume + keep-warm models in a real batch.
- [ ] Throughput mode: 10–20 s scaledown for images without a model to keep warm; measure
      budget utilization (b3: ~30 % of the 9-GPU budget was execution).
- [ ] Workspace bootstrap (images, md5-verified weights, one warm load per image) so
      `batch_shard.py` can spread a batch over every workspace with credit
      (credited, juliansaks, julian-41138 have $13–27 headroom).

**2. Render overlay (5.3 min, `simulate` render part)**
- [ ] Profile the render: Isaac Kit boot vs per-frame RTX render vs encode; GPU util.
- [ ] Decide what the render is for. It is a visual check that no downstream stage reads, so
      take it off the critical path: optional flag, async after delivery, or a sampled
      check on 1 clip in N.
- [ ] If kept: lower resolution/fps, fewer frames, rasterized (non-RTX) mode, reuse the
      Kit process of the build step (one boot instead of two).
- [ ] Gate: build artifacts byte-identical; render (if kept) visually equivalent.

**3. Solvability check (new)**
- [ ] Define "solvable" with the labs' acceptance in mind: object follows the retargeted
      motion in physics (tracking error, no penetration/slip, success by the task's own
      evaluator `lift_hold`).
- [ ] Cheapest candidate first: kinematic replay + physics settle, then a short
      tracking-policy rollout; calibrate both against the 13.4 h expert on known
      good/bad clips.
- [ ] Batch it: many environments per Isaac process (`--num_envs`), one GPU.

**4. Retarget (3.0 min)**
- [ ] Profile prepare / load / Vega / Sharpa sub-steps; most of it is probably Isaac/Kit
      boot and per-frame IK in Python.
- [ ] Levers: one Kit boot for all embodiments, vectorized IK, run embodiments in
      parallel, residents for the loader image.
- [ ] Gate: retargeted joint trajectories within stock run-to-run noise.

**5. Environment build (2.0 min)**
- [ ] Profile asset conversion (mesh → USD, collision decomposition) vs Kit boot.
- [ ] Levers: cache CoACD/collision results by mesh hash, batch several clips per Kit process.

**6. Triage gate (new)**
- [ ] Cheap per-clip checks before any GPU: duration, resolution, blur, hand visibility
      (CPU detector on a few frames), object present (one DINO call).
- [ ] Measure the reject rate on vendor data; every reject saves a full clip's cost.

**7. Ingest (1.4 min → 16–26 s in batch)**
- [ ] Done in batch mode (`setup --skip-reconstruction` per clip). Remaining: input upload
      path for vendor data (bulk to volume, no per-file round trips).

**8. Setup / workspaces**
- [ ] Ignore `**/__pycache__` and `**/*.pyc` in `reconstruction/modules/.dockerignore`
      (today only one level deep, so a local test run invalidates all 12 images). One-time
      rebuild + cold image loads per workspace when applied.
- [ ] Workspace bootstrap script: build images, copy weights (md5-verified), warm-load
      each image once, seed resident hints; then `batch_shard.py` can use it.

**9. Task/reward spec, export, delivery (new)** — build correct first, then fast.

**10. Reconstruct, remaining**
- [ ] gsplat backward (~100 s, launch-bound): CUDA graphs / fused kernels.
- [ ] Seed SAM3D (changes stock outputs → needs approval).
