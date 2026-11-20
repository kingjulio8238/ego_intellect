# Optimization order and record

Stages are listed in [STAGES.md](STAGES.md). Rank = time on our per-clip path × how much of it can
be cut. Quality bar for every change: outputs within the stock pipeline's own run-to-run noise
(reconstruction: `tools/fastrecon/compare_bundles.py`), equal success.

## Order

| Rank | Stage | Why here | Status |
|---|---|---|---|
| 1 | Batch orchestration / GPU scheduling | One GPU wait (up to 15 min) exceeds any stage; dominates throughput and $/environment at scale | **Done (rounds 1–2)**: 2.4–6 → 13.8 clips/h per workspace, $2.76 → $1.01–1.20/clip; sharding across workspaces, preemption-safe |
| 2 | Render overlay (5.3 min) | Largest unoptimized stage; likely removable from the critical path (optional / async / low-res) | **Done**: 5.3 min → 0 on the critical path; numeric QA in `simulate` (+1 min, same boot); overlay video 20 s after delivery for flagged + 1 in 20 — see below |
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
| F11 | **Modal GPU fallback lists fall back within ~3 s.** Which type is short changes by the hour and by workspace (13:57: L40S scarce, H100 3 s; 15:10 on credited: H100 110–200 s, L40S and A100-40GB 3 s; 15:20: A100-80GB minutes) | function with `gpu=["H100","A100-80GB","L40S","A100-40GB"]`: 4/4 started in 3.0–3.1 s on A100-40GB/L40S while H100-only took 196 s |
| F10 | A local `__pycache__` under `reconstruction/modules` changes all 12 module images' build context (`.dockerignore` only ignores one level) → full rebuild + cold loads | launch from a clean snapshot; fix is in the todo list |

### Round 2: GPU-type flexibility, numerics gate, workspaces (2026-11-19)

**Gate: can a module run on another GPU type?** Seeded tissue_box clip (stock calibration, as
final2), bundle gated against stock3:

| Run | GPU per module | Camera / object | Hands | Verdict |
|---|---|---|---|---|
| b3_tbseed | all L40S | pass | pass (hand rot 0.19°, scale 0.0011, betas 0.042) | PASS |
| h1 | all H100 | pass | **fail**: hand rot 1.73° (tol 0.35), fingers 2.9°, scale 0.0096, betas 0.070 | FAIL |
| h2 | all A100-80GB | pass | **fail**: hand rot 1.85°, fingers 3.1°, scale 0.0114, betas 0.078 | FAIL |
| h3 | hand chain + calibration L40S, rest H100 | pass | rot 0.20° pass; **scale 0.0119, betas 0.086 fail** | FAIL |
| h4 | hand chain + calibration L40S, rest A100-80GB | pass | pass (rot 0.19°, scale 0.0053, betas 0.044) — but FoundationPose's scale search landed at 1.067 vs 1.164 on L40S; gsplat corrected it this time | PASS (1 sample; same instability as h3) |

Why (isolated per stage, H100 vs L40S on identical inputs): the hand chain is bit-identical
run to run on L40S, but on H100 WiLoR's detection moves by 0.002, SAM2's hand prompt masks by
1.3 % of pixels, and HaMeR's tracks cascade from there. With the hand chain on L40S, MoGe's depth
on H100 differs by only 0.02 % (p95), yet FoundationPose's discrete scale search lands 2 % away
(1.139 vs 1.164); replaying that FoundationPose call on L40S with the H100 inputs gives the same
1.139, so FoundationPose itself is arch-stable and the flip comes from its input. **Decision
(user, 2026-11-19): keep the strict bar → every module stays on L40S.** The fallback machinery
ships switched off (`V2D_MODAL_MODULE_GPU`, `V2D_MODAL_GPU_PER_IMAGE`, `V2D_MODAL_HEDGE_GPU`).
Note the strict bar is tighter than stock itself across machines: AnyCalib's re-encode depends
on the host core count and moves hand rotation by 3.4°.

H100 is not faster for this pipeline (exec per call vs L40S: MoGe 29 vs 41 s, gsplat ~140 vs
~140 s, FoundationPose 72 vs 70 s, SAM3D 106 vs ~90 s) and costs 2× per hour.

**Workspaces:** `bootstrap_workspace.py` prepared credited and juliansaks from sales-32662
(secret, weights copied and md5-verified: 573/573 identical, 14/14 images built, warm probe).

**b5: production config, sharded over two fresh workspaces, with an injected preemption.**
8 clips (6 tissue_box incl. one seeded, 2 airplane), `batch_shard.py` → credited 5 + juliansaks 3;
L40S only, keep-warm models (240 s scaledown on model images, 20 s elsewhere), hedges after
30 s. Partway through, the credited control container was stopped (`modal container stop`,
what a Modal preemption does).

| Shard | Clips ok | Makespan | Start wait p50 / p95 | Warm-model hits | Notes |
|---|---|---|---|---|---|
| credited | 4/5 | 22.0 min (incl. preemption) | 4.5 s / 157 s | 13 of 32 model calls | restart skipped `init`, resumed every clip; b5_tbseed gate **PASS** after resume; b5_ap1 failed: video corrupted by the duplicate-hedge bug below (fixed) |
| juliansaks | 3/3 | 19.3 min | 10.9 s / 209 s | 3 of 21 | |
| combined | 7/8 | 24 min wall for 8 clips on 2 workspaces | | | ~17.5 good clips/h across the two |

**Bugs found and fixed in round 2:**
- Hedged call ran twice: the claim was deleted when the call finished, and the cancelled
  primary started 31 s later anyway. Its orphaned AnyCalib encode then wrote into the next
  call's mount and corrupted another clip's video (b5_ap1). Fixed: claims stay as tombstones;
  a worker kills its command's whole process group when its input is cancelled.
- Bootstrap: Modal client profile must be pinned before import; create the state volume
  before uploading inputs; never edit a build snapshot while images build from it.

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

## Render overlay (stage 8), 2026-11-20

Decisions (A2/A3): the video is **internal QA**; numbers gate clips (phase C), a video goes to
**flagged clips plus 1 in 20**. RTX video stays for demos; phase E (per-clip RTX) is not planned.

**Phase A, profile.** The 5.3 min "environment on video" was `train-expert` borrowed for
10 iterations × 24 steps with RTX cameras on a Vast A6000 (Isaac Lab 2.3.2 cameras crash on
Modal's driver). The same environment without cameras, on Modal L40S (tissue_box, Vega,
1 env, 200 steps):

| Phase | Cold container | Warm (2nd run, same container) |
|---|---|---|
| Kit boot (no cameras) | 20 s | 10.5 s |
| Scene creation | 15.2 s | 5.9 s |
| Simulation start + managers | 4.5 s | 3.0 s |
| 200 physics steps + close | 16 s | 15 s |
| **Total** | **58 s** | **35 s** |

So ≥ 4.3 of the 5.3 min is cameras/RTX (shader start-up, per-frame render, encode) plus the
10 training iterations. The RTX split is not measured: it needs an RTX host outside Modal, and
was deferred because only demos need RTX now (script ready: a cold/warm 200-frame render and
the exact old command, self-stopping, 45 min cap).

**Phase B, off the critical path.**
- `run_e2e.sh render [--embodiment] [--video-length 200]`: a separate, optional stage. Zero-action
  replay with the reference markers on (`dummy_agent.py --record_video --debug_vis`), one MP4
  per embodiment in `RUN_ROOT/render/<embodiment>/`, success marker `render.done`. No
  training iterations.
- `train-expert` no longer records video: training is camera-free by default (it used to add an
  8-step video on the 1-iteration default, which needs RTX cameras).
- Batches record a render queue at launch (`.e2e/batch/<id>/render_queue.json` and the progress
  store): flagged clips plus a stable 1-in-`render_every` sample (hash of the clip name;
  manifest `render_every`, default 20, 0 = none). It is drained after delivery, never inline.
- `train.py --video`: an interval shorter than the length is raised to the length (interval 1 /
  length 200 had written 240 one-frame clips).
- On Modal, `render` refuses up front with the reason (RTX cameras crash on its driver) rather
  than crashing inside Isaac; `V2D_MODAL_ALLOW_RTX=1` is for testing a fix.

**Phase C, numbers instead of looking.** `simulate` (Vega) now builds the environment and
replays the whole clip in physics without cameras (`robotic_grounding/scripts/rsl_rl/replay_metrics.py`):
frame-0 reset, zero residual actions (the robot PD-tracks the retargeted reference), object
assist (VOC) and curriculum off. It writes `RUN_ROOT/simulate/vega/replay_metrics.json` with
the checks below and a `flag`; one Isaac boot does build and QA (35–60 s on L40S, replay
13–19 s). The open-loop replay cannot hold the object through a whole clip even when the clip
is good (tissue_box drops it at 40 %: the trained policy supplies the corrections), so the
checks use the phases it does judge, plus one kinematic check that needs no simulator
(`scripts/data_quality_checks/grasp_drift.py`, also a `data_assessor.py` check).

| Check (limit) | What it catches | good | good, 2nd run | +4 cm (floats) | −2 cm (sinks) | +5 cm x (overlaps hands) | object 0.5 s late |
|---|---|---|---|---|---|---|---|
| Object error, first 0.5 s, max (1.5 cm) | floats, sinks, bad placement | 0.64 cm | 0.64 cm | **3.7** | **4.3** | **7.1** | 0.68 |
| Object error before first contact, p95 (3.5 cm) | drift at rest | 2.4 | 2.4 | **5.5** | **4.1** | **6.8** | 2.6 |
| Object error, first 1 s of contact, p95 (4.5 cm) | object does not follow the grasp | 3.1 | 3.1 | **6.1** | — | — | 2.5 |
| Missed contact, first 1 s of contact, mean (0.5) | hands miss the object | 0 | 0 | 0 | — | — | 0 |
| Wrist error, p95 (5 cm) | robot cannot follow its reference | 3.3 | 3.4 | 3.2 | 3.4 | 3.9 | 3.9 |
| Wrist drift in the object frame during contact, p95 (3.5 cm) | hands and object out of sync | 2.3 | 2.3 | 2.4 | 2.2 | 2.4 | **4.9** |
| Replay ends before the first contact | blow-ups | no | no | no | **yes (5 steps)** | **yes (4 steps)** | no |
| **Flag** | | no | no | **yes** | **yes** | **yes** | **yes** |

6/6 classified correctly; two runs of the good clip agree to 0.2 mm. Caveats: one reviewed
clip is a thin baseline (re-calibrate as reviewed clips accumulate); the defects are
synthetic; missed-contact and wrist checks were not triggered by any defect here; the
floating-hand embodiment keeps the 2-step smoke (checks calibrated for Vega only).

**G, robot-stack image rebuilds.** Any edit under `robotic_grounding/` (even a stray
`__pycache__`) rebuilt the whole rg image on Modal (~18 min): it was one Docker layer with the
directory as context, and the GR00T layers came after it. Now it is built in three parts like the
module images (`images.robotic_grounding_image`): the Dockerfile's steps before its first COPY
(empty context), then GR00T/nvcc/host packages, then the source COPY and the steps after it;
`.dockerignore` now also drops nested `__pycache__`/`*.pyc`. Measured on sales-32662: cold build
16 min (once per workspace); a source edit **66 s** (rebuilds from `COPY scripts/` only); a
`__pycache__`-only change **15 s, same image** (no rebuild).

**Phase D, overlay video on Modal.** `render` now defaults to `--backend overlay`:
`scripts/render_replay_overlay.py` draws `simulate`'s physics replay (`replay_metrics.py` saves
`replay_states.npz`: joints, object and reference poses, wrists) with pyrender/EGL: robot
(pinocchio FK from the simulated joints), object where physics put it (orange), reference ghost
(green), wrist targets vs simulated wrists, support surface, and a caption with the QA verdict.
No Isaac, no RTX: **18–21 s per clip on Modal L40S, ~$0.02**, vs 5.3 min on a Vast A6000 for the
RTX clip. `--backend rtx` (Isaac cameras) stays for demos on an RTX host and is refused on
Modal. Compared with the stored RTX clip of tissue_box: the RTX clip is a distant view of the
robot with axis markers; the overlay is a close view where the physics object and its reference
are drawn together, so a float, sink, drop or desync shows as the two boxes separating, and the
QA verdict is printed on the frame. Each synthetic defect is visible (see the D3 check below).

**Phase F, batches.** A batch manifest now runs, per clip, `robot_steps` (default
`["retarget", "simulate"]`) in one rg GPU container after the reconstruction phase
(`app.batch_pipeline`), reads each clip's QA flag, and drains the render queue (flagged clips
plus a stable 1-in-`render_every` sample) last, one clip at a time. Result record:
`.e2e/batch/<id>/pipeline.json` and the progress store. Verified on sales-32662 with two
reconstructed tissue_box clips (`b3_tbseed`, `b3_tb2`; `render_every: 1` to exercise the
drain): 2/2 delivered in 5.3 min (retarget 3.5 min + simulate 1.1 min per clip, 2 in parallel),
renders 30 s each after delivery, ~$0.5 in total. Both clips were **flagged**: the reference
object drifts 4.8–5.2 cm before contact and the wrists drift 5.5–5.7 cm in the object frame
during contact (reviewed tissue_box_simple: 2.4 and 2.3 cm). The overlay shows the reference
box moving away from the resting physics box before the grasp and the hands not enclosing the
box: these fast-pipeline reconstructions (SAM3D mesh, no cleaned scan) look worse than the
reviewed clip, but no human has reviewed them yet, so whether this is a real defect or limits that are too tight is open.

**Re-calibration (2026-11-20, after review).** Watched side by side with tissue_box_simple, the
two flagged clips were judged fine, so the limits were too tight. Changes: the wrist-drift
check measures drift from the median wrist position in each contact segment (contact onset is
where reconstructions are noisiest; measured from the first frame it charged good clips for
that frame), and the limits moved to cover the three reviewed-good clips:

| Check | Limit (was) | tissue_box_simple | b3_tbseed | b3_tb2 | +4 cm | −2 cm | +5 cm x | 0.5 s late |
|---|---|---|---|---|---|---|---|---|
| Object error, first 0.5 s, max | 2.5 cm (1.5) | 0.64 | 1.32 | 1.57 | **3.7** | **4.3** | **7.1** | 0.68 |
| Object error before contact, p95 | 6.5 cm (3.5) | 2.4 | 5.2 | 4.8 | 5.5 | 4.1 | **6.8** | 2.7 |
| Object error, first 1 s of contact, p95 | 7.5 cm (4.5) | 3.1 | 5.9 | 5.9 | 6.1 | — | — | 2.5 |
| Wrist error, p95 | 5 cm | 3.4 | 3.0 | 3.0 | 3.2 | 3.4 | 3.9 | 3.9 |
| Wrist drift in object frame (median), p95 | 4.1 cm (3.5, from first frame) | 1.5 | 3.6 | 3.7 | 1.5 | 1.5 | 1.6 | **4.6** |
| Replay ends before first contact | — | no | no | no | no | **yes** | **yes** | no |
| **Flag** | | no | no | no | **yes** | **yes** | **yes** | **yes** |

7/7 correct. Placement defects keep wide margins (settle 1.6 cm worst good vs 3.7 cm best
defect, plus the early end). The timing defect is caught with a thin one: drift 3.7 cm worst
good vs 4.6 cm for a 0.5 s lag (about 0.4–0.5 cm each side of the limit); a smaller lag would
pass. The before-contact and grasp object errors are now loose backstops: real
reconstructions drift up to ~5 cm before contact and still look right.

Limits: the robot phase starts after every clip's reconstruction (the reconstruction container
holds all the clips' run-root locks until it ends); per-clip hand-off is a follow-up. The
`rtx` backend was not re-run (same command as before, refused on Modal).

## Chip choice: L40S vs RTX PRO 6000 Blackwell (2026-11-20)

**Decision: standardize on L40S.** RTX PRO 6000 does not meet the pre-agreed bar (≥1.3× faster
or ≥20 % cheaper per clip, and Isaac Sim working); revisit when the two blockers below clear.

| Check | Result |
|---|---|
| Images | All 12 module images port to Blackwell (`V2D_MODAL_ARCH=blackwell`: PyTorch 2.7.1 + CUDA 12.8, CUDA code for sm 12.0, TensorRT 10.9). Fixes needed: CMake 4 policy, two CUDA ops using a dispatch form PyTorch 2.7 removed (GroundingDINO, FoundationPose), WiLoR-mini pinning torch ≤ 2.5, PyTorch ≥ 2.6 `weights_only` checkpoint loading. Every compiled extension loads and runs on sm 12.0 (spconv, gsplat, pytorch3d, nvdiffrast, kaolin, lietorch, torch-scatter, TensorRT) |
| Reconstruct, 2 seeded tissue_box clips | 2/2 ok |
| Repeatable on itself | **PASS**: Blackwell vs Blackwell hand rot 0.17°, fingers 0.40°, betas 0.053 (L40S vs L40S: 0.26°, 0.52°, 0.071) |
| Close to L40S | Blackwell vs L40S: hand rot 1.57°, fingers 2.76° (strict tol 0.35° / 0.70°: fail); within stock's own cross-machine spread (encoder thread count: 3.42° / 4.33°); camera, object, hand translation, betas pass |
| Speed (GPU exec per clip) | 617 s L40S → 567–579 s first runs (one-time TensorRT engine and gsplat compile), **~485 s steady (1.27×)**. Per module: MoGe 1.6×, SAM2 1.5–1.7×, DROID 1.75×, gsplat 1.0–1.4×, HaMeR 1.2×, SAM3D 0.7–1.1×, FoundationPose 0.6–1.1×, WiLoR 1.0× |
| Cost per clip (GPU time × list price) | Modal $3.03 vs $1.95/h: Blackwell **~22 % more** per clip; market medians ($2.23 vs $1.58): ~11 % more; purchase (~$15–20k vs ~$9k): more again unless packed |
| Isaac Sim (simulate stage) | **FAIL on Modal**: Isaac Sim 5.1 (our Isaac Lab 2.3.2 image, and vanilla 5.1.0) segfaults in the RTX renderer on Blackwell with Modal's driver 610/620; NVIDIA lists driver 580.x for 5.1. Vanilla Isaac Sim 6.0.0 boots headless on the same GPU |
| Headroom for packing | Blackwell 96 GB, per-call peak ≤ 23 GiB, GPU util 8–67 %: 3–4 calls could share one GPU. Not measured: the executor runs one call per container |
| Availability on Modal | RTX-PRO-6000 started in 1–37 s throughout (L40S waited minutes at times) |

**Revisit Blackwell when** (a) the robot stack moves to Isaac Sim ≥ 6.0, or dedicated machines
run driver 580.x for Isaac Sim 5.1; and (b) the executor packs several calls per GPU — the 96 GB
is Blackwell's real advantage for this pipeline, not per-call speed.

## Next: todo for the remaining stages (for review)

Same method as reconstruct: profile → isolate each call → smoke levers in parallel against
recorded inputs → gate against stock → one e2e. Ordered by the ranking above.

**1b. Batch / GPU scheduling, round 2 — done (see Round 2 above)**
- [x] GPU-type flexibility built (fallback lists, per-image GPU types, hedge GPU type,
      multi-arch gsplat prebuild, per-arch extension caches); **off by default** after the gate.
- [x] Numerics gate on H100 and A100-80GB: hands and object scale move beyond the strict bar;
      L40S only (user decision, 2026-11-19).
- [x] Preemption resume and keep-warm models validated in a real sharded batch (b5).
- [x] Short scaledown for non-model images (20 s) in b5.
- [x] Workspace bootstrap: credited and juliansaks prepared and used.
- [ ] Open: re-run b5_ap1 (needs credit: credited $21.6, juliansaks $23.2 of the $25 line).
- [ ] Open, only if the bar is relaxed to stock's own cross-machine spread: measure that
      spread (stock on hosts with different core counts / GPU types), then re-gate the fallback.

**2. Render overlay (5.3 min)** — plan approved 2026-11-20; record above
- [x] A. Profile (camera-free part on L40S; RTX split deferred) and decide: internal QA,
      flagged + 1 in 20.
- [x] B. Separate optional `render` stage; sampled render queue per batch; recorder fix.
- [x] C. Numeric checks from a camera-free physics replay in `simulate` (one boot), plus
      the kinematic grasp-drift check; calibrated on tissue_box and 4 synthetic defects.
      Building block for the solvability check (rank 3).
- [x] G. rg image split: a robotic_grounding edit rebuilds in 66 s (was ~18 min).
- [x] D. Overlay renderer of the physics replay (default `render` backend), 18–21 s/clip on
      Modal; every synthetic defect visible; compared with the RTX clip.
- [x] F. Batches run retarget → simulate per clip, gate on the QA flag, drain the render
      queue last; verified on 2 clips. (E, per-clip RTX, not planned: internal QA only.)
- [x] Reviewed the 2 flagged fast-pipeline clips (fine) and re-calibrated: 3 good clips
      pass, 4 defects flagged.
- [ ] Follow-ups: grow the reviewed set (real bad clips especially; the timing check's
      margin is thin); per-clip hand-off from reconstruction to
      the robot phase; floating-hand QA/overlay; RTX profile only if demos need it.

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
