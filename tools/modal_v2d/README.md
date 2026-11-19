# Run the V2D E2E workflow on Modal

This package runs the repository-root launcher, `run_e2e.sh`, on Modal GPUs. Every
subcommand is supported: setup, reconstruct, retarget, simulate, train-expert, collect,
finetune and evaluate. The repository's own code is unchanged apart from one Modal-aware
doctor check in `tools/e2e/stages.py`.

> **Status (2026-11-19):** every stage has run on Modal end to end (tissue_box, 2026-11-18).
> `reconstruct --fast` runs in ~9:45 (stock 45:47) with the bundle inside stock run-to-run
> noise. Batch mode (many clips, one app) is the current work; see
> [docs/pipeline/OPTIMIZATION.md](../../docs/pipeline/OPTIMIZATION.md). Tests:
> `python -m pytest tools/modal_v2d/tests`.

## How it maps onto the repository's docker model

The repository orchestrates on the host and runs inference in containers: every model runs
through `docker run` or `docker exec`. Modal cannot run a docker daemon, so this package
does the following instead:

| Docker on a GPU box | On Modal (this package) |
|---|---|
| Per-module images (`docker/build.py`) | `img_<module>` Modal functions. Each image is built by Modal from the module's **unmodified** Dockerfile, using the same `reconstruction/modules` build context |
| `robotic-grounding` image (`workflow/Dockerfile`) | Built from the same Dockerfile with two mechanical edits: a pinned FROM, and the `python` wrapper moved to `/opt/isaac/bin`. Isaac-GR00T is added at `/opt/Isaac-GR00T`, in its own Python 3.10 uv environment |
| The host running `run_e2e.sh` | A role container. `control` is CPU-only and runs init/use/status/setup/reconstruct/import-reconstruction/inspect. `rg` runs on an L40S and handles doctor/retarget/simulate/train-expert/collect/finetune/evaluate |
| `docker` CLI | `docker_shim.py`, placed on `PATH`. If the image is the current container, the command runs in-process and bind mounts become symlinks. Any other image becomes a call to that image's Modal function. `docker build` is a no-op |
| Bind mounts (`-v host:/data/x`) | The parent process syncs each mounted directory to the `v2d-e2e` volume (delta only), the worker symlinks it into place, and changed files are synced back. Weights stay on the `v2d-weights` volume and are never copied |
| Host disk under the run root | `/e2e/runs/<name>` on local disk. Only the active run, the runs named on the command line, and `/e2e/inputs` are checked out at start. They are checkpointed every 10 minutes and checked in at exit, with mtimes preserved so `tools/e2e` resume fingerprints stay valid. Any sync error aborts the invocation rather than risk deleting volume data |
| `--dev` (mount `reconstruction/modules` over `/workspace`) | A no-op. Modal builds every module image from the current checkout at launch, so `/workspace` already holds that code, along with the compiled artifacts that docker's anonymous volumes preserved |
| Container environment | Commands get the image's own `PYTHONPATH` and `WORKDIR`, both recorded at build time, plus the `-e` flags. Modal's runtime variables are dropped |

Only one `run_e2e.sh` invocation may run at a time. A `modal.Dict` lock enforces this,
because two concurrent check-ins would clobber each other.

## One-time setup

```bash
export MODAL_PROFILE=<profile>          # pin it; never rely on `modal profile activate`
cd ~/Desktop/code/ego_intellect

# Secrets: HF token with access to the gated facebook/sam-3d-objects (+ GR00T base model)
modal secret create v2d-hf HF_TOKEN=<token>

# Licensed inputs (MANO is never in the repo or an image)
modal volume create v2d-weights
modal volume put v2d-weights MANO_LEFT.pkl  mano/models/MANO_LEFT.pkl
modal volume put v2d-weights MANO_RIGHT.pkl mano/models/MANO_RIGHT.pkl

# Your own videos or bundles (optional; repo assets also work). Anything under fs/e2e/ on the
# volume is checked out to local disk at /e2e/ in every role container, so it behaves like a
# host file. Avoid passing /vol/... paths directly: an open volume file blocks the
# commit/reload calls that sync module outputs.
modal volume create v2d-e2e
modal volume put v2d-e2e my_demo.mp4 fs/e2e/inputs/my_demo.mp4     # → /e2e/inputs/my_demo.mp4
```

Isaac-GR00T is pinned to the `n1.7-release` tag commit (`23ace64`), which contains every `gr00t`
module the pipeline calls. Override with `V2D_GROOT_REF`.

Set a **usage limit** on the workspace at `modal.com/settings/<workspace>/usage`. That
server-side cap is what actually stops spend; this package only sets per-call timeouts.

## Smoke ladder (run in order; each step is cheaper than the next)

All commands go through the wrapper. It pins the profile, sets a server-side timeout for
each subcommand, and detaches, so closing the laptop does not kill the run.

```bash
W=tools/modal_v2d/run_e2e_modal.sh
RUN=/e2e/runs/tissue_box                 # run roots must live under /e2e/runs/

# 0a. Build all 14 images (CPU builders; cached afterwards). One subprocess and one log per
#     image (.e2e/modal_build_logs/), a progress table every minute, and automatic retries
#     for transient network errors only. `modal run` would build them all at once instead,
#     and a single PyPI-mirror timeout aborts every build.
$W --build

# 0a'. CPU-only import check of every module image (catches missing shared libraries,
#      e.g. open3d -> libusb, before any GPU is rented; cents)
$W --smoke

# 0b. Platform gate (one L40S, ≤45 min cap): nvidia-smi, Vulkan ICD, Isaac Kit headless boot
#     with cameras, Isaac deps, GR00T env. If this fails, nothing in the rg role can work.
$W --doctor

# 1. Control-role plumbing (CPU only)
$W init --run-root $RUN --sequence-id tissue_box --embodiment-contract vega_sharpa_joint \
   --task-profile /workspace/video_to_data/robotic_grounding/groot_finetune/task_profiles/tissue_box_lift_hold.json
$W status

# 2. Setup: host wrappers, weight downloads into v2d-weights (one GPU container per module)
$W setup --mano-dir /weights/mano --isaac-groot-dir /opt/Isaac-GR00T --accept-nvidia-model-eula
$W doctor

# 3. Reconstruction (each module step = one GPU call; artifacts land in $RUN/reconstruction)
$W reconstruct --video /workspace/video_to_data/reconstruction/assets/tissue_box.mp4 \
   --hand-tracking hamer --run-gsplat-refinement --dev
$W inspect

# 4. Robot side
$W retarget --embodiment vega
$W simulate
V2D_MODAL_TIMEOUT_H=1 $W train-expert --embodiment vega --max-iterations 50 --num-envs 128   # smoke first
```

Apart from `--run-root` and the input paths, the arguments are exactly those in
`docs/e2e_workflow.md`. Inside containers, the repo is at `/workspace/video_to_data`, MANO
at `/weights/mano`, and Isaac-GR00T at `/opt/Isaac-GR00T`.

**Before the full `train-expert` run** (20,000 iterations): measure seconds per iteration
from the 50-iteration smoke, then project hours × the L40S rate and compare that to the
workspace's month-to-date headroom. A single call can last at most 24 h. If the run does
not fit, use the bundled tissue-box ONNX expert, which `docs/e2e_workflow.md` documents:

```bash
$W collect --rl-checkpoint /workspace/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/policies/e2e_example/tissue_box/vega_sharpa_policy.onnx \
   --target-successes 10 --pilot-attempts 32 --collection-max-steps <computed> --render-envs 1
$W finetune --epochs 5 --global-batch-size 32
$W evaluate --checkpoint $RUN/finetune/checkpoint-<N> --episodes 1 --num-envs 1 --execution-length 4
```

## Batch mode: many clips per workspace

```bash
# manifest: tools/modal_v2d/batch.py docstring; results: docs/pipeline/OPTIMIZATION.md
MODAL_PROFILE=<profile> tools/modal_v2d/run_e2e_modal.sh --batch manifest.json --max-parallel 6
MODAL_PROFILE=<profile> python -m tools.modal_v2d.batch_report <batch_id>     # clips/h, waits, $/clip
python -m tools.modal_v2d.batch_shard manifest.json --profiles a,b --usd-per-clip 1.1   # several workspaces
```

One Modal app and (by default) one control container run every clip; a GPU ledger keeps the
batch under the workspace's 10-GPU cap and reuses idle containers across clips; calls with
no container after 30 s are hedged on a twin pool. Per-workspace setup (images, weights) is
done once; each clip runs `init`, `setup --skip-reconstruction`, `reconstruct`.

## Watching and retrieving results

Every invocation prints formatted progress to its logs (what `modal app logs` and the
attached terminal show):

```
[v2d 10:12:03Z] ━━━━ reconstruct (control) invocation 1117-1012-ab3f | 6 stages | cap 14:12Z | ~$0.33/h for this container ━━━━
[v2d 10:12:40Z] ━━━━ ▶ stage 2/6 reconstruct.run ━━━━
  10:12:41Z │ [1/1] reconstruct the egocentric video: python ...
[v2d 10:12:55Z] → module v2d_moge (call 9c1e0d2a7b44, L40S): python run_moge.py ...
[v2d_moge 10:13:20Z] <the module's own output, tagged and timestamped>
[v2d 10:13:55Z] ♥ reconstruct | stage 1/6 | reconstruct.run 0:01:15 | module v2d_moge 0:01:00 | › 144/390 [00:12<00:20] | elapsed 0:01:52 | cap 14:12Z | ~$0.04
[v2d 10:15:10Z] ✔ module v2d_moge exit=0 in 0:02:15 (~$0.08)
[v2d 10:41:02Z] ━━━━ ✔ reconstruct.run done in 0:28:22 (2/6) | ~$1.21 so far ━━━━
```

- Stage banners come from the launcher's own `[RUN]`/`[DONE]`/`[SKIP]` markers. The stage
  total comes from a `--dry-run` of the same command, made before the real run starts.
- A heartbeat (`♥`) every 60 s shows the current stage and module call, the newest line from
  the remote module, any `n/m` counter (RSL-RL iterations, tqdm bars), an ETA, and the spend
  estimate.
- Spend figures are list-price **estimates** (`config.RATES`). The board below shows the
  workspace's real month-to-date billing next to them.

**Board** (one row per invocation, read from the live `v2d-e2e-progress` Dict, plus billing):

```bash
$W --board --watch            # refreshes every 60 s; STALE marks a run with no heartbeat for 3 min
$W --board --journal <invocation>   # the durable event log (/vol/progress/*.jsonl)
```

```bash
modal app list                                           # find the detached run's app id
modal app logs <app-id>                                  # live logs, including module calls
modal volume ls v2d-e2e fs/e2e/runs/tissue_box           # checkpointed every 10 min
modal volume get v2d-e2e fs/e2e/runs/tissue_box/reconstruction/result_slam_gravity_aligned ./bundle
modal volume ls v2d-e2e calls                            # full log of every module call
```

A run root on the volume is a normal run root. You can download one and open it with the
Track A tools in `docs/learning_path.md`, for example the viser replay.

If a run dies and leaves the lock behind, check `modal app list` to confirm nothing is
running, then pass `V2D_MODAL_BREAK_LOCK=1`. After any disconnect, run `modal app list` and
`modal app stop <id>` before you diagnose anything else.

## Where GPU time goes, and what this package does about it

| Source of cost on Modal | Mitigation |
|---|---|
| Weight downloads pass `--gpus all`, so each one rented an L40S (first run: 7 m 50 s of GPU time to reach a 22 s download) | Commands matching `C.CPU_ONLY_MARKERS` (`download_weights`) run on a CPU copy of the same image (`cpu_<image>`). `V2D_MODAL_DOWNLOADS_ON_GPU=1` restores the original behaviour |
| `download_ego_reconstruction_weights.sh` runs 11 downloads one after another | `setup` first lists the script's wrapper calls, without running them, and runs them **concurrently** (`prefetch.weights`, journaled as its own stage). The script's own pass then finds each identical `docker run` already done and returns at once. `V2D_MODAL_PREFETCH=0` turns this off |
| `setup` starts the E2E container with `docker exec <c> true` | Answered by the shim locally: no GPU container just to run `true` |
| Every invocation copied the repo, built a git index and pip-installed host wrappers (~80 s, on an L40S in the `rg` role) | Baked into the `control` and `rg` images (`bake.py`). A changed checkout rebuilds only those last layers, on a CPU builder |
| Module images rebuilt for unrelated reasons | Image builder pinned (`2025.06`); only edits under `reconstruction/modules` rebuild module images |

## Benchmarking: time and cost per step, stage and module call

Every invocation journals a benchmark record to `progress/*.jsonl` on the state volume:

| Level | Recorded |
|---|---|
| Step (one `run_e2e.sh` invocation) | wall time, status, stages done/skipped/failed, container size and list rate, Modal app id. The app is **tagged** `v2d_invocation`/`v2d_command`, so `modal billing report` gives its real billed cost |
| Stage | wall time, the role container's share of cost, module calls made, GPU utilisation (mean/peak) and peak GPU memory (sampled from `nvidia-smi` every 5 s on GPU roles) |
| Module call | total time split into push sync, startup (scheduling and cold start), volume reload, execution, commit, and pull; bytes moved; GPU utilisation and memory in the module container; estimated cost (billed only while the remote call runs) |
| Image build | per image and attempt: seconds, success, cached or not (`.e2e/modal_build_logs/builds.jsonl`) |

After the pipeline (or at any point):

```bash
$W --bench --run tissue_box      # → .e2e/bench/tissue_box/
```

This downloads the journals, pulls hourly tagged billing, and writes:
- `report.md`: per step, per stage, slowest module calls, invocations, builds, and flags
  for low-utilisation GPU calls.
- `stages.csv`, `calls.csv`, `invocations.csv`, `builds.csv`, `billing.csv`, `summary.json`.
- `charts.pdf`: light theme, one chart per page, covering stage wall times, estimated vs
  billed cost per step, module execution vs overhead, and GPU utilisation.

`cost_calibrated` scales each stage's estimate so a step's stages sum to its billed cost.
Billing is hourly and lags by up to a few hours, so re-run `--bench` later for final numbers.
A failed billing query is reported as UNKNOWN, never as $0.

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `V2D_MODAL_TIMEOUT_H` | per subcommand (0.5–24) | Server-side lifetime cap for the role container |
| `V2D_MODAL_MODULE_TIMEOUT_H` | 2 | Cap for each reconstruction module call |
| `V2D_MODAL_RG_GPU` / `V2D_MODAL_MODULE_GPU` | `L40S` | Keep both on RT-core GPUs. FoundationPose TensorRT engines are tied to a GPU architecture |
| `V2D_MODAL_SYNC_INTERVAL_S` | 600 | Interval between checkpoints of `/e2e` to the volume |
| `V2D_MODAL_PREFETCH` / `V2D_MODAL_PREFETCH_PARALLEL` | `1` / `12` | Run setup's weight downloads concurrently first |
| `V2D_MODAL_DOWNLOADS_ON_GPU` | unset | `1` sends weight downloads to the GPU functions, as docker would |
| `V2D_GROOT_REF` | `23ace64` (n1.7-release) | Isaac-GR00T commit baked into the rg image |
| `V2D_MODAL_SECRETS` | `v2d-hf` | Comma-separated secret names attached to every function |
| `V2D_MODAL_DETACH` | 1 | Set to `0` to keep the local client attached |

## Known risks and limitations (unverified until the smoke ladder runs)

1. **Isaac Sim rendering on Modal.** Modal's own Isaac Lab example uses L40S, and its
   `play.py` records video, so this is expected to work. Isaac Lab 2.3.2 with
   `--enable_cameras` has not been checked yet. Step 0 checks it.
2. **Isaac-GR00T install.** The image runs `uv sync --python 3.10` in the checkout. If the
   GR00T revision you pin needs extras or compiled kernels, edit the GR00T `run_commands` in
   `images.py`.
3. **Every module call uses a GPU container**, including weight downloads and CPU-only
   renders. Containers scale down after 60 s idle.
4. **Image rebuilds.** Each module Dockerfile runs `COPY . /workspace` over all of
   `reconstruction/modules`, exactly as `docker build` does. Any edit there therefore
   rebuilds the later layers of every module image on the next launch.
5. **Logs arrive after each call finishes.** Module-call output reaches the launcher log
   only once the call completes. Use `modal app logs` to watch it live.
6. Real `docker` features outside what the repo uses are rejected loudly: `--mount`,
   `--env-file`, published ports, named volumes, and others.
7. Mount points are symlinks, so a path such as `<mount>/..` resolves to the volume rather
   than to the container directory around the mount. The repo passes absolute
   `/data/<arg>/...` paths, so this is only a concern for new module code.
