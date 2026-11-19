# V2D learning path — run it, inspect it, then extend it

This guide is a hands-on route through the Video-to-Data (V2D) repository. Each step tells you
what to run, which artifact to open, and what you should see there.

**Track A** runs on this Mac (Apple M2, no NVIDIA GPU). Every command in it was executed on
2026-11-14 and the stated results were observed. **Track B** needs a Linux x86 machine with one
48 GB NVIDIA GPU (A6000, L40 or L40S; Blackwell `sm_120` is not supported). Its commands come from
`--dry-run` output and the code, and have not been executed here.

---

## 0. The mental model: one spine, eight data contracts

The pipeline is a sequence of files on disk. Every stage reads the previous stage's file and
writes its own. To understand V2D, learn what each of these files contains and which code writes
and reads it.

```
iPhone mp4 ─(reconstruction, ~14 Docker modules)─► result_slam_gravity_aligned/   [A] result.npz + mesh.obj + manifest.json + threejs_scene/
    │                                                     │
    │  (optional, independent) video_ingestion_agent      ▼ loader image (MANO FK, GPU)
    └─► graph.db + vector.db  (clip search)        ego_recon_loaded parquet          [B]
                                                          ▼ robotic-grounding image (Pink/Pinocchio IK)
                                     motion_v1 parquet + support .usda + QA report    [C]
                                                          ▼ Isaac Lab RSL-RL PPO (VegaSharpa-WholeBody-Manip-v0)
                                                   expert model_*.pt  (or bundled ONNX) [D]
                                                          ▼ collect (rollouts → select → replay with 3 cameras)
                                                     recording/data.h5 + semantic_audit [E]
                                                          ▼ convert_to_gr00t
                                                    LeRobot gr00t_dataset/ + stats     [F]
                                                          ▼ GR00T N1.7 finetune (external Isaac-GR00T venv)
                                                 checkpoint-N + open_loop/trajectory_0.png [G]
                                                          ▼ closed-loop eval (GR00T server + Isaac client)
                                                  closed_loop/evaluation_<N>.json + success videos [H]
```

| Boundary | Writer | Reader | Contract lives in |
|---|---|---|---|
| A `result.npz` | `reconstruction/modules/v2d_common/.../result_bundle.py` | loader `ego_recon_loader`, `ego_recon_to_dexmate_sharpa.py` | `result_bundle.py:673-793` (+ gravity pass `:446-562`) |
| C motion_v1 | `scripts/retarget/ego_recon_to_dexmate_sharpa.py` | Isaac Lab tasks, replay, support recon | `robotic_grounding/.../motion_schema/{schema,reader,writer}.py` |
| E/F/G | `groot_finetune/*` | GR00T | `groot_finetune/contracts.py` (`VEGA_SHARPA_JOINT` :287) |
| run state | `tools/e2e/runner.py` | `run_e2e.sh status` | `run_manifest.json`, `e2e_config.json` |

Three facts that are easy to miss:
- The launcher runs **reconstruct, setup and finetune on the host**, but **retarget, train,
  collect and evaluate inside the `robotic-grounding` container** through `workflow/run.sh e2e-run`.
- The ingestion agent is not part of `run_e2e.sh`. It is a separate front end for choosing *which*
  clips to reconstruct.
- `run_e2e.sh` skips a stage only when the stage's signature matches. The signature is a hash of
  its commands, inputs, outputs and the repository's runtime files. Inside `reconstruction`, stages
  skip when their output exists, and nothing is hashed. To rerun one of those, delete its output
  directory.

---

## Track A — on this Mac (verified)

### A1. Environment (one-time)

```bash
cd ~/Desktop/code/ego_intellect
./tools/learn/setup_mac_env.sh            # → ~/.venvs/v2d-learn (CPU torch, pinocchio, pink, viser, …)
source ~/.venvs/v2d-learn/bin/activate
export TMPDIR=$(python -c "import os,tempfile;print(os.path.realpath(tempfile.gettempdir()))")
```

`TMPDIR` must be a resolved path. macOS's `/var` is a symlink to `/private/var`. With the default
`TMPDIR`, 6 launcher tests fail on path comparison. That is a test artifact, not a bug in the
launcher.

### A2. Read the contracts through their tests (the fastest way to learn the spec)

```bash
python -m pytest -q -p no:cacheprovider tools/e2e/tests tests                    # 63 passed — launcher, resume, docker mounts
(cd video_ingestion_agent && python -m pytest -q -p no:cacheprovider tests)        # 381 passed, 1 skipped — ingestion + retrieval (models mocked)
python -m pytest -q -p no:cacheprovider --no-cov reconstruction/modules/v2d_common/tests   # 35 passed — result bundle, gravity alignment
(cd robotic_grounding && python -m pytest -q -p no:cacheprovider --no-cov groot_finetune)  # 100 passed — contracts, HDF5→LeRobot, evaluator
(cd robotic_grounding && PYTHONPATH=source/robotic_grounding python -m pytest -q -p no:cacheprovider --no-cov \
   tests/test_joint_limits.py tests/test_reference_plane.py tests/test_motion_schema*.py tests/test_replay_data.py)  # 73 passed
```

Then read the tests that matter most, in this order:
1. `reconstruction/modules/v2d_common/tests/test_result_bundle_gravity.py`: how the bundle is
   assembled, how object scale is baked into the mesh, and the z-up gravity frame.
2. `robotic_grounding/tests/test_motion_schema.py`: the motion_v1 fields every consumer relies on.
3. `robotic_grounding/groot_finetune/test_convert_to_gr00t.py`: the HDF5 → LeRobot mapping and
   the reordering of the 58 joints.
4. `tools/e2e/tests/test_workflow_run.py`: the exact `docker run` invocation, captured through a
   fake docker.

### A3. Walk the whole workflow without running it (`--dry-run`)

```bash
./run_e2e.sh init --run-root "$PWD/.e2e/tissue_box" --sequence-id tissue_box \
  --embodiment-contract vega_sharpa_joint \
  --task-profile "$PWD/robotic_grounding/groot_finetune/task_profiles/tissue_box_lift_hold.json"
cat .e2e/tissue_box/e2e_config.json | python -m json.tool | less     # every resolved workflow value
ls .e2e/tissue_box/contracts/                                        # snapshotted embodiment + task profile
./run_e2e.sh status
./run_e2e.sh --dry-run reconstruct --video "$PWD/reconstruction/assets/tissue_box.mp4" \
  --hand-tracking hamer --run-gsplat-refinement --dev
./run_e2e.sh --dry-run train-expert --embodiment vega --max-iterations 20000 --num-envs 128
```

Each dry run prints the real command, its working directory and the outputs it expects.
- `reconstruct` resolves to `run_ego_reconstruction.py … --undistort --run_droid_slam
  --run_gravity_alignment --export_threejs_result`.
- `train-expert` resolves to `scripts/rsl_rl/train.py --task VegaSharpa-WholeBody-Manip-v0
  --motion_file ego_recon/processed/sequence_id=tissue_box/robot_name=vega_sharpa`.
- `retarget`, `collect`, `finetune` and `evaluate` refuse to dry-run until `setup` has saved
  real MANO and GR00T paths, and dry-runs are not persisted. Read `tools/e2e/stages.py` for those
  stages: retarget is at :350-560, collect at :750-969 and finetune at :972.

`python reconstruction/modules/v2d_pipelines/run_ego_reconstruction.py --help` also works on
the Mac and lists every reconstruction knob.

### A4. Inspect the real end products already in the repo

No `result.npz` is checked in. The tissue-box example *downstream* of reconstruction is
checked in. It comes from the same 390-frame, 30 fps `reconstruction/assets/tissue_box.mp4`.

```bash
M=robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed/sequence_id=tissue_box_simple
python tools/learn/inspect_motion.py $M/robot_name=vega_sharpa    # motion_v1: 42 cols, 649 frames @ 50 fps, 58 joints
python tools/learn/inspect_motion.py $M/robot_name=sharpa_wave    # floating hands: ManoSharpaData, 62 cols, 390 frames @ 30 fps
ffprobe -v error -show_entries stream=width,height,r_frame_rate,nb_frames reconstruction/assets/tissue_box.mp4
cat robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/reconstructed_stage/tissue_box_simple_vega_sharpa_support.usda | head -60
cat robotic_grounding/groot_finetune/task_profiles/tissue_box_lift_hold.json
python -c "import onnxruntime as o; s=o.InferenceSession('robotic_grounding/source/robotic_grounding/robotic_grounding/assets/policies/e2e_example/tissue_box/vega_sharpa_policy.onnx'); print(s.get_inputs()[0].shape, s.get_outputs()[0].shape)"
```

What to check:
- **Frame counts.** Retargeting resampled 30 fps to 50 fps: 390 / 30 × 50 = 650, and the parquet
  has 649 frames.
- **Two schemas.** The floating Sharpa output still contains the raw `mano_*` inputs, so it is the
  best place to see the result.npz hands after MANO forward kinematics. The Vega parquet holds
  only robot-side data.
- **Joint order.** The Vega `robot_joint_names` order is L_arm, then L fingers (index, middle,
  pinky, ring, thumb), then R_arm, then R fingers. GR00T's order is R first, with the thumb first
  (`contracts.py:280`). The two are remapped by joint name.
- **The bundled expert.** Its input is `obs[1,445]` and its output is `actions[1,58]`, which are
  residual joint targets. Match these 445 dimensions against the observation terms in
  `tasks/v2d_whole_body/config/vega_sharpa/vega_sharpa_manip_env_cfg.py:105`.

### A5. Watch the retargeted robot motion (viser, CPU)

```bash
L=$(python tools/learn/localize_motion.py $M/robot_name=vega_sharpa /tmp/v2d_motion)   # fix container paths (see note)
(cd robotic_grounding && PYTHONPATH=source/robotic_grounding \
  python scripts/replay_motion_viser.py --motion_file "$L" --port 8091 --start-paused)
# open http://localhost:8091 — frame slider, play, contact markers
```

Note: the checked-in parquet stores absolute `/workspace/video_to_data/...` mesh paths, but
`viser_playback.py:171-179` only resolves repo-relative paths, so without the localize step the
box doesn't render. The reader also takes `sequence_id` and `robot_name` from the hive directory
names, so a bare `data.parquet` copy fails with `MissingRequiredField`.

### A6. The two CPU tutorials (the best conceptual material in the repo)

```bash
mkdir -p ~/v2d_tutorials && cd ~/v2d_tutorials
unzip -o ~/Desktop/code/ego_intellect/docs/v2d_corl_tutorial/exercises/exercise_1.zip
unzip -o ~/Desktop/code/ego_intellect/docs/v2d_corl_tutorial/exercises/exercise_3.zip
```
- **E1 (`corl_tutorial_e1_notebook/`): real-to-sim.** It covers three raw guesses that each have
  their own scale (MoGe-2 depth, SAM 3D Body, SAM 3D Objects), FoundationPose's pose candidates,
  CoCoNet with contact refinement against a multi-view reference, and the Track 2 export. It is
  the CARI4D third-person path rather than the ego pipeline, but the scale, tracking and contact
  problems are the same. Setup: `conda env create -f environment-e1.yml`, then
  `python check_setup.py --cells`.
- **Workshop (`V2D_workshop/`, Python 3.12, Newton physics on CPU, tested on Apple Silicon).**
  It covers kinematic replay, then dynamic replay that fails, then contact-wrench support, then
  the trained policy succeeding. This is the clearest explanation of *why* the RL stage exists:
  retargeted kinematics alone drop the object. `chord_workshop/{simulation,policy,contacts}.py`
  are small and readable.

### A7. Ingestion agent on the Mac (needs a VLM endpoint; not verified here)

The VLM (Cosmos3-Nano) needs vLLM on CUDA. SigLIP-2 embeddings run on CPU or MPS. The cleanest
setup is to run `serve.py` on the Track B box and tunnel to it:

```bash
ssh -N -L 8000:localhost:8000 gpu-box &       # vLLM from Track B
cd video_ingestion_agent && cp configs/ingestion.yaml configs/ingestion_mac.yaml
# edit ingestion_mac.yaml: models.device: "mps"  and  models.vllm_local_media: false
python scripts/run_ingestion.py ../reconstruction/assets/tissue_box.mp4 -c configs/ingestion_mac.yaml --no-verify -o runs/tissue
```

Inspect the results:
- `runs/tissue/clips_stage1.jsonl` and `clips_final.jsonl`.
- `report.html`.
- `outputs/graph.db` (`sqlite3 outputs/graph.db '.schema'`, then `select * from action_segments`).
- `outputs/vector.db`: the embeddings are pickled float32 arrays.

The databases go to `outputs/`, not to the run directory. Copy `retrieval.yaml` rather than
editing it, because `test_retrieval_config.py` pins its values.

---

## Track B — on a 48 GB NVIDIA box (from code and dry-runs; not executed here)

**Machine requirements.** A Linux x86_64 VM with Docker and the NVIDIA Container Toolkit. The
pipeline runs `docker run --runtime=nvidia` for every module. Assume about 200 GB of disk or more
for images, weights and recordings. **To use Modal instead**, see
[`tools/modal_v2d/README.md`](../tools/modal_v2d/README.md). It runs the same `run_e2e.sh`
stages, translating each `docker run` into a Modal function call. On Modal, run roots live at
`/e2e/runs/<name>`.

**Licences and credentials to prepare first:**
- MANO: `MANO_LEFT.pkl` and `MANO_RIGHT.pkl`.
- An `HF_TOKEN` with access to the gated `facebook/sam-3d-objects`.
- Acceptance of the NVIDIA Open Model License for FoundationPose.
- An Isaac-GR00T checkout with a working `uv run python`.

**Order of work.** Each stage is cheaper to check than the one after it. Inspect each stage's
gate before running the next.

| # | Run | Inspect (the gate) |
|---|---|---|
| B1 | `./run_e2e.sh init …` (as in A3); `./run_e2e.sh setup --mano-dir … --isaac-groot-dir … --accept-nvidia-model-eula`; `./run_e2e.sh doctor` | `doctor` must pass every check: LFS, GPU, docker, loader image, GR00T env, container mounts |
| B2 | `./run_e2e.sh reconstruct --video $PWD/reconstruction/assets/tissue_box.mp4 --hand-tracking hamer --run-gsplat-refinement --dev` | Walk `<run>/reconstruction/` stage by stage (table below). Read `logs/` if anything is missing |
| B3 | `./run_e2e.sh inspect`; open `reconstruction/result_slam_gravity_aligned/threejs_scene/index.html` | Hands touch the box, the box rests on a level floor, and +Z is up. Then load `result.npz` and check the keys against A2 |
| B4 | `./run_e2e.sh retarget --embodiment vega` (then `--embodiment floating` for comparison) | `vega_retarget_qa/<id>_vega_sharpa_report.json`: converged_pct, position_error_m (wrists, fingertips), and `warnings` empty. Watch the mp4. Replay the parquet with A5 |
| B5 | `./run_e2e.sh simulate` (dummy agent, 2 steps) | The scene builds and the object loads with its support |
| B6 | Smoke first: `./run_e2e.sh train-expert --embodiment vega --max-iterations 50 --num-envs 128` in a separate run root. Then the real one with `--max-iterations 20000` | TensorBoard under `train_expert/vega/logs/rsl_rl/vega_sharpa_whole_body/`: reward terms, episode length, termination mix |
| B7 | `./run_e2e.sh collect --rl-checkpoint <validated .pt> --target-successes 10 --pilot-attempts 32 --collection-max-steps <computed> --render-envs 1`. To skip B6, use the bundled ONNX expert | `source/pilot/manifest.json` (the pilot rate), `selected/manifest.json`, and `semantic_audit.json` with `ok: true` |
| B8 | `./run_e2e.sh finetune --epochs 5 --global-batch-size 32` | `dataset_audit.json` with `ok: true`, `gr00t_dataset/meta/stats.json` finite, `open_loop/trajectory_0.png` |
| B9 | `./run_e2e.sh evaluate --checkpoint <ckpt> --episodes 1 --num-envs 1 --execution-length 4`, then `--episodes 100` | `closed_loop/evaluation_<N>.json`: `success_rate`, `termination_reasons`, `episode_lengths`. Watch the success videos |

Compute `--collection-max-steps` from the B4 report as `ceil(frames / source_fps / 0.5 × 20)`,
plus a 10–20% margin.

**Reconstruction intermediates (B2).** These are listed in execution order and are all under
`<run>/reconstruction/`:

| Dir / file | Module | What to look at |
|---|---|---|
| `anycalib/`, `undistorted.mp4` | AnyCalib | Straightened lines; intrinsics JSON |
| `frames/` | host | Frame count matches ffprobe (390) |
| `depth/`, `intrinsics/`, `intrinsics_stable.json` | MoGe | uint16 inverse depth, `65535/(d+1)` |
| `slam_poses/`, `slam_trajectory.txt` | DROID-SLAM | A smooth camera path; scale comes from MoGe |
| `wilor_raw/`, `hand_detections.json` | WiLoR | Hand boxes (these seed the SAM2 prompts) |
| `dino_detections.json` | GroundingDINO | The object box on the reference frame |
| `masks/<1,2,3>/`, `masks_overlay.mp4` | SAM2 | Object = 1 and the hands = 2/3. This is the most common failure point |
| `mesh/`, `mesh_scaled.obj`, `scale.json` | SAM3D, FoundationPose | Mesh shape and metric size |
| `poses/`, `poses_smoothed/` | FoundationPose, EKF | Object-to-camera poses |
| `hamer*/` | HaMeR | MANO hands aligned to depth, with gaps filled |
| `*_refined_simple/`, `progress_simple.mp4` | gsplat | Joint refinement of object, hand and camera |
| `result/` → `result_slam/` → `result_slam_gravity_aligned/` | host, GeoCalib | The final bundle |

---

## Extending it: where to cut in

| You want to… | Touch | Re-verify with |
|---|---|---|
| Swap or add a perception model | New `reconstruction/modules/v2d_<x>/` with `docker/run_*.py` + `lib/`; see `reconstruction/README.md:758-808` and `reconstruction/CLAUDE.md`. Wire it into `v2d_pipelines/run_ego_wilor.py` | The intermediate dirs in B2 and the bundle keys |
| Change what reaches the robot side | `v2d_common/result_bundle.py` (bundle keys) plus `v2d_task_library_loader/lib/ego_recon_loader.py` | `test_result_bundle_gravity.py`, then B3/B4 |
| Change retargeting (IK weights, collision barriers, placement) | `scripts/retarget/mano_to_dexmate_sharpa.py:385` (`DexmateSharpaIK`), `ego_recon_to_dexmate_sharpa.py` | QA report errors and A5 replay |
| Add a robot/embodiment | URDF in `assets/urdfs/`, a retarget script, an Isaac Lab task in `tasks/`, and a new contract in `groot_finetune/contracts.py` | `test_contracts.py`, B5 |
| Change rewards, observations or terminations | `tasks/v2d_whole_body/config/vega_sharpa/vega_sharpa_manip_env_cfg.py` (+ `mdp/`) | Smoke train (B6). The bundled ONNX will no longer fit if the observations change |
| New task success criteria | `groot_finetune/task_profiles/*.json` + `closed_loop/evaluation.py` | `test_evaluation.py`, B9 |
| Ingestion prompts or schema | `ingestion/segmentation/prompts.py`, `entity_graph/prompts.py`, `database_writer.py` (+ `search_graph.py` and `webapp/services/database_service.py`, which read it) | The ingestion test suite, plus `report.html` |
| New source dataset | `robotic_grounding/.../retarget/dataset_registry.py` first, then a loader in `v2d_task_library_loader/lib/` | `workflow/data_pipeline.md` "Adding a New Dataset" |

---

## Documentation-versus-code discrepancies found while verifying

These are verified on the Mac:
- The checked-in tissue-box parquet uses absolute container mesh paths that `viser_playback`
  cannot resolve (see A5).
- The launcher tests fail on macOS unless `TMPDIR` is resolved (see A1).
- `groot_finetune` tests must be run from `robotic_grounding/`, because they use cwd-relative
  profile paths.

These come from reading the code:
- Only the `vega_sharpa` embodiment can go through collect, finetune and evaluate. The stages
  hard-code it (`stages.py:711, 820, 1211`).
- The retarget stage only checks that `vega_retarget_qa/` exists, not the files the documentation
  requires (`stages.py:555`).
- `--hand-tracking dynhamr` is accepted by the CLI, but `setup` only provisions hamer.
- Reconstruction's skip-if-exists checks are stale-prone: `result_slam/` is not recopied when
  `result/` changes.
- `result.npz` has no fps or timestamps. The fps comes from the video.
- `getting_started.rst` lists `clips_verified.jsonl`, but the code writes
  `verification_results.jsonl`.
