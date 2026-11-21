# Environment package contract (`env_package/v1`)

What a lab receives per clip: one directory (archived at delivery), self-describing through
`manifest.json`. Machine-readable schema: `tools/envpkg/schemas/env_package_v1.schema.json`.
Validator: `python3 -m tools.envpkg validate <package> [--strict]`, which needs `numpy`,
`pyarrow`, `jsonschema`. The validator is also run by the builder before it publishes anything.

## Layout

```
manifest.json                         env_package/v1 (below)
core/                                 simulator-neutral, robot-agnostic
  reconstruction/result.npz           hands (MANO), object pose, camera; frame "world"
  reconstruction/manifest.json        reconstruction metadata (producer paths stripped)
  human/tracks.parquet                MANO hand tracks + joints + contacts + object pose; frame "tracks"
  object/<name>.obj, .mtl, textures   object mesh in metres (visual, also the collision source)
  object/<name>.urdf                  rigid body: inertial + visual/collision mesh
robots/<robot>/                       one folder per retargeted robot
  motion.parquet                      motion_v1/single_robot; frame "robot:<robot>"
  support_surfaces.json               support_surfaces/v1 (static surfaces the object rests on)
  retarget_report.json                retargeting errors and placement (producer paths stripped)
qa/<robot>/physics_replay.json        physics QA replay report (v2d_replay_metrics_v1)
qa/triage.json, qa/reconstruction.json   gate reports (GATES.md)
task/task.json                        task_spec/v1: rewards, terminations, success (TASK_SPEC.md)
task/<robot>/isaaclab_env_spec.json   the resolved Isaac Lab training environment it was built from
isaaclab/                             Isaac Lab export: env.json + the extension's dataset layout (ISAACLAB_EXPORT.md)
mujoco/                               MuJoCo export: env.json + one MJCF scene per robot (MUJOCO_EXPORT.md)
qa/<robot>/mujoco_replay.json         the second physics check (MuJoCo)
```

Every path inside the package, including the ones stored in parquet files (`raw_motion_file`,
`object_mesh_paths`, `object_urdf_paths`), is relative to the package root. A package has no
absolute paths, no `..`, and no symlinks.

## Manifest

| Field | Meaning |
|---|---|
| `env_id` | `<sequence_id>-<first 10 hex of sha256(result.npz)>`: unique per reconstruction |
| `source` | Video hash and size, the object prompt, the instruction, the task profile id |
| `provenance` | Commit and runtime-tree hash of each producing stage (`reconstruct`, `retarget`, `simulate`), stage parameters, and the builder commit with its dirty flag. `complete: false` lists what was missing |
| `frames` | Coordinate frames (below) |
| `core` | Frame count and rate of the reconstruction, and where each core file is |
| `core.object.physics` | Mass, centre of mass, inertia, friction. Each value carries its source: `mass_source` is `default_constant` (0.3 kg for every object today), `estimated` or `measured`. Friction is `physx_default` (0.5 / 0.5, restitution 0): the values the physics QA replay ran with |
| `robots.<robot>` | Robot model id plus the sha256 of its description file; frame; motion (rate, frames, joints); support surfaces; retarget report; QA status (`pass` / `flagged` / `not_run`), flag reasons, completion |
| `task` | `{"spec": "task/task.json", "schema": "task_spec/v1"}`, or `null` when the clip has no resolved training environment yet ([TASK_SPEC.md](TASK_SPEC.md)) |
| `simulators` | `isaaclab` and `mujoco` exports (`{"env": ..., "schema": ...}`) once a task spec exists |
| `qa.verdict` | `rejected` if a gate rejected; `flagged` if Isaac or MuJoCo flagged a robot's replay or the two disagree; `incomplete` if a check did not run; else `pass` |
| `files` | Every file except `manifest.json`, with byte size and sha256 |

## Coordinate frames

All frames are metres, right-handed, z up. `T_from_world` maps world points into the frame:
`p_frame = T @ p_world`.

| Frame | Definition | Used by |
|---|---|---|
| `world` | SLAM world, rotated so gravity is −z | `core/reconstruction/result.npz` (`*_to_world_transform`) |
| `tracks` | `world` translated so the object is centred in xy with its bottom at z = 0 (a pure translation) | `core/human/tracks.parquet` |
| `robot:<robot>` | The robot's placement world: robot root at the origin. This is the retargeter's `T_place` | `robots/<robot>/motion.parquet`, `support_surfaces.json` |

Rates: the core runs at the source video rate (30 fps for tissue_box). Robot motion is
resampled to the robot's control rate (Vega 50 fps) and spans the same duration, within one
source frame.

## What validation checks

1. Schema: the manifest and each `support_surfaces.json`.
2. Files: exactly the listed set is present; sizes and hashes match; no symlinks.
3. References: every path in the manifest, the URDF mesh references, the OBJ's materials and
   textures, and the paths inside the parquet files resolve to package files.
4. Frames: every `T_from_world` is a proper rigid transform.
5. Data agrees with the frames. The object positions in `tracks.parquet` and in each robot
   motion must equal the reconstruction's object positions mapped through the declared
   transform (robot motion resampled to its rate), within 1 mm. For tissue_box they agree to
   6e-8 m, so a mismatched or edited file fails.
6. Data sanity: `result.npz` keys and shapes, MANO track shapes, finite joint trajectories,
   matching durations, a symmetric positive-definite inertia, URDF mass equal to the manifest,
   and an object extent between 5 mm and 2 m (catches unit errors).
7. QA: each robot's status agrees with its replay report, and `qa.verdict` agrees with the
   robot statuses.

`--strict` adds what delivery requires: complete provenance, verdict `pass`, a clean builder
commit, a known source-video hash, and a task spec. Warnings (not errors) currently mark the constant
default mass and the collision mesh being the visual mesh. Each of
these is an open item for a later stage.

## Building

```
python3 -m tools.envpkg build --run-root <run_root> --out <package> [--robot vega_sharpa]
```

The builder reads the run-root layout the e2e stages write (`tools/e2e/stages.py`). It refuses
the following unless the matching override is passed:

- a clip whose physics QA flagged it (`--allow-flagged`);
- a run root without `run_manifest.json` / `e2e_config.json` (`--allow-missing-provenance`,
  which gives a package that fails `--strict`);
- a tracks file that is not a pure translation of the world (no override);
- a support USDA with shapes other than z-axis disks with translate only (no override).

Robots need a packaging rule in `tools/envpkg/build.py` (`ROBOTS`). Today that is
`vega_sharpa`; the floating Sharpa hand is added once its motion has a declared frame.

Checked on two real clips (b3_tbseed, b3_tb2: Modal outputs, Vega): both build and validate.
Each package is 4.7 MB.
