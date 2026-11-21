# Task/reward spec (`task_spec/v1`)

`task/task.json` in each environment package says what a lab's RL run optimises and how
success is judged. It is simulator-neutral:

- every term names its reference implementation (`function`, as `module:name`) and says what
  it computes (`definition`);
- the Isaac Lab exporter wires the same terms;
- the MuJoCo exporter re-implements them.

Schema: `tools/envpkg/schemas/task_spec_v1.schema.json`. Builder: `tools/envpkg/task.py`.

## Where it comes from (nothing is typed by hand)

1. `simulate` already builds the clip's Isaac Lab training environment for the physics QA
   replay. `replay_metrics.py` now also writes `RUN_ROOT/simulate/vega/env_spec.json`. That
   file is the training configuration exactly as Isaac Lab resolved it for this clip, taken
   **before** the QA replay's own overrides. It contains:
   - rewards, terminations, actions, observations, resets, assist curriculum, robot actuators,
     timing;
   - the built environment's dimensions and term lists.
2. The clip's task profile (from the run configuration) supplies the success evaluator.
3. The package builder combines the two into `task/task.json` and keeps the raw dump next to
   it as `task/<robot>/isaaclab_env_spec.json`.

## What the builder refuses

- **A term without a definition** in `task.py`'s catalogue. A recipe change cannot reach a
  package undescribed.
- **A success criterion the clip's own reference motion fails.** That would be a task no
  policy can pass by tracking the reference.
- **A runtime dump that failed, or timing that does not add up**, for example
  `step_dt` ≠ `sim_dt × decimation`, or a reference length that disagrees with Isaac Lab's.

## Contents (one task; one section per robot)

| Field | Content |
|---|---|
| `task_id`, `instruction`, `object`, `profile` | e.g. `tissue_box-b16e171e14.tissue_box_lift_hold`, "lift the box and hold it" |
| `success` | Evaluator (`lift_hold`), its parameters and a definition. A self-contained implementation is in `tools/envpkg/success.py`; it matches `groot_finetune`'s evaluator on 200 test traces and is checked in the tests |
| `robots.<robot>.reference` | Motion path, 50 fps, 649 frames, playback speed 0.5, so **519 control steps** at 20 Hz |
| `timing` | Physics 0.01 s × decimation 5 = control 0.05 s |
| `action` | 58 joints: target = reference joint position + 0.15 × EMA(clip(a, ±1), 0.3), clipped to joint limits |
| `observations` | 14 terms, 445 dims, with per-term dims and ±0.01 uniform noise where configured |
| `rewards` | 11 terms with weights: termination −100; object keypoints 2.0 (var 0.1); hand keypoints 0.5 (std 0.775); force closure 5.0; action rate −5e-4; action L2 −5e-3. Five contact terms are present at weight 0 |
| `terminations` | Reference end (time-out); wrist > 0.15 m; object > 0.10 m or > 0.5 rad from the reference; divergence (joint speed > 100 rad/s) |
| `reset` | Uniform random start frame, 50-step freeze, finger openness 0.7, all noise ranges 0 |
| `assist` | Virtual object controller (PD wrench toward the reference pose, stiffness 50/10, force/torque cap 60). The curriculum anneals its scale 1.0 → 0.0 in 11 linear stages over 12,500 PPO updates |
| `physics` | Robot gravity off; hand friction 2.0 (object at the simulator default 0.5); robot–object collisions on, robot–support off; object solver 8/0; arm and finger actuators |
| `episode.train` | From the reset distribution until the reference ends (time-out) or a failure termination. There is no length cap, because no `time_out` term is configured. The recipe's `episode_length_s` = 20 s is recorded but not enforced: Isaac Lab ends episodes only through termination terms. The builder sets a cap only when the environment has a `time_out` term |
| `episode.evaluation` | Frame 0, assist 0, 519 steps. A pass needs no non-timeout termination before the reference ends **and** the success criterion |
| `reference_check` | The success criterion applied to the reference itself. tissue_box lifts 0.227 m, holds for 224 steps and is placed back 4 mm above its start: **pass** |
| `notes` | Anything a lab must know (below) |

## Notes for labs

`notes` lists anything a lab must know about a clip's task. One example is a training
episode cap shorter than a frame-0 pass over the reference. The tissue_box clips have none.
Their episodes run to the reference end (519 steps plus the 50-step freeze).

## Validation

`python3 -m tools.envpkg validate` checks the task spec whenever a package has one:

- the schema;
- that the files exist;
- that every definition is exactly the catalogued one;
- that the reference motion, frame count and rate match the package motion;
- that control steps match the motion length at the playback speed;
- that the evaluation horizon equals the reference;
- that observation term dims sum to the observation dim, and that action dim = joint count;
- the reference check, **recomputed** from the motion file.

`--strict` (delivery) requires a task spec.

Exporters read the spec through `tools.envpkg.task.load_task_spec(package)`.

## Verified

| Check | Result |
|---|---|
| `replay_metrics.py` with the dump, on Modal (L40S, CPU physics), clips b3_tbseed and b3_tb2 | Both runs exit 0 with the same QA verdicts as before (pass, completion 0.330 / 0.328). The dump is 29 KB with 445 observation dims, 58 action dims and 519 reference steps |
| Packages for both clips | Build and validate. The reference check passes for each |
| Unit tests | 19 envpkg tests, including the episode cap derived from the terminations, including the task spec, refusals (reference that never lifts, unknown reward term, no environment dump) and a hand-edited definition being caught |
