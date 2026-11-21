# MuJoCo export (`mujoco_env/v1`) and the second physics check

The package's `mujoco/` folder holds one self-contained MJCF scene per robot. It loads with
plain `mujoco.MjModel.from_xml_path`, so it works with MuJoCo, MJX or mujoco_warp. The builder
writes the scene, replays the clip in it on CPU, and records whether MuJoCo and Isaac agree.

Code: `tools/envpkg/mujoco_export.py` and `tools/envpkg/mujoco_replay.py`.

## Layout

```
mujoco/env.json                       mujoco_env/v1 (below)
mujoco/<robot>/scene.xml              robot + object + support + floor + actuators + options
mujoco/<robot>/assets/                robot collision meshes (2.2 MB for Vega) + Sharpa licence/notice
qa/<robot>/mujoco_replay.json         the second physics check's report
```

## How the scene matches Isaac

Every value comes from the package or from the clip's resolved Isaac Lab training environment
(`task/<robot>/isaaclab_env_spec.json`). Nothing is typed by hand.

| Aspect | Isaac Lab (training env) | MuJoCo scene |
|---|---|---|
| Robot | `vega_sharpa_reduced.urdf`, fixed base, convex-hull colliders | Same URDF, converted by MuJoCo's own URDF compiler; collision geometry only (MuJoCo collides on convex hulls) |
| Gravity on robot | Off (`disable_gravity`) | `gravcomp = 1` on every robot body |
| Servos | Arm: DelayedPD (sysid stiffness/damping, effort limits, armature, 4-physics-step delay). Fingers: implicit PD with stiffness, damping, effort, armature, joint friction | One position actuator per joint with the same kp/kd, force range, armature and friction loss. The arm delay is a runtime property (`env.json`), applied by the replay |
| Timing | 0.01 s physics, decimation 5, 20 Hz control, reference played at half speed | Same (`timestep` 0.01, implicitfast integrator, elliptic cones) |
| Object | Package mesh and URDF inertial (0.3 kg), free body | Core mesh (referenced, not copied), same mass, centre of mass and inertia, free joint, frame-0 reference pose |
| Support | Package support disks, static | Same disks as static cylinders, plus a floor |
| Friction | Hand 2.0 and object 0.5; PhysX averages, so the pair is 1.25 | MuJoCo takes the maximum, so hand geoms get 1.25 and everything else 0.5. Each pair gets the same coefficient |
| Collisions | Robot self-collisions on; robot–object on; robot–support off | Contact bitmasks give the same three rules |
| Not modelled | Arm joint velocity limit 2.4 rad/s; PhysX solver iteration counts | Listed in `env.json` |

## The second physics check (`mujoco_replay.py`)

This is the same experiment as Isaac's `replay_metrics.py`:

- start at reference frame 0 and drive every joint to the reference, with zero residual and the
  arm delay applied;
- no virtual object assist;
- stop at the task spec's terminations: object off by more than 0.10 m or 0.5 rad, or a wrist
  off by more than 0.15 m;
- compute the same checks with **Isaac's own thresholds** from the package.

There are two differences:

- **`grasp_missed_contact`:** Isaac measures the fraction of expected contact-wrench
  directions missing. MuJoCo measures the fraction of hands expected in contact that touch the
  object nowhere. Both use the same 0.5 limit.
- **`initial_penetration_m` (MuJoCo only, limit 10 mm):** this is how deep the object starts
  inside the support or the hands. PhysX pops such an object out, and Isaac's settle check then
  catches it. MuJoCo's soft contacts push it out gently, so the overlap is measured directly.

The kinematic grasp-drift check uses robotic_grounding's function on the package motion,
exactly as Isaac does.

**Agreement.** The two simulators must reach the same verdict, completion must be within 0.05,
and each position check (settle, rest, grasp, wrist) must be within 15 mm. A package where
MuJoCo flags the clip, or where the two disagree, gets the verdict `flagged`.

`python3 -m tools.envpkg.mujoco_replay <package> --check` re-runs the replay and compares it
with the recorded one. It is deterministic on CPU.

## Calibration (2026-11-21, CPU, 0.3 s per replay)

**Real clips:** MuJoCo against the Isaac verdict in the package.

| Clip | MuJoCo | Isaac | Largest check difference |
|---|---|---|---|
| b3_tbseed | pass, completion 0.322, terminated at step 166 (object) | pass, 0.330, step 170 (object) | settle 9.2 mm; others ≤ 2.9 mm; completion 0.008 |
| b3_tb2 | pass, 0.321, step 165 | pass, 0.332, step 171 | grasp 5.7 mm; others ≤ 5.0 mm; completion 0.012 |

MuJoCo settles the object slightly more on its support (settle 21–22 mm against 13–16 mm). That
is still inside the 25 mm limit, and it sets the 15 mm agreement margin.

**Defects:** the four physics-QA calibration defects (OPTIMIZATION.md, simulate), applied to
both clips. **All 8 are flagged in MuJoCo:**

| Defect | MuJoCo flags it through |
|---|---|
| Object 4 cm above its support | settle 65 mm, rest 95–98 mm (and grasp on tb2) |
| Object 2 cm into its support | initial penetration 20 mm |
| Object 5 cm sideways | settle 28–44 mm, initial penetration 27 mm |
| Object 0.5 s behind the hands | grasp drift 64 mm |

## Limits

- **Robot meshes.** The Sharpa hand meshes are Apache-2.0 (licence and notice copied). The
  source repository does not record a licence for the Dexmate Vega meshes: confirm
  redistribution rights before any external delivery.
- **The arm delay and the policy's residual are runtime, not MJCF.** A lab's MuJoCo
  environment applies them as `env.json` describes. The replay is the reference
  implementation.
- **No visual meshes or textures.** The scene is for physics; the object is drawn flat-coloured.
- **Vega only.** The good set is two tissue_box clips; recalibrate the agreement limits as more
  objects come through.
