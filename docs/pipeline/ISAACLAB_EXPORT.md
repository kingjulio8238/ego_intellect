# Isaac Lab export (`isaaclab_env/v1`)

The package's `isaaclab/` folder makes the environment runnable in Isaac Lab with the
**unmodified** robotic_grounding extension (the code that built and QA'd it). The package
builder writes it whenever the clip has a task spec. Code: `tools/envpkg/isaaclab.py`.

## Layout

```
isaaclab/env.json                                   isaaclab_env/v1 (below)
isaaclab/human_motion_data/                          → HUMAN_MOTION_DATA_DIR
  ego_recon/processed/sequence_id=<seq>/robot_name=<robot>/data.parquet
  ego_recon/processed/<object>.urdf .obj .mtl textures
  ego_recon/reconstructed_stage/<seq>_<robot>_support.usda
```

This is the extension's native dataset layout:

- **Motion:** it is the package's core robot motion with its asset paths re-pointed (relative
  to `HUMAN_MOTION_DATA_DIR`). Every other column is identical, and the validator checks this.
- **Object files:** they are copies of `core/object/`, about 2 MB. The extension finds object
  assets by file name beside the motion, so the copy saves changing its loader. The validator
  checks that the copy's hashes are identical.
- **Support USDA:** it is generated from the simulator-neutral `support_surfaces.json`. The
  validator parses it back and requires the same disks.

## `env.json`

Per robot it records:

- the Isaac Lab task id (from the task spec);
- the `--motion_file` argument;
- the robot description's sha256;
- the env dump the task spec was built from;
- the physics-QA verdict the clip got in `simulate` (`expected_replay`);
- ready-to-run commands: `replay_check`, `train` and `play`.

It also records the extension's repository path and revision. Run the commands from the
extension's checkout root (`robotic_grounding/`) inside its Isaac Lab image:

```
HUMAN_MOTION_DATA_DIR=<package>/isaaclab/human_motion_data \
  python scripts/rsl_rl/train.py --headless --task VegaSharpa-WholeBody-Manip-v0 \
  --motion_file ego_recon/processed/sequence_id=tissue_box/robot_name=vega_sharpa --num_envs 4096
```

## Acceptance check: `tools/envpkg/isaaclab_check.py`

The check works on a scratch copy of the package. For each robot it runs the export's own
`replay_check` in a fresh Isaac Lab process and requires three things:

1. the robot description Isaac Lab actually loaded hashes to the package's robot model hash;
2. the resolved training environment equals the dump the task spec was built from: rewards,
   terminations, actions, observations, events, curriculum and command, plus the runtime
   dimensions and term lists. Machine paths are ignored;
3. the physics-QA replay reaches the clip's `simulate` verdict: the same flag, completion
   within 0.02, and each check within 5 mm (or 0.05 for fractions). Isaac's CPU physics varies
   by up to about 1.5 mm between containers.

It also reports any file Isaac wrote into the package copy (none so far).

```
python -m tools.envpkg.isaaclab_check <package> --rg-root <robotic_grounding checkout> \
    --isaac-python "/workspace/isaaclab/isaaclab.sh -p" --out report.json      # exit 0 = matches
```

## Verified (2026-11-21, Modal L40S, robotic-grounding image, julian-41138, about $0.2)

| Package | Result | Replay |
|---|---|---|
| b3_tbseed | **OK**: environment identical, robot hash matches | no flag, completion 0.3301 (package: 0.3301); largest check difference 1.2 mm |
| b3_tb2 | **OK** | no flag, completion 0.3282 (package: 0.3282) |
| b3_tbseed with its support disk raised 4 cm (a deliberate defect) | **FAILED**, as it should | flagged: the hands terminate at step 0; completion 0.002 |

Each check took 19–31 s, most of it Isaac start-up. No files were written into the packages.

## Limits

- **A lab needs the robotic_grounding extension** at the recorded revision (or one that
  reproduces the same environment), because the MDP terms live there. The acceptance check
  is how to confirm a different checkout still matches.
- **The robot description is not copied into the package.** The extension ships it, and the
  hash proves it is the same file.
- **Vega only.** The floating Sharpa hand follows once its motion has a declared frame (see
  ENV_PACKAGE.md).
