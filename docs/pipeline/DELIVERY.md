# Delivery (stages 11–12)

Each clip goes through three steps: **export** makes its package, an **archive** freezes the
package, and **deliver** collects the accepted archives with a manifest and a verifier labs run.

Code: `tools/envpkg/delivery.py` and `tools/envpkg/verify_env.py`.

## Export: `run_e2e.sh export` (control container, CPU, after `simulate`)

1. Build the package: the core, the gate verdicts, the task spec, the Isaac Lab export, and the
   MuJoCo export with its physics check.
2. Validate it in **strict (delivery) mode**. That requires complete provenance, a clean source
   commit, the source-video hash, a task spec, and the verdict `pass`.
3. Archive it to `RUN_ROOT/package/<env_id>.tar.gz` and write `RUN_ROOT/qa/export.json`.

If the QA rejected or flagged the clip (gates, Isaac, MuJoCo, or the two disagreeing), there is
no package. The command records `verdict: reject` and exits **3**, which counts as a result,
the same as for the gates. Any other problem is a failure (exit 2). A typical one is a run
launched from a dirty working tree, which strict mode refuses.

`python3 -m tools.envpkg export --run-root R --no-strict` is for development only. It skips the
delivery checks and allows a run without run records.

## Archive

`<env_id>.tar.gz` holds one top-level `<env_id>/` folder. Entries are sorted, times and owners
are zeroed, and modes are normalised, so **the same package always gives the same bytes**: two
archives of tissue_box have identical sha256. tissue_box takes 6.9 MB compressed (10 MB
unpacked).

## Deliver

In a batch this runs automatically after the exports. It can also be run by hand:

```
python3 -m tools.envpkg deliver <run_root>... --out <dir> --delivery-id <id> [--external]
```

```
<delivery>/
  <env_id>.tar.gz ...         the accepted archives (copied; sha256 checked against the export)
  SHA256SUMS                  sha256sum format
  delivery.json               delivery/v1: environments (task, robots, simulators, QA), skipped clips and why, robot asset licences
  README.md                   verify + use, for the lab
  verify/verify_env.py        the verifier, plus the validator it needs (verify/envpkg/)
```

**Batch mode** (`--batch manifest.json`):

1. after the robot phase, every simulated clip the physics QA did not flag is exported in its own
   control container;
2. the accepted clips are delivered to `/e2e/deliveries/<batch_id>` on the state volume (fetch
   it with `modal volume get v2d-e2e fs/e2e/deliveries/<batch_id>`);
3. `pipeline.json` gives each clip an outcome: `delivered`, `rejected` (a gate), `flagged`
   (physics QA) or `failed`. Only `failed` fails the batch.

The manifest keys `export`, `deliver` and `external` switch the steps.

**Upload: none yet (decided 2026-11-21).** The delivery folder is the hand-off. An upload
adapter (Hugging Face, S3/R2 presigned URLs) is added when the first lab names its channel.

**Robot asset licences (decided 2026-11-21).** `--external` (manifest `external: true`) refuses
any robot whose asset redistribution `tools/envpkg/robot_assets.json` does not confirm. Today
that is Vega: the Dexmate Vega meshes have no recorded licence. Internal deliveries are not
gated, and `delivery.json` records the licence status of each robot either way.

## Verify (what a lab runs)

```
python3 verify/verify_env.py <env_id>.tar.gz [--mujoco] [--isaaclab --rg-root <robotic_grounding> --isaac-python <cmd>]
```

1. The archive's sha256 is checked against `SHA256SUMS` beside it, when present.
2. The archive is unpacked safely: absolute paths, `..`, links and devices are refused, and it
   must hold one top-level folder.
3. The full package contract is checked in strict mode.
4. With `--mujoco`, the MuJoCo physics replay is re-run and compared with the recorded one.
5. With `--isaaclab`, the Isaac Lab replay is re-run (see ISAACLAB_EXPORT.md).

The verifier needs Python 3.10+, numpy, pyarrow and jsonschema; `--mujoco` also needs mujoco.
It never writes into the package it checks. Every package carries the same verifier as
`verify/verify_env.py`, so a single package can be checked on its own.

## Verified

- **Unit tests** (`tools/envpkg/tests`, 28):
  - export accepts a clean clip and refuses a flagged one (exit 3 / `--exit-zero`);
  - archives are deterministic and verify;
  - `../` and absolute-path archives are refused;
  - delivery copies accepted clips only and records why the others were skipped;
  - one changed byte fails the checksum;
  - a non-empty delivery folder is refused;
  - external delivery refuses the unconfirmed Vega assets.
- **Two real clips** (b3_tbseed and b3_tb2, local run roots): both export (6.86 and 6.89 MB), are
  delivered, and the delivered verifier runs from a directory with no repository on the path.
  Strict verification reports exactly the missing run records, dirty tree and video hash of
  these locally assembled runs. A Modal run with full provenance from a clean commit (build
  step G) is the strict end-to-end proof.
