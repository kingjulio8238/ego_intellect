"""CPU-only import smoke for every module image: import each `v2d.<module>.lib.*` module
the pipeline can call, and report failures (missing shared libraries, missing packages).

Catches the class of failure that just cost a reconstruct attempt (open3d ->
libusb-1.0.so.0 missing in the SAM3D image) before any GPU is rented.

    MODAL_PROFILE=<p> tools/modal_v2d/run_e2e_modal.sh --smoke [image ...]
"""

import sys
import threading

import modal

from tools.modal_v2d import config as C
from tools.modal_v2d.images import build_images

SCRIPT = r"""
cd /tmp
py=python; command -v python >/dev/null || py=python3
$py - <<'PY'
import importlib, os, sys, traceback, glob
name = os.environ["V2D_MODULE"]
pkg = name[len("v2d_"):]
roots = [p for p in (f"/workspace/{name}/lib", f"/workspace/v2d_{pkg}/lib") if os.path.isdir(p)]
mods = sorted({os.path.splitext(os.path.basename(f))[0] for r in roots for f in glob.glob(r + "/*.py")} - {"__init__"})
ok, bad = 0, []
for stem in mods:
    if stem.startswith("test"):
        continue
    mod = f"v2d.{pkg}.lib.{stem}"
    try:
        importlib.import_module(mod)
        ok += 1
    except BaseException as exc:  # noqa: BLE001
        last = traceback.format_exception_only(type(exc), exc)[-1].strip()
        bad.append((mod, last[:220]))
print(f"IMPORTS {name}: {ok} ok, {len(bad)} failed (of {len(mods)})")
for mod, err in bad:
    print(f"  FAIL {mod}: {err}")
PY
"""


# robotic-grounding image: every script and tool module tools/e2e calls, via the same Isaac
# `python` wrapper, with --help (imports run; Isaac Kit does not start). Plus the GR00T env.
RG_SCRIPT = r"""
cd /workspace/video_to_data/robotic_grounding
ok=0; bad=0
for s in scripts/retarget/ego_recon_to_dexmate_sharpa.py scripts/retarget/ego_recon_to_sharpa.py \
         scripts/retarget/vis_retargeted.py scripts/reconstruct_support_surfaces.py scripts/generate_rigid_urdfs.py \
         scripts/rsl_rl/dummy_agent.py scripts/rsl_rl/train.py scripts/rsl_rl/export_parallel_rollouts.py \
         scripts/rsl_rl/replay_record.py; do
  # Imports succeeded if argparse got to print usage (some scripts parse required args
  # before honouring --help) and no traceback appeared.
  timeout 300 /opt/isaac/bin/python $s --help > /tmp/o.txt 2>&1
  if grep -q "usage:" /tmp/o.txt && ! grep -q "Traceback" /tmp/o.txt; then ok=$((ok+1)); else bad=$((bad+1)); echo "  FAIL $s: $(grep -E 'Error' /tmp/o.txt | tail -1 | cut -c1-200)"; fi
done
for m in groot_finetune.tools.select_successful_episodes groot_finetune.tools.plan_groot_run groot_finetune.tools.audit_groot_run groot_finetune.convert_to_gr00t; do
  if timeout 300 /opt/isaac/bin/python -m $m --help > /tmp/o.txt 2>&1; then ok=$((ok+1)); else bad=$((bad+1)); echo "  FAIL $m: $(grep -E 'Error' /tmp/o.txt | tail -1 | cut -c1-200)"; fi
done
if (cd /opt/Isaac-GR00T && timeout 300 uv run --no-sync python -c "import gr00t" > /tmp/o.txt 2>&1); then ok=$((ok+1)); else bad=$((bad+1)); echo "  FAIL gr00t env: $(tail -1 /tmp/o.txt | cut -c1-200)"; fi
echo "IMPORTS robotic-grounding: $ok ok, $bad failed (of $((ok+bad)))"
"""


def main() -> int:
    names = sys.argv[1:] or [*C.MODULE_IMAGES, C.RG_IMAGE]
    images = build_images()
    app = modal.App.lookup("v2d-import-smoke", create_if_missing=True)
    results: dict[str, str] = {}

    def run(name: str) -> None:
        try:
            script = RG_SCRIPT if name == C.RG_IMAGE else SCRIPT
            sandbox = modal.Sandbox.create(
                "bash", "-c", script, image=images[name], app=app, timeout=1800, cpu=4, memory=16384,
                env={"V2D_MODULE": name, "ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y"},
            )
            sandbox.wait()
            results[name] = sandbox.stdout.read() + sandbox.stderr.read()[-400:]
        except Exception as exc:
            results[name] = f"probe error: {type(exc).__name__}: {exc}"
        print(f"\n##### {name}\n{results[name]}", flush=True)

    threads = [threading.Thread(target=run, args=(n,)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    failed = [n for n, text in results.items() if "FAIL" in text or "probe error" in text or "IMPORTS" not in text]
    print(f"\n[v2d-smoke] {len(names) - len(failed)}/{len(names)} images import cleanly" + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
