#!/usr/bin/env bash
# Run one run_e2e.sh invocation on Modal.
#
#   MODAL_PROFILE=<profile> tools/modal_v2d/run_e2e_modal.sh <run_e2e.sh args...>
#   MODAL_PROFILE=<profile> tools/modal_v2d/run_e2e_modal.sh --build    # pre-build all images (progress table)
#   MODAL_PROFILE=<profile> tools/modal_v2d/run_e2e_modal.sh --smoke    # CPU import check of every module image
#   MODAL_PROFILE=<profile> tools/modal_v2d/run_e2e_modal.sh --doctor   # platform preflight
#   MODAL_PROFILE=<profile> tools/modal_v2d/run_e2e_modal.sh --board    # live status of runs + billing
#   MODAL_PROFILE=<profile> tools/modal_v2d/run_e2e_modal.sh --bench --run tissue_box   # time/cost report
#   MODAL_PROFILE=<profile> tools/modal_v2d/run_e2e_modal.sh --batch manifest.json [--max-parallel N]   # many clips
#   MODAL_PROFILE=<profile> tools/modal_v2d/run_e2e_modal.sh --probe [--images v2d_moge,...] [--rounds N]  # cold starts
#
# Environment:
#   MODAL_PROFILE            required; pinned on every modal command (never the global active one)
#   V2D_MODAL_TIMEOUT_H      server-side lifetime cap for this invocation (default per subcommand)
#   V2D_MODAL_DETACH=0       keep the local client attached (default: detached, survives disconnect)
#   V2D_MODAL_BREAK_LOCK=1   clear a stale state lock left by a dead invocation
#   V2D_MODAL_SECRETS        comma-separated Modal secret names (default: v2d-hf)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

: "${MODAL_PROFILE:?set MODAL_PROFILE explicitly; this wrapper never uses the global active profile}"
export MODAL_PROFILE
echo "[v2d-modal] profile: $(modal profile current) (MODAL_PROFILE=${MODAL_PROFILE})"

if [[ $# -eq 0 ]]; then
  sed -n '2,/^set -euo/p' "$0" | sed '$d'
  exit 2
fi

PY="${V2D_MODAL_PYTHON:-python3}"
"${PY}" -c "import modal" 2>/dev/null || { echo "error: ${PY} cannot import modal; set V2D_MODAL_PYTHON" >&2; exit 2; }
export PYTHONUNBUFFERED=1

case "$1" in
  --build) shift; exec "${PY}" -m tools.modal_v2d.build "$@" ;;
  --board) shift; exec "${PY}" -m tools.modal_v2d.board "$@" ;;
  --bench) shift; exec "${PY}" -m tools.modal_v2d.bench "$@" ;;
  --smoke) shift; exec "${PY}" -m tools.modal_v2d.smoke "$@" ;;
  --doctor) exec modal run -m tools.modal_v2d.app::doctor ;;
  --probe) shift; exec modal run -m tools.modal_v2d.app::startup_probe "$@" ;;
  --batch)
    shift
    manifest="${1:?--batch needs a manifest path}"; shift
    export V2D_MODAL_CONTROL_TIMEOUT_H="${V2D_MODAL_TIMEOUT_H:-4}"
    # Per-image scaledown windows are function settings, fixed when the app is defined.
    V2D_MODAL_SCALEDOWN_PER_IMAGE="$("${PY}" -c 'import json, sys; print(json.dumps((json.load(open(sys.argv[1])).get("options") or {}).get("scaledown") or {}))' "${manifest}")"
    export V2D_MODAL_SCALEDOWN_PER_IMAGE
    flags=(--detach); [[ "${V2D_MODAL_DETACH:-1}" == "0" ]] && flags=()
    echo "[v2d-modal] batch ${manifest}: server-side timeout ${V2D_MODAL_CONTROL_TIMEOUT_H}h per clip and for the driver"
    exec modal run ${flags[@]+"${flags[@]}"} -m tools.modal_v2d.app::batch --manifest "${manifest}" "$@" ;;
esac

# Subcommand (skip global flags) -> default server-side cap in hours.
subcommand=""
skip_next=0
for arg in "$@"; do
  if [[ $skip_next -eq 1 ]]; then skip_next=0; continue; fi
  case "$arg" in
    --config) skip_next=1 ;;
    -*) ;;
    *) subcommand="$arg"; break ;;
  esac
done
case "$subcommand" in
  init|use|status|inspect|import-reconstruction) default_h=0.5 ;;
  setup) default_h=3 ;;
  reconstruct) default_h=4 ;;
  doctor|simulate) default_h=1 ;;
  render)
    # `--backend rtx` needs Isaac Sim 5.1 RTX cameras, which crash under Modal's NVIDIA driver
    # (610/620; NVIDIA lists 580.x): see docs/pipeline/OPTIMIZATION.md. The default overlay
    # backend (pyrender of simulate's physics replay) runs here. V2D_MODAL_ALLOW_RTX=1 is for
    # testing a fix only.
    rtx=0; prev=""
    for arg in "$@"; do
      [[ "$arg" == "--backend=rtx" || ( "$prev" == "--backend" && "$arg" == "rtx" ) ]] && rtx=1
      prev="$arg"
    done
    if [[ $rtx -eq 1 && "${V2D_MODAL_ALLOW_RTX:-0}" != "1" ]]; then
      echo "render --backend rtx needs Isaac RTX cameras, which crash on Modal's driver; use the" \
        "default overlay backend here, or run rtx with ./run_e2e.sh on an RTX host with driver 580.x" >&2
      exit 2
    fi
    default_h=0.5 ;;
  retarget) default_h=2 ;;
  train-expert) default_h=24 ;;
  collect|finetune) default_h=12 ;;
  evaluate) default_h=8 ;;
  *) echo "unknown run_e2e.sh subcommand: ${subcommand:-<none>}" >&2; exit 2 ;;
esac
for arg in "$@"; do [[ "$arg" == "--dry-run" ]] && default_h=0.5; done
timeout_h="${V2D_MODAL_TIMEOUT_H:-$default_h}"
export V2D_MODAL_CONTROL_TIMEOUT_H="$timeout_h" V2D_MODAL_RG_TIMEOUT_H="$timeout_h"

# JSON keeps arguments exact (no shell re-quoting between here and the container).
argv_json="$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "$@")"
modal_flags=()
[[ "${V2D_MODAL_DETACH:-1}" == "0" ]] || modal_flags+=(--detach)
entry_flags=(--argv-json "${argv_json}")
[[ "${V2D_MODAL_BREAK_LOCK:-0}" == "1" ]] && entry_flags+=(--break-lock)

echo "[v2d-modal] ${subcommand}: server-side timeout ${timeout_h}h"
echo "[v2d-modal] live board in another terminal: MODAL_PROFILE=${MODAL_PROFILE} $0 --board --watch"
# ${arr[@]+...} keeps `set -u` happy with empty arrays on macOS bash 3.2.
exec modal run ${modal_flags[@]+"${modal_flags[@]}"} -m tools.modal_v2d.app::main "${entry_flags[@]}"
