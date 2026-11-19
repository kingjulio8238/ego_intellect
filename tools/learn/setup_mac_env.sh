#!/usr/bin/env bash
# CPU-only learning environment for the V2D repo (Apple Silicon / no NVIDIA GPU).
# Installs just enough to run every package's contract tests, inspect the checked-in
# example artifacts, and replay retargeted motion in viser. No Docker, no CUDA.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${V2D_LEARN_VENV:-$HOME/.venvs/v2d-learn}"

uv venv -q -p 3.11 "$VENV"
export VIRTUAL_ENV="$VENV"
# --no-sources: the ingestion pyproject pins torch to a Linux cu130 index; this gets CPU/MPS torch.
uv pip install -q -e "$REPO/video_ingestion_agent[local,webapp,dev]" --no-sources
uv pip install -q -e "$REPO/reconstruction/modules/v2d_common"
uv pip install -q pyarrow pandas onnxruntime h5py scipy av imageio imageio-ffmpeg trimesh \
  pin pin-pink daqp qpsolvers viser
echo "ready: source $VENV/bin/activate"
