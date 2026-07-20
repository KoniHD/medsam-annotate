#!/usr/bin/env bash
# One-command setup for the Path A (local, offline) MVP — see design.md §8.
#
# Idempotent: safe to re-run. Everything is uv-managed; no conda, no bare pip.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CHECKPOINT_URL="https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt"
MODEL_DIR="apps/legus/model"
CHECKPOINT="$MODEL_DIR/MedSAM2_latest.pt"

command -v uv >/dev/null || { echo "uv not found — https://docs.astral.sh/uv/"; exit 1; }

echo "==> Submodules (pinned; pushes disabled)"
git submodule update --init --recursive
# Belt and braces: a fresh clone re-derives the push URL from .gitmodules, but an older clone
# may not have it. Never let an accidental push reach an upstream we don't own.
for sm in external/MONAILabel external/MedSAM2; do
    git -C "$sm" remote set-url --push origin DISABLE 2>/dev/null || true
done

echo "==> Python environment (uv, 3.12)"
# uv.lock pins everything including monailabel from the submodule; --no-build-isolation for it is
# declared in pyproject.toml, since it lists torch in setup_requires.
uv sync

echo "==> MedSAM2 checkpoint"
if [ -f "$CHECKPOINT" ]; then
    echo "    already present: $CHECKPOINT"
else
    mkdir -p "$MODEL_DIR"
    echo "    downloading (~149MB, CC-BY-SA-4.0) ..."
    curl -fL --retry 3 -o "$CHECKPOINT" "$CHECKPOINT_URL"
fi

echo "==> Verifying"
uv run python -c "
import torch, monailabel, sam2
print(f'    torch {torch.__version__}  monailabel {monailabel.__version__}')
print(f'    cuda={torch.cuda.is_available()}  mps={torch.backends.mps.is_available()}')
"
uv run python -c "
import sys; sys.path.insert(0, 'apps/legus')
from lib.infers.medsam2 import available_devices
print('    devices (best first):', available_devices())
"

cat <<'EOF'

Bootstrap complete.

Next:
  uv run scripts/fetch_data.py --limit 8   # public CAMUS demo data
  ./scripts/start_server.sh                # MONAI Label on http://localhost:8000

Then in 3D Slicer: install the MONAI Label extension (Extension Manager, or
Developer > Extension Wizard on external/MONAILabel/plugins/slicer/MONAILabel),
point it at http://localhost:8000, and pick medsam2_2d.
EOF
