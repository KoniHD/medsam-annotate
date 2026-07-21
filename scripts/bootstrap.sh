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

# Ultrasound-specific warm start for the fine-tune loop (design.md Sec 4 / Sec 6 -- M5). Same
# repo, same ~149MB size as MedSAM2_latest.pt. lib/trainers/medsam2.py falls back to
# MedSAM2_latest.pt if this is absent, so it is fetched here but not fatal if the download fails.
US_CHECKPOINT_URL="https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_US_Heart.pt"
US_CHECKPOINT="$MODEL_DIR/MedSAM2_US_Heart.pt"

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

# M6 gap: a network drop mid-`curl` under `set -e` aborts the script but can still leave a
# truncated file at the *final* path -- a bare `[ -f "$CHECKPOINT" ]` on the next run would then
# see "already present" forever and never retry, silently serving a corrupt checkpoint at demo
# time. Download to a sibling .tmp path and `mv` into place only once curl exits 0, so the final
# path only ever holds a fully-written file.
#
# Deliberately NOT reliant on `set -e` to enforce that ordering: this function is called as the
# condition of an `if`/`if !` at both call sites below, and POSIX/bash disable `errexit` for the
# duration of a function invoked in a command-list/condition position. A `curl` failure there
# would silently fall through to `mv`, installing a truncated `.part` at the final path and
# returning the *mv* exit status (0) -- exactly the "truncated file masquerading as present
# forever" failure this function exists to prevent. So enforce it explicitly: only run `mv` when
# `curl` itself reported success, and always clean up the partial file on any failure so the next
# run retries instead of seeing a bogus "already present".
_fetch_checkpoint() {
    local url="$1" dest="$2" tmp="$2.part"
    mkdir -p "$(dirname "$dest")"
    echo "    downloading (~149MB, CC-BY-SA-4.0) ..."
    if curl -fL --retry 3 -o "$tmp" "$url"; then
        mv "$tmp" "$dest"
        return 0
    else
        local rc=$?
        rm -f "$tmp"
        return "$rc"
    fi
}

echo "==> MedSAM2 checkpoint"
if [ -f "$CHECKPOINT" ]; then
    echo "    already present: $CHECKPOINT"
else
    _fetch_checkpoint "$CHECKPOINT_URL" "$CHECKPOINT"
fi

echo "==> MedSAM2 ultrasound checkpoint (fine-tune warm start, design.md Sec 4)"
if [ -f "$US_CHECKPOINT" ]; then
    echo "    already present: $US_CHECKPOINT"
else
    if ! _fetch_checkpoint "$US_CHECKPOINT_URL" "$US_CHECKPOINT"; then
        echo "    WARNING: download failed -- the fine-tune trainer will fall back to" \
             "MedSAM2_latest.pt (lib/trainers/medsam2.py). Not fatal; re-run bootstrap.sh later" \
             "to retry." >&2
        rm -f "$US_CHECKPOINT" "$US_CHECKPOINT.part"
    fi
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

# M6 gap: the checks above import torch/monailabel/sam2 but never actually load
# MedSAM2_latest.pt, so a corrupt/truncated checkpoint (partial disk, bad copy) would pass
# bootstrap and only surface as an opaque failure in front of the annotator. Load it for real on
# cpu (works everywhere, no accelerator needed) and confirm it comes back as the ~39.0M-parameter
# SAM 2.1 tiny model the demo was built and verified against (design.md Sec 8 / task brief).
echo "==> Verifying checkpoint loads (cpu, offline)"
uv run python -c "
from sam2.build_sam import build_sam2
model = build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml', '$CHECKPOINT', device='cpu')
n_params = sum(p.numel() for p in model.parameters())
print(f'    loaded OK -- {n_params/1e6:.1f}M params')
assert 35e6 < n_params < 45e6, f'expected ~39.0M params, got {n_params}'
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
