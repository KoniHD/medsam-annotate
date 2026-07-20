#!/usr/bin/env bash
# Start the MONAI Label server for the Path A demo — offline, local, no cold start.
#
#   ./scripts/start_server.sh                 # auto device (MPS here, CUDA on a GPU box)
#   LEGUS_DEVICE=cpu ./scripts/start_server.sh   # pin the zero-surprise reference path
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

STUDIES="${STUDIES:-$REPO_ROOT/data/legus}"
PORT="${PORT:-8000}"

# Let MPS fall back to CPU for ops it hasn't implemented, rather than raising at the annotator
# (design.md §8 Path A). The adapter also sets this at import time; belt and braces.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# `monailabel start_server` re-execs `python` found on PATH — NOT the interpreter that launched
# it. Without the venv on PATH it dies with "No module named 'monailabel'". This line is the
# whole reason this script exists.
export PATH="$REPO_ROOT/.venv/bin:$PATH"

if [ ! -f "apps/legus/model/MedSAM2_latest.pt" ]; then
    echo "MedSAM2 checkpoint missing — run ./scripts/bootstrap.sh first." >&2
    exit 1
fi

if [ ! -d "$STUDIES" ] || [ -z "$(ls -A "$STUDIES" 2>/dev/null)" ]; then
    echo "No studies in $STUDIES — run: uv run scripts/fetch_data.py --limit 8" >&2
    exit 1
fi

echo "Serving $STUDIES on http://localhost:$PORT  (device: ${LEGUS_DEVICE:-auto})"
exec .venv/bin/monailabel start_server \
    --app "$REPO_ROOT/apps/legus" \
    --studies "$STUDIES" \
    --port "$PORT"
