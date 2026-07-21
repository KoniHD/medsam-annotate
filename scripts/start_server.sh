#!/usr/bin/env bash
# Start the MONAI Label server for the Path A demo — offline, local, no cold start.
#
#   ./scripts/start_server.sh                 # defaults to cpu -- design.md Sec 8 Path A's
#                                              # "zero-surprise" choice, not MPS
#   LEGUS_DEVICE=mps ./scripts/start_server.sh   # explicit opt-in to the faster, rougher-edged path
#
# design.md Sec 8 Path A is explicit that CPU, not MPS, is the demo path: "Run inference on CPU
# ... to avoid Apple MPS rough edges entirely; latency of a few seconds/image is fine for a demo.
# (MPS possible with PYTORCH_ENABLE_MPS_FALLBACK=1, but CPU is the zero-surprise choice.)" So this
# script pins LEGUS_DEVICE=cpu by default -- MPS remains available and still works (the two-layer
# accelerator-fallback in lib/infers/medsam2.py covers it either way), but it is no longer what a
# plain, no-env-vars run of this script hands the Slicer device chooser as its default entry.
#
# design.md Sec 10 "Reliability rules for the demo": pre-warm/pre-load everything so the
# annotator's FIRST prompt is not the one that pays model-load latency. This script starts the
# server, waits for it to answer, fires one real box-prompt request at a demo image to force the
# predictor to build (measured: ~1-2s cold vs ~0.2s warm on this machine, see M6 report), and only
# then hands control to the foreground so Ctrl-C still stops the server the way it always did.
#
# LEGUS_SKIP_PREWARM=1 skips the warm-up call (e.g. if you deliberately want to demonstrate the
# cold-start cost, or the datastore has no images yet).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

STUDIES="${STUDIES:-$REPO_ROOT/data/legus}"
PORT="${PORT:-8000}"
STARTUP_TIMEOUT="${LEGUS_STARTUP_TIMEOUT:-90}"

# design.md Sec 8 Path A: CPU is the zero-surprise default for the demo. `available_devices()`
# (lib/infers/medsam2.py) pins whatever LEGUS_DEVICE names to the front of the device list that
# populates the Slicer device chooser, so this line is what actually makes cpu the packaged
# default instead of MPS -- set it before the export below so LEGUS_DEVICE is always defined by
# the time the server process reads it. An operator/caller-set LEGUS_DEVICE always wins.
export LEGUS_DEVICE="${LEGUS_DEVICE:-cpu}"

# Let MPS fall back to CPU for ops it hasn't implemented, rather than raising at the annotator
# (design.md Sec 8 Path A). The adapter also sets this at import time; belt and braces. Harmless
# no-op when LEGUS_DEVICE=cpu (nothing runs on MPS to fall back from).
export PYTORCH_ENABLE_MPS_FALLBACK=1

# `monailabel start_server` re-execs `python` found on PATH — NOT the interpreter that launched
# it (confirmed: .venv/bin/monailabel ends in `exec ${PYEXE:-python} -m monailabel.main`, a real
# exec so the PID is preserved but the interpreter is whatever `python` resolves to on PATH).
# Without the venv on PATH it dies with "No module named 'monailabel'". This line is the whole
# reason this script exists.
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

# Launch in the background (not `exec`) so this script can wait for readiness and pre-warm
# before an annotator ever touches it. `wait` at the end still blocks the terminal exactly like
# the old `exec` did, and the trap below still gets Ctrl-C to stop the real server process (exec
# means the PID captured here IS the server process, not a wrapper around it).
.venv/bin/monailabel start_server \
    --app "$REPO_ROOT/apps/legus" \
    --studies "$STUDIES" \
    --port "$PORT" &
SERVER_PID=$!

cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "Waiting for the server to become reachable ..."
ready=0
for _ in $(seq 1 "$STARTUP_TIMEOUT"); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server process exited before becoming reachable — see the log above for the real error." >&2
        exit 1
    fi
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 2 "http://localhost:$PORT/info/" 2>/dev/null || true)
    if [ "$code" = "200" ]; then
        ready=1
        break
    fi
    sleep 1
done
if [ "$ready" != "1" ]; then
    echo "Server did not become reachable within ${STARTUP_TIMEOUT}s (port $PORT)." >&2
    exit 1
fi
echo "Server reachable."

if [ "${LEGUS_SKIP_PREWARM:-0}" = "1" ]; then
    echo "LEGUS_SKIP_PREWARM=1 — leaving the model cold; the first real prompt will pay load latency."
else
    echo "Pre-warming the model with one real box-prompt request ..."
    if warm_output=$(.venv/bin/python "$REPO_ROOT/scripts/legus_probe.py" round-trip \
        --base-url "http://localhost:$PORT" --studies "$STUDIES" 2>&1); then
        echo "  $warm_output"
        echo "Pre-warm complete — the annotator's first prompt will be fast."
    else
        echo "  pre-warm request did not succeed (server is still up; the annotator's first" >&2
        echo "  prompt will now pay the load latency instead). Output:" >&2
        echo "  $warm_output" >&2
    fi
fi

echo ""
echo "Ready. In 3D Slicer: MONAI Label extension -> http://localhost:$PORT -> medsam2_2d / medsam2_3d"
echo "(Ctrl-C stops the server. If this was started in the background and Ctrl-C doesn't reach"
echo " it, run: pkill -f monailabel.main -- verified to stop it reliably either way.)"
wait "$SERVER_PID"
