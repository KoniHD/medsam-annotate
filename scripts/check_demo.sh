#!/usr/bin/env bash
# Preflight for the Path A demo (design.md Sec 10 + "Reliability rules for the demo").
#
# Run this ~10 minutes before the annotator sits down:
#
#   ./scripts/check_demo.sh
#
# It checks, and prints a PASS/FAIL line for each of:
#   1. MedSAM2 checkpoint present (and not a truncated/corrupt file)
#   2. Datastore non-empty and containing at least one unlabeled image
#   3. Server reachable (starts one via start_server.sh if nothing is listening yet --
#      pre-warmed, so it's left running and ready for the demo)
#   4. A real box-prompt REST round trip returns a non-empty mask
#   5. That mask flows through the measurement export into a real CSV
#
# Exits 0 only if every check passes; non-zero (and a summary of what failed) otherwise. Nothing
# here is destructive and nothing here touches the network beyond localhost.
set -uo pipefail  # deliberately NOT -e: every check must run and report, not abort on the first

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

STUDIES="${STUDIES:-$REPO_ROOT/data/legus}"
PORT="${PORT:-8000}"
BASE_URL="http://localhost:$PORT"
export PATH="$REPO_ROOT/.venv/bin:$PATH"

FAILURES=0
STARTED_SERVER_PID=""
SCRATCH_DIR="$(mktemp -d)"

pass() { printf "PASS  %s\n" "$1"; }
fail() { printf "FAIL  %s\n" "$1"; FAILURES=$((FAILURES + 1)); }

cleanup() {
    rm -rf "$SCRATCH_DIR"
    # Only tear down a server *this script* launched, and only if it never became reachable --
    # a healthy one it started is the server the demo will use, so leave it running.
    if [ -n "$STARTED_SERVER_PID" ] && [ "$SERVER_BECAME_READY" != "1" ]; then
        kill "$STARTED_SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
SERVER_BECAME_READY=0

echo "== LEGUS demo preflight =="
echo

# 1. Checkpoint present and not obviously corrupt -----------------------------------------------
CHECKPOINT="$REPO_ROOT/apps/legus/model/MedSAM2_latest.pt"
if [ -f "$CHECKPOINT" ]; then
    size=$(wc -c <"$CHECKPOINT" | tr -d ' ')
    if [ "$size" -gt 50000000 ]; then
        pass "checkpoint present ($CHECKPOINT, $((size / 1000000)) MB)"
    else
        fail "checkpoint at $CHECKPOINT is only $size bytes -- looks truncated; re-run scripts/bootstrap.sh"
    fi
else
    fail "checkpoint missing at $CHECKPOINT -- run scripts/bootstrap.sh"
fi

# 2. Datastore non-empty and has unlabeled images -----------------------------------------------
if [ -d "$STUDIES" ]; then
    image_count=$(find "$STUDIES" -maxdepth 1 -type f \( -name '*.nii.gz' -o -name '*.nii' \) 2>/dev/null | wc -l | tr -d ' ')
    labeled_count=$(find "$STUDIES/labels/final" -maxdepth 1 -type f \( -name '*.nii.gz' -o -name '*.nii' \) 2>/dev/null | wc -l | tr -d ' ')
else
    image_count=0
    labeled_count=0
fi
if [ "$image_count" -gt 0 ]; then
    unlabeled=$((image_count - labeled_count))
    if [ "$unlabeled" -gt 0 ]; then
        pass "datastore has $image_count image(s) under $STUDIES, $unlabeled unlabeled"
    else
        fail "datastore has $image_count image(s) but all are already in labels/final -- fetch more: uv run scripts/fetch_data.py --limit 8"
    fi
else
    fail "no images under $STUDIES -- run: uv run scripts/fetch_data.py --limit 8"
fi

# 3. Server reachable (start + pre-warm one if nothing is listening) ----------------------------
code=$(curl -s -o /dev/null -w '%{http_code}' -m 2 "$BASE_URL/info/" 2>/dev/null || true)
if [ "$code" = "200" ]; then
    pass "server already running and reachable at $BASE_URL"
    SERVER_BECAME_READY=1
else
    echo "  no server at $BASE_URL -- starting one now (this also pre-warms it) ..."
    LEGUS_SKIP_PREWARM=0 PORT="$PORT" STUDIES="$STUDIES" \
        nohup "$REPO_ROOT/scripts/start_server.sh" >"$SCRATCH_DIR/start_server.log" 2>&1 &
    STARTED_SERVER_PID=$!
    ready=0
    for _ in $(seq 1 90); do
        if ! kill -0 "$STARTED_SERVER_PID" 2>/dev/null; then
            break
        fi
        code=$(curl -s -o /dev/null -w '%{http_code}' -m 2 "$BASE_URL/info/" 2>/dev/null || true)
        if [ "$code" = "200" ]; then
            ready=1
            break
        fi
        sleep 1
    done
    if [ "$ready" = "1" ]; then
        pass "started a server at $BASE_URL (left running for the demo -- pid $STARTED_SERVER_PID)"
        SERVER_BECAME_READY=1
    else
        fail "could not reach a server at $BASE_URL within 90s -- see $SCRATCH_DIR/start_server.log"
        echo "  ---- start_server.sh output ----"
        sed 's/^/  /' "$SCRATCH_DIR/start_server.log" 2>/dev/null | tail -20
        echo "  ---------------------------------"
    fi
fi

# design.md Sec 8 Path A wants cpu as the demo's zero-surprise default; start_server.sh now pins
# that, but the preflight should say out loud which device the server actually selected rather
# than assuming -- so a stray LEGUS_DEVICE=mps left set in the shell doesn't silently demo on the
# rougher-edged path with nobody noticing until something goes wrong live.
if [ "$SERVER_BECAME_READY" = "1" ]; then
    default_device=$(curl -s -m 5 "$BASE_URL/info/" 2>/dev/null | .venv/bin/python -c "
import json, sys
try:
    info = json.load(sys.stdin)
    devices = info['models']['medsam2_2d']['config']['device']
    print(devices[0] if devices else 'unknown')
except Exception:
    print('unknown')
" 2>/dev/null)
    if [ "$default_device" = "cpu" ]; then
        pass "server default device is cpu (design.md Sec 8 Path A zero-surprise path)"
    elif [ -n "$default_device" ] && [ "$default_device" != "unknown" ]; then
        echo "  NOTE: server default device is '$default_device', not cpu -- design.md Sec 8 Path A" \
             "recommends cpu as the zero-surprise demo default. Set LEGUS_DEVICE=cpu (or unset it)" \
             "before starting the server if this wasn't intentional."
    fi
fi

# 4 & 5: only meaningful once the server is actually reachable ----------------------------------
MASK_PATH="$SCRATCH_DIR/mask.nii.gz"
if [ "$SERVER_BECAME_READY" = "1" ] && [ "$image_count" -gt 0 ]; then
    if round_trip_out=$(.venv/bin/python "$REPO_ROOT/scripts/legus_probe.py" round-trip \
        --base-url "$BASE_URL" --studies "$STUDIES" --out-mask "$MASK_PATH" 2>&1); then
        pass "box-prompt REST round trip returned a non-empty mask ($round_trip_out)"
    else
        fail "box-prompt REST round trip did not return a usable mask -- $round_trip_out"
    fi
else
    fail "skipped REST round-trip check (server unreachable or no images)"
fi

if [ -s "$MASK_PATH" ]; then
    image_for_measure=$(.venv/bin/python "$REPO_ROOT/scripts/legus_probe.py" pick-box --studies "$STUDIES" | cut -d' ' -f1)
    image_path="$STUDIES/$image_for_measure.nii.gz"
    csv_out="$SCRATCH_DIR/measurements.csv"
    if measure_out=$(.venv/bin/python "$REPO_ROOT/apps/legus/lib/measure/statistics.py" \
        --image "$image_path" --mask "$MASK_PATH" --subject-id "$image_for_measure" --out "$csv_out" 2>&1); then
        rows=$(($(wc -l <"$csv_out" | tr -d ' ') - 1))
        pass "measurement CSV generated ($rows row(s), see \$STUDIES-relative image=$image_for_measure)"
    else
        fail "measurement export failed -- $measure_out"
    fi
else
    fail "skipped measurement export check (no mask from the round-trip check to feed it)"
fi

echo
if [ "$FAILURES" -eq 0 ]; then
    echo "ALL CHECKS PASSED. Server is up and pre-warmed at $BASE_URL -- point 3D Slicer at it."
    exit 0
else
    echo "$FAILURES check(s) FAILED -- see above. Do not start the demo until these are fixed."
    exit 1
fi
