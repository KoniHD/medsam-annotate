#!/usr/bin/env bash
# Stop the demo server and reset the datastore back to a clean, unlabeled state.
#
#   ./scripts/reset_demo.sh              # stop the server AND clear submitted annotations
#   ./scripts/reset_demo.sh --stop-only  # just stop the server, keep annotations
#
# "Reset annotations" means: throw away the masks the annotator SUBMITTED during a run-through
# (labels/final/) so every image reads as unlabeled again and "Next Sample" has work to serve. It
# deliberately KEEPS:
#   * the images and their per-image <id>.json sidecars,
#   * labels/original/  -- the public CAMUS ground truth, part of the demo data, not annotator work.
# It removes datastore_v2.json (MONAI Label's index), which is rebuilt from disk on the next server
# start -- so after a reset the datastore is reconciled fresh with only the original labels.
#
# The server must be stopped first: mutating the datastore files under a live server races its own
# reconcile loop. To re-fetch the demo data from scratch instead, delete data/legus and re-run
# scripts/fetch_data.py -- this script never touches the network.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

STUDIES="${STUDIES:-$REPO_ROOT/data/legus}"
STOP_ONLY=0
[ "${1:-}" = "--stop-only" ] && STOP_ONLY=1

echo "==> Stopping any MONAI Label server"
if pkill -f monailabel.main 2>/dev/null; then
    # Wait for it to actually exit so the datastore is no longer being written.
    for _ in $(seq 1 20); do
        pgrep -f monailabel.main >/dev/null 2>&1 || break
        sleep 0.5
    done
    if pgrep -f monailabel.main >/dev/null 2>&1; then
        echo "    still running after 10s; force-killing"
        pkill -9 -f monailabel.main 2>/dev/null || true
        sleep 1
    fi
    echo "    stopped."
else
    echo "    no server was running."
fi

if [ "$STOP_ONLY" = "1" ]; then
    echo "--stop-only: annotations left untouched."
    exit 0
fi

# Guard: only operate on something that actually looks like our datastore, so a mistyped STUDIES
# can't delete an unrelated directory.
if [ ! -d "$STUDIES/labels" ]; then
    echo "No datastore at $STUDIES (no labels/ dir) -- nothing to reset." >&2
    echo "If you meant a different location, set STUDIES=... ; to recreate the demo data run:" >&2
    echo "  uv run scripts/fetch_data.py --limit 8" >&2
    exit 1
fi

FINAL_DIR="$STUDIES/labels/final"
echo "==> Clearing submitted annotations ($FINAL_DIR)"
if [ -d "$FINAL_DIR" ]; then
    n=$(find "$FINAL_DIR" -type f | wc -l | tr -d ' ')
    find "$FINAL_DIR" -mindepth 1 -delete
    echo "    removed $n submitted label file(s) (labels/original/ ground truth kept)."
else
    echo "    none present."
fi

echo "==> Removing datastore index (rebuilt clean on next start)"
rm -f "$STUDIES/datastore_v2.json"
# Older MONAI Label versions used datastore.json; remove it too if present.
rm -f "$STUDIES/datastore.json"
echo "    done."

# The SAM2 image cache under ~/.cache/monailabel is ephemeral inference scratch, not annotations,
# and is safe to leave; clear it too for a fully cold next run.
CACHE="${HOME}/.cache/monailabel/sam2"
[ -d "$CACHE" ] && rm -rf "$CACHE" && echo "==> Cleared inference image cache ($CACHE)"

echo ""
echo "Reset complete. The datastore now holds only the original (unlabeled) images."
echo "Start again with:  ./scripts/check_demo.sh    (or  ./scripts/start_server.sh )"
