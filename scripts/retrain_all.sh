#!/usr/bin/env bash
# Retrain every model in models/ from its corpus in corpora/, as one
# single self-contained run — no further approval/input needed once
# launched, safe to walk away from.
#
# Resumable: each corpus gets a marker file in .retrain-state/ once it
# finishes successfully; rerunning this script skips anything already
# done, rather than redoing it.
#
# Crash-safe mid-corpus too: each training run checkpoints its own
# progress periodically (--checkpoint-every), so a hard crash partway
# through a corpus still leaves a usable, recent model on disk instead
# of losing everything back to zero for whatever was in flight.
#
# Order is deliberate: cheapest/fastest corpora first, most expensive
# (shakespeare) last — so if something does stall or fail partway
# through an unattended run, you wake up to as much finished, usable
# work as possible, not a stall on the very first (long) one.
set -uo pipefail

cd "$(dirname "$0")/.."   # repo root, since this script lives in scripts/

# Self-contained: don't assume the invoking shell already has cargo on
# PATH (it won't, in a fresh non-interactive process).
if ! command -v cargo >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi

STATE_DIR="$(pwd)/.retrain-state"
LOG_DIR="$(pwd)/.retrain-logs"
mkdir -p "$STATE_DIR" "$LOG_DIR"

echo "=== Building release binary ==="
if ! cargo build --release; then
    echo "BUILD FAILED — aborting before touching any models."
    exit 1
fi
BIN="./target/release/tinylm"

# name : epochs : checkpoint-every
# Epoch counts come from `tinylm --data-set corpora/<name>.txt --analyze`
# runs at the current --vocab-size/--hidden-dim defaults, not guessed.
CORPORA=(
    "mothergoose:83:10"
    "alice:51:10"
    "macbeth:78:10"
    "grimm:15:3"
    "sherlock:15:3"
    "shakespeare:15:1"
)

TRAINED=()
SKIPPED=()
FAILED=()

for entry in "${CORPORA[@]}"; do
    IFS=":" read -r name epochs ckpt <<< "$entry"
    corpus="corpora/$name.txt"
    marker="$STATE_DIR/$name.done"
    log="$LOG_DIR/$name.log"

    if [[ -f "$marker" ]]; then
        echo "=== $name: already done (marker found), skipping ==="
        SKIPPED+=("$name")
        continue
    fi

    if [[ ! -f "$corpus" ]]; then
        echo "=== $name: $corpus not found — skipping, not marking done ==="
        FAILED+=("$name (missing corpus)")
        continue
    fi

    echo "=== $name: training ($epochs epochs, checkpoint every $ckpt epochs) — log: $log ==="
    start=$(date +%s)
    if "$BIN" --data-set "$corpus" --epochs "$epochs" \
        --checkpoint-every "$ckpt" --length 200 \
        --save-model "models/$name.model" > "$log" 2>&1; then
        touch "$marker"
        elapsed=$(( $(date +%s) - start ))
        echo "=== $name: DONE in ${elapsed}s ==="
        TRAINED+=("$name (${elapsed}s)")
    else
        elapsed=$(( $(date +%s) - start ))
        echo "=== $name: FAILED after ${elapsed}s — see $log — continuing to next corpus ==="
        FAILED+=("$name")
    fi
done

echo ""
echo "================ SUMMARY ================"
echo "Trained:  ${TRAINED[*]:-none}"
echo "Skipped:  ${SKIPPED[*]:-none}"
echo "Failed:   ${FAILED[*]:-none}"
echo "==========================================="

[[ ${#FAILED[@]} -eq 0 ]]
