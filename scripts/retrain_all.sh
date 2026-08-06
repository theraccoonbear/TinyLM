#!/usr/bin/env bash
# Retrain every model in models/ from its corpus in corpora/, as one
# single self-contained run — no further approval/input needed once
# launched, safe to walk away from.
#
# Resumable: each corpus gets a marker file in .retrain-state/ once it
# passes; rerunning this script skips anything already done, rather than
# redoing it.
#
# Crash-safe mid-corpus too: each training run checkpoints its own
# progress periodically (--checkpoint-every), so a hard crash partway
# through a corpus still leaves a usable, recent model on disk instead
# of losing everything back to zero for whatever was in flight.
#
# Self-correcting: "trained" isn't just "the process exited 0" — tinylm
# runs an automatic sanity check after every training run (does the
# model actually use common words at realistic rates?) and exits with
# status 2 specifically when that check fails. This script watches for
# exit 2 and automatically resumes training the SAME corpus with more
# epochs (via --load-model) rather than accepting an undertrained model
# just because the process didn't crash. No formula reliably predicts
# how many epochs a given corpus needs — this measures the actual
# outcome and keeps going until it's real, up to a retry cap.
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

CORPORA=(mothergoose alice macbeth grimm sherlock shakespeare)

# Epochs per attempt, and how many attempts (cumulative, via --load-model)
# before giving up on a corpus. With Adam this is deliberately a small
# starting budget, not a per-corpus prediction — the retry loop is the
# actual mechanism that finds "enough," not this number.
EPOCHS_PER_ATTEMPT=20
CHECKPOINT_EVERY=5
MAX_ATTEMPTS=5

TRAINED=()
SKIPPED=()
FAILED=()

for name in "${CORPORA[@]}"; do
    corpus="corpora/$name.txt"
    marker="$STATE_DIR/$name.done"
    log="$LOG_DIR/$name.log"
    model="models/$name.model"

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

    start=$(date +%s)
    passed=false
    total_epochs=0

    for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
        total_epochs=$(( total_epochs + EPOCHS_PER_ATTEMPT ))
        args=(--data-set "$corpus" --epochs "$EPOCHS_PER_ATTEMPT" --checkpoint-every "$CHECKPOINT_EVERY" --length 200 --save-model "$model")
        if [[ $attempt -gt 1 ]]; then
            args+=(--load-model "$model")
        fi

        echo "=== $name: attempt $attempt (+${EPOCHS_PER_ATTEMPT} epochs, ${total_epochs} cumulative) — log: $log ==="
        "$BIN" "${args[@]}" > "$log" 2>&1
        exit_code=$?

        if [[ $exit_code -eq 0 ]]; then
            echo "=== $name: PASSED sanity check after ${total_epochs} epochs (attempt $attempt) ==="
            passed=true
            break
        elif [[ $exit_code -eq 2 ]]; then
            echo "=== $name: attempt $attempt did not pass the sanity check yet — resuming with more epochs ==="
            continue
        else
            echo "=== $name: attempt $attempt CRASHED (exit $exit_code) — see $log — moving to next corpus ==="
            break
        fi
    done

    elapsed=$(( $(date +%s) - start ))
    if [[ "$passed" == true ]]; then
        touch "$marker"
        echo "=== $name: DONE in ${elapsed}s (${total_epochs} total epochs) ==="
        TRAINED+=("$name (${total_epochs}ep, ${elapsed}s)")
    else
        echo "=== $name: FAILED after ${elapsed}s (${total_epochs} epochs attempted) — see $log — continuing to next corpus ==="
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
