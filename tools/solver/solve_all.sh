#!/bin/bash
# Solve every A-class seed representative, forward then backward, one at a time.
#
# The three A-class seeds (A1, A2, A3) have stabilizer 4 under the order-32 symmetry
# group, against C3's 32, so each is ~8x C3 in both layer size and time: ~3 h forward
# plus ~2.2 h backward, ~16 h for all three. Each run saturates all 16 cores, so they
# are strictly sequential — running two at once would not finish sooner.
#
# Runs in space-saving mode (`keep >= 12`): a layer's buckets are dropped once
# flattened, and a layer's keys/values are dropped once the layer above it is valued
# and it sits below 12 empties. Deep layers exist only to value the shallow ones, and
# the runtime player live-solves below ~16 empties. Without this the three seeds plus
# the last one's working set need ~280 GB against ~288 GB free — too tight to leave
# unattended.
#
#   ./tools/solver/solve_all.sh [OUT_DIR] [WORKERS]

set -euo pipefail

OUT_DIR="${1:-/tmp/solve-data2}"
WORKERS="${2:-16}"
CHUNK=50000
KEEP=12

cd "$(dirname "$0")/../.."

for seed in A1 A2 A3; do
  for phase in forward backward; do
    printf '\n=== %s %s  %s ===\n' "$seed" "$phase" "$(date '+%Y-%m-%d %H:%M:%S')"
    df -h /System/Volumes/Data | tail -1
    if [ "$phase" = forward ]; then
      uv run python tools/solver/forward.py "$seed" "$OUT_DIR" "$CHUNK" "$WORKERS"
    else
      uv run python tools/solver/backward.py "$seed" "$OUT_DIR" "$CHUNK" "$WORKERS" "$KEEP"
    fi
  done
  printf '=== %s COMPLETE  %s ===\n' "$seed" "$(date '+%Y-%m-%d %H:%M:%S')"
  du -sh "$OUT_DIR/$seed"
done

printf '\n=== ALL SEEDS COMPLETE  %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S')"
