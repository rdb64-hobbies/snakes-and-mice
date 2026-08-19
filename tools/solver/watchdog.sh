#!/bin/bash
# Kill the solver if the machine runs low on memory, and say so.
#
# The solver's peak is per-worker and hits all 16 workers at once, so a bad sizing
# constant does not slow the machine down — it exhausts RAM and swap within a layer.
# This is the backstop for running unattended: it stops the run before macOS starts
# killing applications, sacrificing the run to keep the machine.
#
# Watches the three signals that actually discriminate on macOS:
#   * swap used
#   * pages held by the compressor
#   * kern.memorystatus_vm_pressure_level (1 normal, 2 warn, 4 critical)
# It deliberately does NOT watch "Pages free". macOS keeps that low by design and
# holds reclaimable memory as inactive/speculative, so a free-pages threshold fires on
# a perfectly healthy machine — an earlier version of this script killed a good run
# that way, with swap at 470 MB against a 10 GB limit. For reference, the failure this
# guards against showed swap at 30.5 GB and the compressor at 37.3 GB, against ~0.5 GB
# and ~1.2 GB when healthy.
#
# Also watches free disk. A mouse-to-move layer writes its shard files before the
# dedup consumes them — A1's layer 12 held ~108 GB at once — so a full volume is a
# real way for this to end, and macOS behaves badly at zero free space.
#
#   ./tools/solver/watchdog.sh [SWAP_LIMIT_MB] [COMPRESSOR_LIMIT_MB] [LOG] [MIN_DISK_GB]

set -uo pipefail

SWAP_LIMIT_MB="${1:-12288}"
COMPRESSOR_LIMIT_MB="${2:-24576}"
LOG="${3:-/tmp/solve-watchdog.log}"
MIN_DISK_GB="${4:-25}"
PAGE=$(vm_stat | head -1 | grep -o '[0-9]*')

while true; do
  swap_mb=$(sysctl -n vm.swapusage | awk '{gsub(/M/,"",$6); print int($6)}')
  comp_pages=$(vm_stat | awk '/occupied by compressor/ {gsub(/\./,"",$5); print $5}')
  comp_mb=$(( comp_pages * PAGE / 1048576 ))
  level=$(sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null || echo 1)
  disk_gb=$(df -g /System/Volumes/Data | tail -1 | awk '{print $4}')

  if [ "$swap_mb" -gt "$SWAP_LIMIT_MB" ] || [ "$comp_mb" -gt "$COMPRESSOR_LIMIT_MB" ] \
     || [ "$level" -ge 4 ] || [ "$disk_gb" -lt "$MIN_DISK_GB" ]; then
    {
      echo "=== WATCHDOG TRIPPED $(date '+%Y-%m-%d %H:%M:%S') ==="
      echo "swap ${swap_mb}MB (limit ${SWAP_LIMIT_MB}) compressor ${comp_mb}MB (limit ${COMPRESSOR_LIMIT_MB}) pressure ${level} disk ${disk_gb}GB (min ${MIN_DISK_GB})"
      echo "killing the solver; the machine is kept, the run is not"
    } >> "$LOG"
    pkill -f solve_all.sh
    pkill -f 'solver/forward.py'
    pkill -f 'solver/backward.py'
    # Pool workers have a generic multiprocessing command line and do NOT match the
    # names above — killing only the parent leaves 16 workers holding all the memory.
    pkill -f "multiprocessing.spawn"
    pkill -f "multiprocessing.resource_tracker"
    exit 1
  fi
  sleep 20
done
