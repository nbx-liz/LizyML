#!/usr/bin/env bash
# heavy.sh -- the only way to run anything that trains models
# (pytest, probes, negative controls). After the 2026-10-13/14 overload:
#   1. serialised with every other heavy job (run-exclusive.sh's flock);
#   2. capped threads (LightGBM / OpenMP / BLAS use every core by default);
#   3. own process group, wall-clock timeout and address-space ceiling
#      (~/.claude/scripts/run-heavy.sh), so a stopped or hung job cannot linger.
#
#   bash heavy.sh <label> <timeout-sec> <command...>
set -uo pipefail
LABEL="${1:?label}"; TIMEOUT="${2:?timeout seconds}"; shift 2
THREADS="${HEAVY_THREADS:-4}"
export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
export LIGHTGBM_NUM_THREADS="$THREADS"
exec bash /home/rem/repos/LizyML/docs/audits/2026-09-defect-discovery/instruments/run-exclusive.sh "$LABEL" \
    bash /home/rem/.claude/scripts/run-heavy.sh --mem-mb 24576 --timeout "$TIMEOUT" -- "$@"
