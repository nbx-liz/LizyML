#!/usr/bin/env bash
# Serialise CPU-heavy jobs on this host. One at a time, enforced by a lock.
#
# Why this exists, twice over:
#   1. Three traced pytest runs were started concurrently. LightGBM uses every
#      core per `lgb.train`, so the box went to load 90 on 32 cores and a suite
#      that takes 259 s took 22 minutes to reach test 24. The cause was
#      misdiagnosed as an external process, because `ps` inside this sandbox
#      shows only the current shell.
#   2. After writing that down as "one CPU-heavy job at a time", R5 and R6 were
#      started concurrently anyway. Load 63.5 on 32 cores.
#
# A note in a memory file does not fire at the moment a job is launched. A lock
# does. Per the mechanize-on-recurrence rule, the second occurrence is the point
# at which documentation must be replaced by something that executes.
#
#   ./run-exclusive.sh <label> <command...>
#
# Blocks until the lock is free, prints who holds it while waiting, and always
# releases -- including on interrupt.

set -uo pipefail

LOCK="/tmp/lizyml-discovery-plan/.heavy.lock"
LABEL="${1:?usage: run-exclusive.sh <label> <command...>}"
shift

exec 9>"$LOCK"

if ! flock -n 9; then
    holder="$(cat "${LOCK}.owner" 2>/dev/null || echo unknown)"
    echo "[run-exclusive] waiting: '$holder' holds the lock" >&2
    flock 9
fi

printf '%s (pid %s, since %s)\n' "$LABEL" "$$" "$(date -Is)" > "${LOCK}.owner"
trap 'rm -f "${LOCK}.owner"' EXIT INT TERM

echo "[run-exclusive] $LABEL starting; load was $(cut -d' ' -f1 /proc/loadavg)" >&2
start=$(date +%s)
"$@"
status=$?
echo "[run-exclusive] $LABEL finished in $(( $(date +%s) - start ))s, exit $status" >&2
exit $status
