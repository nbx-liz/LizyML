#!/usr/bin/env bash
# H-0144 acceptance 9: with the 3.1.0 files, the full suite (slow included) on
# pandas 3.1.0 in the latest-deps mode, and on the lock environment.
#   bash docs/audits/2026-10-i389/probes/accept_310.sh
set -uo pipefail
D=/tmp/claude-1000/wt-389
cd "$D"
run() {  # <label> <python> [env...]
    local label=$1 py=$2; shift 2
    env "$@" PYTHONPATH="$D" "$py" -m pytest -p no:cacheprovider -q --no-header -m "" tests \
        > "/tmp/claude-1000/i389-accept310-$label.txt" 2>&1
    echo "== $label"
    grep -E '^H-0138 oracle|[0-9]+ (passed|failed)' "/tmp/claude-1000/i389-accept310-$label.txt" | tail -2
    grep -E '^(FAILED|ERROR)' "/tmp/claude-1000/i389-accept310-$label.txt" | head -10
}
run pandas310 /tmp/claude-1000/venvci/bin/python GITHUB_ACTIONS=true GITHUB_JOB=quality-latest-deps LIZYML_H0138_ORACLE_MODE=latest-same-minor
run lock "$D/.venv/bin/python"
