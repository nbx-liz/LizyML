#!/usr/bin/env bash
# H-0144 acceptance 5-7 on the latest-deps environments, in the same-minor
# mode with the CI job named, plus the strict lanes 2.3.3 / 2.0.0.
#   bash docs/audits/2026-10-i389/probes/accept_latest.sh
set -uo pipefail
D=/tmp/claude-1000/wt-389
T=tests/test_features/test_h0138_faithful_oracle.py
CI_ENV=(GITHUB_ACTIONS=true GITHUB_JOB=quality-latest-deps LIZYML_H0138_ORACLE_MODE=latest-same-minor)
run() {  # <label> <python> [env...]
    local label=$1 py=$2; shift 2
    echo "== $label"
    (cd "$D" && env "$@" PYTHONPATH="$D" "$py" -m pytest -p no:cacheprovider -q --no-header \
        -o addopts="" -rs "$T" 2>&1 | grep -E '^H-0138 oracle|passed|failed|^SKIPPED' | sed -E 's/^(SKIPPED \[[0-9]+\]) [^:]*:[0-9]+:/\1/' | sort | uniq -c | sort -rn | head -8)
}
# acceptance 5: pandas 3.0.6 against 3.0.1, models skipped and never computed
run "3.0.6 same-minor" /tmp/claude-1000/venvlatest/bin/python "${CI_ENV[@]}"
# acceptance 2: the same environment without the mode fails (no exact oracle)
run "3.0.6 strict" /tmp/claude-1000/venvlatest/bin/python
# acceptance 2: the mode outside the named job fails
run "3.0.6 mode outside the job" /tmp/claude-1000/venvlatest/bin/python LIZYML_H0138_ORACLE_MODE=latest-same-minor
# acceptance 7: pandas 3.1.0 on this tree: no same-minor oracle, red
run "3.1.0 same-minor" /tmp/claude-1000/venvci/bin/python "${CI_ENV[@]}"
# the blocking lanes stay strict and green
run "2.3.3 strict" /tmp/claude-1000/venv310/bin/python
run "2.0.0 strict" /tmp/claude-1000/venvlow/bin/python
