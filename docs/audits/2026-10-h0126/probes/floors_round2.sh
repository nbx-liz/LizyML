#!/usr/bin/env bash
# Round 2 on the run1 venv: add pytest-cov (test tooling), find the lowest
# optuna minor that has get_all_study_names, run the full suite there.
# Usage (repo root): bash floors_round2.sh <run1-dir> <out-dir>
set -uo pipefail

RUN1="${1:?run1 dir}"
OUT="${2:?out dir}"
mkdir -p "$OUT"
PY="$RUN1/venv/bin/python"
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache"

uv pip install --python "$PY" "pytest-cov==7.0.0" >"$OUT/cov_install.txt" 2>&1 || { tail -20 "$OUT/cov_install.txt"; exit 1; }

: >"$OUT/optuna_api.tsv"
FIRST=""
for v in 3.1 3.2 3.3 3.4 3.5 3.6 4.0 4.1 4.2 4.3 4.4 4.5 4.6 4.7; do
    uv pip install --python "$PY" --resolution lowest-direct "optuna==$v.*" >"$OUT/optuna_install_$v.txt" 2>&1 \
        || { printf '%s\tinstall-failed\n' "$v" >>"$OUT/optuna_api.tsv"; continue; }
    got=$("$PY" -c "import importlib.metadata as m; print(m.version('optuna'))")
    has=$("$PY" -c "import optuna; print(hasattr(optuna, 'get_all_study_names'))")
    printf '%s\t%s\n' "$got" "$has" >>"$OUT/optuna_api.tsv"
    if [ "$has" = "True" ]; then FIRST="$got"; break; fi
done
cat "$OUT/optuna_api.tsv"
[ -n "$FIRST" ] || { echo "no optuna minor has the API"; exit 2; }

"$PY" "$RUN1/versions.py" >"$OUT/resolved_suite.txt"
cat "$OUT/resolved_suite.txt"
"$PY" -m pytest -q -p no:cacheprovider -rf tests/ >"$OUT/pytest_full.txt" 2>&1
status=$?
tail -2 "$OUT/pytest_full.txt"
exit $status
