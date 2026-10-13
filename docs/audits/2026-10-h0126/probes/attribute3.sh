#!/usr/bin/env bash
# Platt TNC case: from the "upper" set, raise one package toward the lock.
# Usage (repo root): bash attribute3.sh <run1-dir> <out-dir>
set -uo pipefail

RUN1="${1:?run1 dir}"
OUT="${2:?out dir}"
mkdir -p "$OUT"
PY="$RUN1/venv/bin/python"
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache"

UPPER=(numpy==1.26.4 pandas==2.2.3 scikit-learn==1.5.2 scipy==1.13.1 lightgbm==4.6.0 optuna==4.7.0 pydantic==2.12.5)
FLOORS=(numpy==1.24.0 pandas==2.0.0 scikit-learn==1.3.0 scipy==1.10.0 lightgbm==4.0.0 optuna==3.4.0 pydantic==2.11.0)
CAL_TEST="tests/test_codegen/test_calibration_params_codegen.py::test_generated_fitter_matches_the_runtime_calibrator"

restore() { uv pip install --python "$PY" "$@" >"$OUT/restore.txt" 2>&1 || { tail -20 "$OUT/restore.txt"; exit 1; }; }
one() {
    "$PY" -m pytest -q -p no:cacheprovider "$CAL_TEST" >"$OUT/$1.txt" 2>&1
    printf '== %s: %s | %s\n' "$1" "$(tail -1 "$OUT/$1.txt")" "$("$PY" "$RUN1/versions.py" | grep -E '^(numpy|pandas|scikit-learn|scipy)==' | tr '\n' ' ')"
}

for up in scipy==1.14.1 scipy==1.15.3 scipy==1.16.3 scipy==1.17.1 numpy==2.0.2 numpy==2.4.2 scikit-learn==1.8.0 pandas==3.0.1; do
    restore "${UPPER[@]}"
    uv pip install --python "$PY" "$up" >"$OUT/up.txt" 2>&1 || { echo "== $up: install failed"; tail -3 "$OUT/up.txt"; continue; }
    one "up_$up"
done
restore "${FLOORS[@]}"
