#!/usr/bin/env bash
# (1) lowest LightGBM minor for which the early_stopping_round=0 case passes;
# (2) reverse attribution of the Platt TNC failure: everything raised, then
#     one package at a time put back to its floor.
# Usage (repo root): bash attribute2.sh <run1-dir> <out-dir>
set -uo pipefail

RUN1="${1:?run1 dir}"
OUT="${2:?out dir}"
mkdir -p "$OUT"
PY="$RUN1/venv/bin/python"
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache"

FLOORS=(numpy==1.24.0 pandas==2.0.0 scikit-learn==1.3.0 scipy==1.10.0 lightgbm==4.0.0 optuna==3.4.0 pydantic==2.11.0)
UPPER=(numpy==1.26.4 pandas==2.2.3 scikit-learn==1.5.2 scipy==1.13.1 lightgbm==4.6.0 optuna==4.7.0 pydantic==2.12.5)
ES_TEST="tests/test_tuning/test_space_merge.py::test_removed_training_dimension_admission"
CAL_TEST="tests/test_codegen/test_calibration_params_codegen.py::test_generated_fitter_matches_the_runtime_calibrator"

restore() { uv pip install --python "$PY" "$@" >"$OUT/restore.txt" 2>&1 || { tail -20 "$OUT/restore.txt"; exit 1; }; }
one() {
    local label="$1" test="$2"
    "$PY" -m pytest -q -p no:cacheprovider "$test" >"$OUT/$label.txt" 2>&1
    printf '== %s: %s\n' "$label" "$(tail -1 "$OUT/$label.txt")"
}

echo "--- (1) LightGBM minors, early_stopping_round=0 case"
restore "${FLOORS[@]}"
for v in 4.0.0 4.1.0 4.2.0 4.3.0 4.4.0 4.5.0 4.6.0; do
    uv pip install --python "$PY" "lightgbm==$v" >"$OUT/lgb_$v.txt" 2>&1 || { echo "== lightgbm $v: install failed"; continue; }
    one "lgb_$v" "$ES_TEST"
done

echo "--- (2) Platt TNC: all raised, then one back at its floor"
restore "${UPPER[@]}"
one "cal_all_upper" "$CAL_TEST"
for i in "${!FLOORS[@]}"; do
    restore "${UPPER[@]}"
    uv pip install --python "$PY" "${FLOORS[$i]}" >"$OUT/down.txt" 2>&1 || { echo "== down ${FLOORS[$i]}: install failed"; continue; }
    one "cal_down_${FLOORS[$i]}" "$CAL_TEST"
done
restore "${FLOORS[@]}"
