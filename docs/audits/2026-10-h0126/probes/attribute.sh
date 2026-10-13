#!/usr/bin/env bash
# Attribute the 5 remaining floor failures: raise one package at a time from
# the floor set, run only those tests, restore the floors.
# Usage (repo root): bash attribute.sh <run1-dir> <out-dir>
set -uo pipefail

RUN1="${1:?run1 dir}"
OUT="${2:?out dir}"
mkdir -p "$OUT"
PY="$RUN1/venv/bin/python"
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache"

FLOORS=(numpy==1.24.0 pandas==2.0.0 scikit-learn==1.3.0 scipy==1.10.0 lightgbm==4.0.0 optuna==3.4.0 pydantic==2.11.0)
TESTS=(
    tests/test_tuning/test_space_merge.py::test_removed_training_dimension_admission
    tests/test_tuning/test_objective_identity.py::TestSummariseFrame::test_hash_pandas_object_alone_misses_the_auxiliary_cases
    tests/test_core/test_param_domain.py::test_unchanged_is_decided_by_type_and_not_by_printed_text
    tests/test_core/test_param_domain.py::test_a_value_with_no_plain_stand_in_is_refused_rather_than_rounded
    tests/test_codegen/test_calibration_params_codegen.py::test_generated_fitter_matches_the_runtime_calibrator
)

run_case() {
    local label="$1"
    "$PY" "$RUN1/versions.py" | tr '\n' ' ' >"$OUT/$label.versions"
    "$PY" -m pytest -q -p no:cacheprovider -rf "${TESTS[@]}" >"$OUT/$label.txt" 2>&1
    printf '== %s: %s\n' "$label" "$(tail -1 "$OUT/$label.txt")"
    grep -E '^FAILED' "$OUT/$label.txt" | sed 's/ - .*//; s/^FAILED tests\//  /'
}

uv pip install --python "$PY" "${FLOORS[@]}" >"$OUT/restore.txt" 2>&1 || { tail -20 "$OUT/restore.txt"; exit 1; }
run_case baseline

for up in numpy==1.26.4 pandas==2.2.3 scikit-learn==1.5.2 scipy==1.13.1 lightgbm==4.6.0 optuna==4.7.0 pydantic==2.12.5; do
    uv pip install --python "$PY" "$up" >"$OUT/up_$up.txt" 2>&1 || { echo "== $up: install failed"; tail -5 "$OUT/up_$up.txt"; continue; }
    run_case "up_$up"
    uv pip install --python "$PY" "${FLOORS[@]}" >"$OUT/restore.txt" 2>&1 || { tail -20 "$OUT/restore.txt"; exit 1; }
done
