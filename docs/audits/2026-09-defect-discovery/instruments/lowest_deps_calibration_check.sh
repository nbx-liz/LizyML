#!/usr/bin/env bash
# PR 3c (H-0100): run the calibration tests against the lowest supported
# scikit-learn and scipy, and record what was actually resolved.
#
# Why this exists: the CI "lowest-direct" lane runs
#   uv sync --frozen --dev --resolution lowest-direct
# and --frozen installs the lockfile as it is, so that lane is not evidence that
# scikit-learn 1.3 / scipy 1.10 were ever exercised. The Platt fit, the method
# table and the unknown-option probe were measured on scipy 1.17 only.
#
# Usage (from the repository root):
#   bash docs/audits/2026-09-defect-discovery/instruments/lowest_deps_calibration_check.sh <out-dir>
#
# Writes <out-dir>/resolved.txt (the installed versions) and <out-dir>/pytest.txt.

set -uo pipefail

OUT="${1:?usage: lowest_deps_calibration_check.sh <out-dir>}"
mkdir -p "$OUT"
VENV="$OUT/venv"

# Python 3.11: the floors being checked are scikit-learn 1.3 and scipy 1.10, both
# of which publish 3.11 wheels. The CI lane uses 3.10, but this host has no 3.10
# and its uv Python store is read-only; the Python minor is not what is under test.
uv venv --python 3.11 "$VENV" >"$OUT/venv.txt" 2>&1 || { echo "venv failed"; cat "$OUT/venv.txt"; exit 1; }

# The package with its calibration extra at the lowest direct versions, plus the
# test tooling. scikit-learn 1.3 / scipy 1.10 are pinned explicitly so the run
# cannot silently resolve upward. pydantic is held at the locked 2.12.5 because
# the declared floor 2.0 cannot import the config schema at all (#295, not
# caused by PR 3c); without this pin the run measures that defect instead.
uv pip install --python "$VENV/bin/python" --resolution lowest-direct \
    -e ".[calibration]" \
    "scikit-learn==1.3.*" "scipy==1.10.*" \
    "pydantic==2.12.5" \
    "pytest>=8.0" "pyarrow>=14.0" "optuna>=3.0" \
    >"$OUT/install.txt" 2>&1 || { echo "install failed"; tail -30 "$OUT/install.txt"; exit 1; }

"$VENV/bin/python" - >"$OUT/resolved.txt" <<'PY'
import importlib.metadata as m
for name in ("scikit-learn", "scipy", "numpy", "pandas", "lightgbm", "pydantic", "joblib"):
    print(f"{name}=={m.version(name)}")
PY

"$VENV/bin/python" -m pytest -q -p no:cacheprovider \
    -W error::DeprecationWarning -W error::FutureWarning \
    tests/test_calibration/test_platt_mle.py \
    tests/test_calibration/test_calibration_param_contract.py \
    tests/test_calibration/test_beta_calibration.py \
    tests/test_calibration/test_calibration.py \
    tests/test_core/test_calibration_params_reach.py \
    tests/test_codegen/test_calibration_params_codegen.py \
    >"$OUT/pytest.txt" 2>&1
status=$?

cat "$OUT/resolved.txt"
tail -15 "$OUT/pytest.txt"
exit $status
