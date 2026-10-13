#!/usr/bin/env bash
# #295 / H-0126 probe: install every declared direct floor (core + extras) with
# --resolution lowest-direct, then find the lowest pydantic minor that imports
# LizyML, then run the full suite there.
#
# Usage (repo root): bash floors_probe.sh <out-dir>
set -uo pipefail

OUT="${1:?usage: floors_probe.sh <out-dir>}"
mkdir -p "$OUT"
VENV="$OUT/venv"
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache"

uv venv --python 3.11 "$VENV" >"$OUT/venv.txt" 2>&1 || { echo "venv failed"; cat "$OUT/venv.txt"; exit 1; }

# Test tooling is pinned to the locked versions so a tooling floor cannot be
# what fails; every [project] dependency and extra resolves to its floor.
uv pip install --python "$VENV/bin/python" --resolution lowest-direct \
    -e ".[tuning,explain,plots,calibration]" \
    "pytest==9.0.2" "pyarrow==23.0.1" "nbconvert==7.17.0" "ipykernel==7.2.0" \
    >"$OUT/install.txt" 2>&1 || { echo "install failed"; tail -40 "$OUT/install.txt"; exit 1; }

cat >"$OUT/versions.py" <<'PY'
import importlib.metadata as m
for name in ("pydantic", "pydantic-core", "numpy", "pandas", "scikit-learn", "lightgbm",
             "pyyaml", "joblib", "optuna", "shap", "plotly", "scipy", "numba"):
    try:
        print(f"{name}=={m.version(name)}")
    except m.PackageNotFoundError:
        print(f"{name}==<absent>")
PY
"$VENV/bin/python" "$OUT/versions.py" >"$OUT/resolved_initial.txt"
cat "$OUT/resolved_initial.txt"

: >"$OUT/pydantic_import.tsv"
FIRST=""
for minor in 0 1 2 3 4 5 6 7 8 9 10 11 12; do
    uv pip install --python "$VENV/bin/python" --resolution lowest-direct "pydantic==2.$minor.*" \
        >"$OUT/pyd_install_2.$minor.txt" 2>&1 || { printf '2.%s\tinstall-failed\n' "$minor" >>"$OUT/pydantic_import.tsv"; continue; }
    got=$("$VENV/bin/python" -c "import importlib.metadata as m; print(m.version('pydantic'))")
    if "$VENV/bin/python" -c "import lizyml, lizyml.config.schema" >"$OUT/pyd_import_$got.txt" 2>&1; then
        printf '%s\timport-ok\n' "$got" >>"$OUT/pydantic_import.tsv"
        [ -z "$FIRST" ] && FIRST="$got"
    else
        printf '%s\timport-fail\t%s\n' "$got" "$(tail -1 "$OUT/pyd_import_$got.txt")" >>"$OUT/pydantic_import.tsv"
    fi
done
cat "$OUT/pydantic_import.tsv"
[ -n "$FIRST" ] || { echo "no pydantic minor imports"; exit 2; }

uv pip install --python "$VENV/bin/python" "pydantic==$FIRST" >"$OUT/pyd_final_install.txt" 2>&1
"$VENV/bin/python" "$OUT/versions.py" >"$OUT/resolved_suite.txt"
cat "$OUT/resolved_suite.txt"

"$VENV/bin/python" -m pytest -q -p no:cacheprovider -x --maxfail=50 tests/ >"$OUT/pytest.txt" 2>&1
status=$?
tail -40 "$OUT/pytest.txt"
exit $status
