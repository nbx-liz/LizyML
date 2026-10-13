#!/usr/bin/env bash
# The quality-lowest-deps lane of .github/workflows/ci.yml, step for step, in a
# fresh venv outside the repo (the lane uses ./.venv; here --python points at it).
# Usage (repo root): bash lane_local.sh <out-dir> [python-version]
set -uo pipefail
OUT="${1:?out dir}"
PYV="${2:-3.11}"
mkdir -p "$OUT"
VENV="$OUT/.venv"
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache"
rm -rf "$VENV"
uv venv --python "$PYV" "$VENV" >"$OUT/1_venv.txt" 2>&1 || { cat "$OUT/1_venv.txt"; exit 1; }
uv pip install --python "$VENV/bin/python" --resolution lowest-direct -e ".[tuning,explain,plots,calibration]" \
    >"$OUT/2_floors.txt" 2>&1 || { tail -30 "$OUT/2_floors.txt"; exit 1; }
uv export --frozen --only-group lowest-lane --no-hashes --no-emit-project -o "$OUT/tools.txt" >"$OUT/3_export.txt" 2>&1 || { cat "$OUT/3_export.txt"; exit 1; }
uv pip install --python "$VENV/bin/python" -r "$OUT/tools.txt" >"$OUT/3_tools.txt" 2>&1 || { tail -30 "$OUT/3_tools.txt"; exit 1; }
"$VENV/bin/python" scripts/check_dependency_floors.py | tee "$OUT/4_check.txt"
check=${PIPESTATUS[0]}
echo "check exit=$check"
[ "$check" -eq 0 ] || exit "$check"
"$VENV/bin/python" -m pytest tests/ -q -p no:cacheprovider -rf >"$OUT/5_pytest.txt" 2>&1
status=$?
grep -E '^FAILED|(passed|failed).* in [0-9.]+s' "$OUT/5_pytest.txt" | sed 's/ - .*//'
exit $status
