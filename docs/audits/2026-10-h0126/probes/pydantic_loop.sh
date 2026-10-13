#!/usr/bin/env bash
# Import LizyML under each pydantic minor (2.0 .. 2.12) with the other floors.
# Usage (repo root): bash pydantic_loop.sh <run1-dir> <out-dir>
set -uo pipefail
RUN1="${1:?run1 dir}"
OUT="${2:?out dir}"
mkdir -p "$OUT"
PY="$RUN1/venv/bin/python"
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache"
for minor in 0 1 2 3 4 5 6 7 8 9 10 11 12; do
    uv pip install --python "$PY" --resolution lowest-direct "pydantic==2.$minor.*" >"$OUT/install_2.$minor.txt" 2>&1 \
        || { printf '2.%s\tinstall-failed\n' "$minor"; continue; }
    got=$("$PY" -c "import importlib.metadata as m; print(m.version('pydantic'))")
    if "$PY" -c "import lizyml, lizyml.config.schema" >"$OUT/import_$got.txt" 2>&1; then
        printf '%s\timport-ok\n' "$got"
    else
        printf '%s\timport-fail\t%s\n' "$got" "$(tail -1 "$OUT/import_$got.txt")"
    fi
done
