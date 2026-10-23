#!/usr/bin/env bash
# Regenerate H-0138 oracles from the b297de4 tree: 3.0.1 (lock venv, must match
# the committed bytes) and 3.1.0 (venvci) twice; then the H-0139 overlays from
# the develop worktree for the generated 3.1.0 oracle.
set -euo pipefail
W=/tmp/claude-1000/wt-b297
D=/tmp/claude-1000/wt-389
OUT=/tmp/claude-1000/i389-gen
mkdir -p "$OUT"
cd "$W"
run() {  # <python> <tag>
    rm -f tests/test_features/data/h0138_oracle_pandas-*.json.gz
    PYTHONHASHSEED=0 PYTHONPATH=.:docs/audits/2026-10-h0138/firing "$1" -B \
        docs/audits/2026-10-h0138/oracle/make_oracle.py > "$OUT/$2.log" 2>&1
    cp tests/test_features/data/h0138_oracle_pandas-*.json.gz "$OUT/$2.json.gz"
}
run /home/rem/repos/LizyML/.venv/bin/python oracle-3.0.1
run /tmp/claude-1000/venvci/bin/python oracle-3.1.0-a
run /tmp/claude-1000/venvci/bin/python oracle-3.1.0-b
sha256sum "$OUT"/*.json.gz /home/rem/repos/LizyML/tests/test_features/data/h0138_oracle_pandas-3.0.1.json.gz
# Overlays: run from the develop worktree with the generated 3.1.0 oracle in place.
cd "$D"
cp "$OUT/oracle-3.1.0-a.json.gz" tests/test_features/data/h0138_oracle_pandas-3.1.0.json.gz
for tag in a b; do
    PYTHONHASHSEED=0 PYTHONPATH=.:docs/audits/2026-10-h0139/firing /tmp/claude-1000/venvci/bin/python -B \
        docs/audits/2026-10-h0139/oracle/make_overlay.py > "$OUT/overlay-3.1.0-$tag.log" 2>&1
    cp tests/test_features/data/h0139_overlay_pandas-3.1.0.json.gz "$OUT/overlay-3.1.0-$tag.json.gz"
done
sha256sum "$OUT"/overlay-*.json.gz
cat "$OUT/overlay-3.1.0-a.log"
