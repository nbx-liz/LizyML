#!/usr/bin/env bash
# H-0144 acceptance 1 / provenance: regenerate the three shipped H-0138 oracles
# from b297de4 (twice each) and the three H-0139 overlays with the frozen
# generator, and print each environment.
#   bash docs/audits/2026-10-i389/probes/regen_existing.sh
set -euo pipefail
W=/tmp/claude-1000/wt-b297          # b297de4 + develop's docs/audits/2026-10-h0138/{oracle,firing}
D=/tmp/claude-1000/wt-389           # the branch under review
OUT=/tmp/claude-1000/i389-regen
mkdir -p "$OUT"
declare -A PY=(
    [3.0.1]=/home/rem/repos/LizyML/.venv/bin/python
    [2.3.3]=/tmp/claude-1000/venv310/bin/python
    [2.0.0]=/tmp/claude-1000/venvlow/bin/python
)
for v in 3.0.1 2.3.3 2.0.0; do
    py=${PY[$v]}
    (cd "$W" && "$py" -c "import sys, pandas, numpy, lightgbm; import importlib.util as u; pa = u.find_spec('pyarrow'); print('$v env: python', sys.version.split()[0], 'pandas', pandas.__version__, 'numpy', numpy.__version__, 'lightgbm', lightgbm.__version__, 'pyarrow', __import__('pyarrow').__version__ if pa else None)")
    for run in a b; do
        (cd "$W" && rm -f tests/test_features/data/h0138_oracle_pandas-*.json.gz \
            && PYTHONHASHSEED=0 PYTHONPATH=.:docs/audits/2026-10-h0138/firing "$py" -B \
                docs/audits/2026-10-h0138/oracle/make_oracle.py > "$OUT/oracle-$v-$run.log" 2>&1 \
            && cp tests/test_features/data/h0138_oracle_pandas-$v.json.gz "$OUT/oracle-$v-$run.json.gz")
    done
    (cd "$D" && PYTHONHASHSEED=0 PYTHONPATH=.:docs/audits/2026-10-h0139/firing "$py" -B \
        docs/audits/2026-10-h0139/oracle/make_overlay.py > "$OUT/overlay-$v.log" 2>&1 \
        && cp tests/test_features/data/h0139_overlay_pandas-$v.json.gz "$OUT/overlay-$v.json.gz" \
        && git checkout -q -- tests/test_features/data/h0139_overlay_pandas-$v.json.gz)
done
cd "$D"
for v in 3.0.1 2.3.3 2.0.0; do
    sha256sum "$OUT/oracle-$v-a.json.gz" "$OUT/oracle-$v-b.json.gz" tests/test_features/data/h0138_oracle_pandas-$v.json.gz \
        "$OUT/overlay-$v.json.gz" tests/test_features/data/h0139_overlay_pandas-$v.json.gz | sed "s#$OUT/##"
done
git status --short tests/test_features/data/
