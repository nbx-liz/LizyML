#!/usr/bin/env bash
# H-0144 PR 2: onboard pandas 3.1.0 with the CONTRIBUTING.md procedure.
#   bash docs/audits/2026-10-i389/probes/onboard_310.sh
set -euo pipefail
W=/tmp/claude-1000/wt-b297     # b297de4 + develop's docs/audits/2026-10-h0138/{oracle,firing}
D=/tmp/claude-1000/wt-389
PY=/tmp/claude-1000/venvci/bin/python
OUT=/tmp/claude-1000/i389-onboard
mkdir -p "$OUT"
(cd "$W" && "$PY" -c "import sys, pandas, numpy, lightgbm, pyarrow; print('env: python', sys.version.split()[0], 'pandas', pandas.__version__, 'numpy', numpy.__version__, 'lightgbm', lightgbm.__version__, 'pyarrow', pyarrow.__version__)")
# step 1: the oracle from b297de4, twice
for run in a b; do
    (cd "$W" && rm -f tests/test_features/data/h0138_oracle_pandas-*.json.gz \
        && PYTHONHASHSEED=0 PYTHONPATH=.:docs/audits/2026-10-h0138/firing "$PY" -B \
            docs/audits/2026-10-h0138/oracle/make_oracle.py > "$OUT/oracle-$run.log" 2>&1 \
        && cp tests/test_features/data/h0138_oracle_pandas-3.1.0.json.gz "$OUT/oracle-$run.json.gz")
done
cmp "$OUT/oracle-a.json.gz" "$OUT/oracle-b.json.gz"
cp "$OUT/oracle-a.json.gz" "$D/tests/test_features/data/h0138_oracle_pandas-3.1.0.json.gz"
# step 2: the overlay with the frozen generator, twice
cd "$D"
for run in a b; do
    PYTHONHASHSEED=0 PYTHONPATH=.:docs/audits/2026-10-h0139/firing "$PY" -B \
        docs/audits/2026-10-h0139/oracle/make_overlay.py 2>/dev/null | tee "$OUT/overlay-$run.log"
    cp tests/test_features/data/h0139_overlay_pandas-3.1.0.json.gz "$OUT/overlay-$run.json.gz"
done
cmp "$OUT/overlay-a.json.gz" "$OUT/overlay-b.json.gz"
sha256sum "$OUT"/oracle-a.json.gz "$OUT"/overlay-a.json.gz | sed "s#$OUT/##"
git status --short tests/test_features/data/
