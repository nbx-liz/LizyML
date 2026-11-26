#!/usr/bin/env bash
# #431: regenerate the pandas 3.0.1 / 2.3.3 H-0138 oracles from b297de4 under
# lightgbm 4.6.0 (control: must equal the committed bytes) and 4.7.0 (twice each).
set -euo pipefail
S=/tmp/claude-1000/-home-rem-repos-LizyML/8a6bdac2-e5b9-4be2-a2c4-1b26b17d088f/scratchpad
W=$S/wt-b297
OUT=$S/regen-431
mkdir -p "$OUT"
declare -A PYV=([3.0.1]=3.12.3 [2.3.3]=3.10.22)
for v in 3.0.1 2.3.3; do
    for lg in old new; do
        py=$S/venv-${PYV[$v]}-$lg/bin/python
        for run in a b; do
            (cd "$W" && rm -f tests/test_features/data/h0138_oracle_pandas-*.json.gz \
                && PYTHONHASHSEED=0 PYTHONPATH=.:docs/audits/2026-10-h0138/firing "$py" -B \
                    docs/audits/2026-10-h0138/oracle/make_oracle.py > "$OUT/oracle-$v-$lg-$run.log" 2>&1 \
                && cp tests/test_features/data/h0138_oracle_pandas-$v.json.gz "$OUT/oracle-$v-$lg-$run.json.gz")
            echo "done $v $lg $run"
        done
    done
done
cd /home/rem/repos/LizyML
for v in 3.0.1 2.3.3; do
    sha256sum tests/test_features/data/h0138_oracle_pandas-$v.json.gz "$OUT"/oracle-$v-*.json.gz | sed "s#$OUT/##"
done
