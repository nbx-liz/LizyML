#!/usr/bin/env bash
# #431: install the 4.7.0 oracles on the PR head, regenerate the 3.0.1 / 2.3.3
# H-0139 overlays under 4.7.0 (twice) and check their bytes are unchanged.
set -euo pipefail
S=/tmp/claude-1000/-home-rem-repos-LizyML/8a6bdac2-e5b9-4be2-a2c4-1b26b17d088f/scratchpad
D=$S/wt-431
OUT=$S/regen-431
cp $OUT/oracle-3.0.1-new-a.json.gz $D/tests/test_features/data/h0138_oracle_pandas-3.0.1.json.gz
cp $OUT/oracle-2.3.3-new-a.json.gz $D/tests/test_features/data/h0138_oracle_pandas-2.3.3.json.gz
declare -A PYV=([3.0.1]=3.12.3 [2.3.3]=3.10.22)
cd "$D"
for v in 3.0.1 2.3.3; do git show HEAD:tests/test_features/data/h0139_overlay_pandas-$v.json.gz > $OUT/overlay-$v-committed.json.gz; done
for v in 3.0.1 2.3.3; do
    py=$S/venv-${PYV[$v]}-new/bin/python
    for run in a b; do
        PYTHONHASHSEED=0 PYTHONPATH=.:docs/audits/2026-10-h0139/firing "$py" -B \
            docs/audits/2026-10-h0139/oracle/make_overlay.py > $OUT/overlay-$v-$run.log 2>&1
        cp tests/test_features/data/h0139_overlay_pandas-$v.json.gz $OUT/overlay-$v-$run.json.gz
        cp $OUT/overlay-$v-committed.json.gz tests/test_features/data/h0139_overlay_pandas-$v.json.gz
    done
    sha256sum tests/test_features/data/h0139_overlay_pandas-$v.json.gz $OUT/overlay-$v-a.json.gz $OUT/overlay-$v-b.json.gz | sed "s#$OUT/##"
done
git status --short --untracked-files=no
