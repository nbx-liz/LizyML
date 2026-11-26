#!/usr/bin/env bash
# #431 probe: under lightgbm 4.7.0, hash model_to_string() with the one line
# 4.7.0 adds ("[gpu_device_id_list: ]") removed; if every case then equals the
# committed 4.6.0 oracle, that line is the whole difference. Negative control:
# the unpatched generator (already run) differs in all 74 model_sha256.
set -euo pipefail
S=/tmp/claude-1000/-home-rem-repos-LizyML/8a6bdac2-e5b9-4be2-a2c4-1b26b17d088f/scratchpad
W=$S/wt-b297
G=docs/audits/2026-10-h0138/oracle
cp $W/$G/make_oracle.py $S/make_oracle.orig.py
python3 - "$W/$G/make_oracle.py" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
old = 'hashlib.sha256(booster.model_to_string().encode()).hexdigest()'
new = 'hashlib.sha256(booster.model_to_string().replace("[gpu_device_id_list: ]\\n", "", 1).encode()).hexdigest()'
assert s.count(old) == 1
open(p, "w").write(s.replace(old, new))
EOF
declare -A PYV=([3.0.1]=3.12.3 [2.3.3]=3.10.22)
for v in 3.0.1 2.3.3; do
    py=$S/venv-${PYV[$v]}-new/bin/python
    (cd "$W" && rm -f tests/test_features/data/h0138_oracle_pandas-*.json.gz \
        && PYTHONHASHSEED=0 PYTHONPATH=.:$G/../firing "$py" -B $G/make_oracle.py > $S/regen-431/strip-$v.log 2>&1 \
        && cp tests/test_features/data/h0138_oracle_pandas-$v.json.gz $S/regen-431/strip-$v.json.gz)
done
cp $S/make_oracle.orig.py $W/$G/make_oracle.py
cd /home/rem/repos/LizyML
for v in 3.0.1 2.3.3; do
    echo "== $v"
    python3 $S/diff_oracle.py tests/test_features/data/h0138_oracle_pandas-$v.json.gz $S/regen-431/strip-$v.json.gz
done
