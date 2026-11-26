#!/usr/bin/env bash
# #431 probe: under lightgbm 4.7.0, hash model_to_string() with the one line
# 4.7.0 adds ("[gpu_device_id_list: ]") removed. The patched generator
# requires that line to appear exactly once as a whole line in every model
# text (it raises otherwise), and an EXIT trap restores the generator. The
# stripped oracles are then compared with the committed 4.6.0 oracles by
# full_compare.py, expecting 0 hash changes; the unpatched 4.7.0 oracles are
# the negative control (74 hash changes per file).
set -euo pipefail
S=/tmp/claude-1000/-home-rem-repos-LizyML/8a6bdac2-e5b9-4be2-a2c4-1b26b17d088f/scratchpad
W=$S/wt-b297
D=$S/wt-431
G=docs/audits/2026-10-h0138/oracle
cp $W/$G/make_oracle.py $S/make_oracle.orig.py
trap 'cp $S/make_oracle.orig.py $W/$G/make_oracle.py' EXIT
python3 - "$W/$G/make_oracle.py" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
old = 'hashlib.sha256(booster.model_to_string().encode()).hexdigest()'
new = 'hashlib.sha256(_strip_gpu_line(booster.model_to_string()).encode()).hexdigest()'
anchor = 'def _model_case('
helper = (
    'def _strip_gpu_line(text: str) -> str:\n'
    '    lines = text.split("\\n")\n'
    '    marker = "[gpu_device_id_list: ]"\n'
    '    if lines.count(marker) != 1:\n'
    '        raise RuntimeError(f"expected one {marker} line, found {lines.count(marker)}")\n'
    '    lines.remove(marker)\n'
    '    return "\\n".join(lines)\n'
    '\n\n'
)
assert s.count(old) == 1 and s.count(anchor) == 1
open(p, "w").write(s.replace(old, new).replace(anchor, helper + anchor))
EOF
declare -A PYV=([3.0.1]=3.12.3 [2.3.3]=3.10.22)
for v in 3.0.1 2.3.3; do
    py=$S/venv-${PYV[$v]}-new/bin/python
    (cd "$W" && rm -f tests/test_features/data/h0138_oracle_pandas-*.json.gz \
        && PYTHONHASHSEED=0 PYTHONPATH=.:$G/../firing "$py" -B $G/make_oracle.py > $S/regen-431/strip-$v.log 2>&1 \
        && cp tests/test_features/data/h0138_oracle_pandas-$v.json.gz $S/regen-431/strip-$v.json.gz)
done
cd "$D"
for v in 3.0.1 2.3.3; do
    f=tests/test_features/data/h0138_oracle_pandas-$v.json.gz
    git show eec71dd:$f > $S/regen-431/committed-$v.json.gz
    echo "== $v, committed 4.6.0 vs this head (unpatched 4.7.0)"
    python3 docs/audits/2026-11-i431/full_compare.py $S/regen-431/committed-$v.json.gz $f 74
    echo "== $v, committed 4.6.0 vs stripped 4.7.0"
    python3 docs/audits/2026-11-i431/full_compare.py $S/regen-431/committed-$v.json.gz $S/regen-431/strip-$v.json.gz 0
done
