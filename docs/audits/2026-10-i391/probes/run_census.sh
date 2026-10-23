#!/usr/bin/env bash
# #391: full-suite census of the prototype collapse refusal on the NumPy 2.x
# lock lane and the NumPy 1.24 lowest-direct lane (slow tests included).
#   bash docs/audits/2026-10-i391/probes/run_census.sh
set -uo pipefail
D=/tmp/claude-1000/wt-391
P=docs/audits/2026-10-i391/probes
cd "$D"
for lane in lock:/home/rem/repos/LizyML/.venv/bin/python low:/tmp/claude-1000/venvlow/bin/python; do
    name=${lane%%:*} py=${lane#*:}
    COLLAPSE_CENSUS_OUT="$P/census${TAG:-}_$name.json" PYTHONPATH="$D:$D/$P" "$py" -m pytest \
        -p no:cacheprovider -p collapse_census -q --no-header -m "" tests \
        > "$P/census${TAG:-}_${name}_pytest.txt" 2>&1
    echo "== $name: $(grep -E "[0-9]+ (passed|failed)" "$P/census${TAG:-}_${name}_pytest.txt" | tail -1)"
    grep -E "^(FAILED|ERROR)" "$P/census${TAG:-}_${name}_pytest.txt" | head -20
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('calls', d['calls'], 'fires', d['fires'], 'tests calling', d['tests_calling'], 'tests firing', len(d['tests_firing']))" "$P/census${TAG:-}_$name.json"
done
