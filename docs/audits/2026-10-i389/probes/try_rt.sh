#!/usr/bin/env bash
# Generalized rule R (cast or transform) for pandas 3.1.0: regenerate the
# existing overlays (must stay byte-identical) and run the 3.1.0 oracle tests.
set -uo pipefail
D=/tmp/claude-1000/wt-389
G=/tmp/claude-1000/i389-gen
V=$TMPDIR/make_overlay_t.py
cd "$D"
cp "$G/oracle-3.1.0-a.json.gz" tests/test_features/data/h0138_oracle_pandas-3.1.0.json.gz
gen() {  # <python>
    PYTHONHASHSEED=0 PYTHONPATH=.:docs/audits/2026-10-h0139/firing "$1" -B "$V"
}
gen /tmp/claude-1000/venvci/bin/python
gen /home/rem/repos/LizyML/.venv/bin/python
gen /tmp/claude-1000/venv310/bin/python
gen /tmp/claude-1000/venvlow/bin/python
git status --short tests/test_features/data/
sha256sum tests/test_features/data/h0139_overlay_pandas-3.1.0.json.gz
PYTHONPATH="$D" /tmp/claude-1000/venvci/bin/python -m pytest -p no:cacheprovider -q --no-header -o addopts="" \
    tests/test_features/test_h0138_faithful_oracle.py -rf > /tmp/claude-1000/i389-rt.txt 2>&1
grep -E '^FAILED' /tmp/claude-1000/i389-rt.txt | sed -E 's/\[.*//' | sort | uniq -c
tail -1 /tmp/claude-1000/i389-rt.txt
