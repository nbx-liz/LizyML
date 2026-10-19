"""Summarize the H-0134 load census log (``h0134_load_census.py``).

Usage: python3 summarize.py <log.jsonl>

Prints the number of Model.load calls (raised, by error code, and returned),
then the loads of tuned models, split by whether the history was
restored and by format_version, and the tests that loaded a tuned model
without its history. It exits 1 on any ``error`` record or an unknown kind.
"""

from __future__ import annotations

import json
import sys
from collections import Counter

records = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
errors = [r for r in records if r["kind"] not in ("load", "load_failed")]
loads = [r for r in records if r["kind"] == "load"]
failed = [r for r in records if r["kind"] == "load_failed"]
tuned = [r for r in loads if r["tuned"]]
print(f"Model.load calls: {len(loads) + len(failed)}")
print(f"  raised: {len(failed)}")
for code, n in sorted(Counter(r["code"] for r in failed).items()):
    print(f"    {code}: {n}")
print(f"  returned: {len(loads)}; of tuned models: {len(tuned)}")
by = Counter((r["format_version"], r["history"]) for r in tuned)
for (fv, history), n in sorted(by.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
    print(f"  format_version={fv} history={'yes' if history else 'no'}: {n}")
without = sorted({r["test"].split(" ")[0].split("::")[0] for r in tuned if not r["history"]})
print(f"files loading a tuned model without history: {len(without)}")
for name in without:
    print(f"  {name}")
print(f"error records: {len(errors)}")
sys.exit(1 if errors else 0)
