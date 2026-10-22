"""Summarise an H-0143 census log.

    python docs/audits/2026-10-h0143/probes/census_summary.py <census.jsonl>
"""

import json
import sys
from collections import Counter

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
differs = [r for r in rows if r["differs"]]
errors = sum(1 for r in rows if r["error"])
print(f"state_changes calls: {len(rows)}")
print(f"  with any change today: {sum(1 for r in rows if r['changes'])}")
print(f"  changed by H-0143: {len(differs)}; prototype errors: {errors}")
for name, n in Counter(r["test"].split("::")[0] for r in differs).most_common():
    print(f"    {n:5d}  {name}")
