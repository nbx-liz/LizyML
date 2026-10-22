"""Summarise an #393 census log into n/N counts.

    python docs/audits/2026-10-i393/firing/summarize.py <census.jsonl>
"""

from __future__ import annotations

import json
import sys
from collections import Counter

records = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
declared = [r for r in records if r["declared"]]
fired = [r for r in declared if r.get("non_str")]
print(f"LGBMAdapter.fit calls: {len(records)}")
print(f"  with a categorical declaration: {len(declared)}")
print(f"  declaring a non-str label: {len(fired)}/{len(declared)}")
types: Counter[str] = Counter(t for r in records for t in r["column_types"])
print(f"  frame column label types (calls): {dict(types)}")
for r in fired[:20]:
    print("   ", r["test"].rsplit(" ", 1)[0][:110], r["non_str"])
