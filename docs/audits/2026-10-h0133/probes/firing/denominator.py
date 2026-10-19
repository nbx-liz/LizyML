"""Count the firing log's records outside the deliberate test files, by role,
and how many of them were stand-in columns or outside V (the firing rates).

Usage: python3 denominator.py <log.jsonl> [<log.jsonl> ...]
"""

from __future__ import annotations

import json
import sys
from collections import Counter

from summarize import DELIBERATE

recs = [json.loads(line) for p in sys.argv[1:] for line in open(p, encoding="utf-8")]
outside = [r for r in recs if r.get("kind") == "fit" and not r["test"].startswith(DELIBERATE)]
roles = Counter(r["role"] for r in outside)
print(f"records: {len(recs)}; outside the deliberate files: {len(outside)} {dict(roles)}")
print(f"  stand-in columns: {sum(1 for r in outside if r.get('stand_in'))}")
print(f"  LightGBM cannot write: {sum(1 for r in outside if r['unwritable'])}")
print(f"  outside V (implementation): {sum(1 for r in outside if r.get('impl_outside_v'))}")
