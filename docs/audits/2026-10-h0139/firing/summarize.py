"""Summarise an H-0139 census log (``h0139_census.py``) into n/N counts.

    python docs/audits/2026-10-h0139/firing/summarize.py <census.jsonl>
"""

from __future__ import annotations

import json
import sys
from collections import Counter

records = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
paths: Counter[tuple[str, str, str]] = Counter()
errors = [r for r in records if "rule_error" in r]
for r in records:
    if "rule_error" in r:
        continue
    group = "temporal" if r["temporal"] else "non-temporal"
    paths[(r["surface"], group, r["path"])] += 1
print(f"records {len(records)}; rule errors {len(errors)}")
for (surface, group, path), n in sorted(paths.items()):
    print(f"{n:7} {surface:8} {group:13} {path}")
for surface in ("builder", "encoder"):
    total = sum(n for (s, _, _), n in paths.items() if s == surface)
    temporal = sum(n for (s, g, _), n in paths.items() if s == surface and g == "temporal")
    rebuilt = sum(
        n
        for (s, g, p), n in paths.items()
        if s == surface and g == "temporal" and p in ("cast_raised", "predicate_false")
    )
    refused = sum(n for (s, _, p), n in paths.items() if s == surface and p == "ym_refused")
    print(
        f"{surface}: rebuilt {rebuilt}/{total}, ym_refused {refused}/{total}, "
        f"union {rebuilt + refused}/{total} (temporal visits {temporal})"
    )
for r in errors[:20]:
    print("rule_error", r["surface"], r["column"], r["rule_error"][:150], r["test"][:80])
