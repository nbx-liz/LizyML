"""Tally an H-0138 census log: records per surface, dtype and path.

    python docs/audits/2026-10-h0138/firing/summarize.py <census.jsonl>
"""

from __future__ import annotations

import json
import sys
from collections import Counter

FALLBACK = {"cast_raised", "predicate_false"}
ELIGIBLE = {"faithful", *FALLBACK}

tally: Counter[tuple[str, str, str]] = Counter()
tests: dict[str, set[str]] = {}
for line in open(sys.argv[1], encoding="utf-8"):
    record = json.loads(line)
    path = record.get("path") or "rule_error: " + record.get("rule_error", "")[:80]
    tally[(record["surface"], record.get("dtype", "?"), path)] += 1
    if record.get("path") in FALLBACK or "rule_error" in record:
        tests.setdefault(path, set()).add(record["test"].split(" ")[0])
total = sum(tally.values())
print(f"{total} column records")
for (surface, dtype, path), n in sorted(tally.items()):
    print(f"{n:7} {surface:8} {dtype:22} {path}")
for surface in sorted({s for s, _, _ in tally}):
    eligible = sum(n for (s, _, p), n in tally.items() if s == surface and p in ELIGIBLE)
    fired = sum(n for (s, _, p), n in tally.items() if s == surface and p in FALLBACK)
    print(f"Firing rate ({surface}): {fired}/{eligible} eligible columns build by key")
for path, names in sorted(tests.items()):
    print(f"--- {path}: {len(names)} tests")
    for name in sorted(names)[:40]:
        print("   ", name)
