"""List the tests whose temporal columns fire the H-0139 activation in a census log.

    python docs/audits/2026-10-h0139/firing/by_test.py <census.jsonl>
"""

from __future__ import annotations

import json
import sys
from collections import Counter

fired: Counter[tuple[str, str, str]] = Counter()
for line in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(line)
    if "rule_error" in r:
        continue
    if r["path"] == "ym_refused" or (
        r["temporal"] and r["path"] in ("cast_raised", "predicate_false")
    ):
        test = r["test"].rsplit(" ", 1)[0]
        fired[(test, r["surface"], r["path"])] += 1
for (test, surface, path), n in sorted(fired.items()):
    print(f"{n:4} {surface:8} {path:16} {test}")
