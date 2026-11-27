"""Print the #423 census summary from budget_census.py's JSON output.

    python summarize_census.py census.json
"""

from __future__ import annotations

import json
import sys

GIB = 2**30

for r in json.load(open(sys.argv[1])):
    b = sorted(t["bytes"] for t in r["trials"])
    n = len(b)
    over = "; ".join(f"> {g} GiB: {sum(x > g * GIB for x in b)}/{n}" for g in (1, 2, 4, 8))
    print(
        f"{r['dataset']} ({r['task']}, {r['rows']} rows, {r['features']} features): "
        f"{n} trials; max {b[-1] / GIB:.2f} GiB; {over}"
    )
