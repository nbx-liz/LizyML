"""Summarize an H-0136 census log: how often the proposed rule fires, per exit.

Usage: python summarize.py <census.jsonl>

A call "fires" when the prototype finds a break and the real call passes
today. A call the real code already refuses is counted apart.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict

records = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
by_surface: dict[str, list[dict]] = defaultdict(list)
for r in records:
    by_surface[r["surface"]].append(r)

print(f"records={len(records)}")
for surface in ("Model.export", "persistence.export", "export_code", "fit_warning"):
    rows = by_surface.get(surface, [])
    passes = ("ok", "silent")
    fires = [r for r in rows if r.get("breaks") and r["today"] in passes]
    already = [r for r in rows if r.get("breaks") and r["today"] not in passes]
    errors = [r for r in rows if r.get("rule_error")]
    fails = [r for r in rows if r.get("original_fails")]
    print(
        f"{surface:18s} calls={len(rows):5d} fires={len(fires):4d} "
        f"breaks_already_refused={len(already):3d} rule_error={len(errors):3d} "
        f"original_fails={len(fails):3d} today={dict(Counter(r['today'] for r in rows))}"
    )
    for r in fires:
        kinds = sorted({(b["state"], b["serializer"], b["kind"]) for b in r["breaks"]})
        print(f"    FIRES {r['test'].split(' ')[0]}  {kinds}")
    if surface == "export_code":
        newly = [r for r in rows if r["today"] == "ok" and r.get("today_check", "ok") != "ok"]
        print(f"    export_code passing today that check_category_values refuses: {len(newly)}")
        for r in newly:
            print(f"      {r['test'].split(' ')[0]}  {r['today_check']}")
    for r in errors:
        print(f"    RULE_ERROR {r['test'].split(' ')[0]}  {r['rule_error']}")
