"""Summarize the H-0135 load census: how often each proposed rule would change a load.

Usage: python3 summarize.py <log.jsonl>

The rules evaluated are the ones in the H-0135 design draft:
- A: an artifact of format >= 3 must carry well-formed checksums. In any
  format, a present ``checksums`` key must be well formed (#374).
- B: in format >= 3, ``fit_result.pkl`` and ``refit_model.pkl`` must be listed.
- C: ``analysis_context.pkl`` is read only when listed, if checksums exist
  (#373). A listed file that is missing is refused.

For each rule the script counts the loads it would refuse or change, split by
the outcome the load has today, and names the test files involved. It exits
1 on an unknown record kind. Loads of a directory without a
readable metadata.json (tests of that refusal) are counted, not dropped.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict

records = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
bad = [r for r in records if r.get("kind") != "load"]
unreadable = [r for r in records if r.get("kind") == "load" and "unreadable" in r]
loads = [r for r in records if r.get("kind") == "load" and "unreadable" not in r]


def test_file(r: dict) -> str:
    return r["test"].split(" ")[0].split("::")[0]


rules: dict[str, list[dict]] = defaultdict(list)
for r in loads:
    fv, cs, listed, present = r["format_version"], r["checksums"], r["listed"], set(r["present"])
    if isinstance(fv, int) and fv >= 3 and cs != "ok":
        rules["A: format>=3 without well-formed checksums"].append(r)
    if cs in ("non_object", "bad_object"):
        rules["A: checksums present but malformed"].append(r)
    if isinstance(fv, int) and fv >= 3 and cs == "ok":
        missing = {"fit_result.pkl", "refit_model.pkl"} - set(listed)
        if missing:
            rules["B: format>=3, a required pickle not listed"].append(r)
    if cs == "ok":
        if "analysis_context.pkl" in present and "analysis_context.pkl" not in listed:
            rules["C: context present but undeclared (would be ignored)"].append(r)
        if "analysis_context.pkl" in listed and "analysis_context.pkl" not in present:
            rules["C: context declared but missing (would be refused)"].append(r)

print(f"loads: {len(loads)}")
print("by format_version x checksums:", dict(Counter((r['format_version'], r['checksums']) for r in loads)))
print("outcome today:", dict(Counter(r["outcome"] for r in loads)))
for name in sorted(rules):
    hits = rules[name]
    print(f"\n{name}: {len(hits)}")
    print("  outcome today:", dict(Counter(r["outcome"] for r in hits)))
    for f, n in sorted(Counter(test_file(r) for r in hits).items()):
        print(f"    {n:4d}  {f}")
print(f"\nloads of a directory whose metadata.json is missing or not JSON: {len(unreadable)}")
print("  outcome today:", dict(Counter(r["outcome"] for r in unreadable)))
for f, n in sorted(Counter(test_file(r) for r in unreadable).items()):
    print(f"    {n:4d}  {f}")
print(f"unknown records: {len(bad)}")
sys.exit(1 if bad else 0)
