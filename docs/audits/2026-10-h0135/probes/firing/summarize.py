"""Summarize the H-0135 load census: how often each decided rule changes a load.

Usage: python3 summarize.py <census.jsonl>

The rules are the ones decided for H-0135 (card D-20261020-002):

- R1 (#374): a present ``checksums`` that is not exactly
  ``{"algorithm": "sha256", "files": {str: 64 lowercase hex}}`` is refused,
  in every format.
- R2: an absent ``checksums`` is refused in format >= 3.
- R3: with ``checksums`` present, ``fit_result.pkl`` and ``refit_model.pkl``
  must be listed, in every format.
- R4 (#373): a listed ``analysis_context.pkl`` that is missing is refused.
- I (#373): a present but unlisted ``analysis_context.pkl`` is ignored. This
  is not a refusal.

The script prints each rule's loads, split by the outcome the load has today,
with the test files involved. It then prints the UNION of the refusals (R1 to
R4), which is the firing rate, and the loads that would newly fail because they
load OK today. Loads of a directory without a readable ``metadata.json`` are
counted separately; they are refused before any of these rules. The script
exits 1 on an unknown record kind.
"""

from __future__ import annotations

import json
import sys
from collections import Counter

REQUIRED = {"fit_result.pkl", "refit_model.pkl"}
CONTEXT = "analysis_context.pkl"

records = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
unknown = [r for r in records if r.get("kind") != "load"]
unreadable = [r for r in records if r.get("kind") == "load" and "unreadable" in r]
loads = [r for r in records if r.get("kind") == "load" and "unreadable" not in r]


def test_file(r: dict) -> str:
    return r["test"].split(" ")[0].split("::")[0]


def rules_hit(r: dict) -> list[str]:
    fv, cs = r["format_version"], r["checksums"]
    listed, present = set(r["listed"] or ()), set(r["present"])
    hit = []
    if cs in ("non_object", "bad_object"):
        hit.append("R1")
    if cs == "absent" and isinstance(fv, int) and fv >= 3:
        hit.append("R2")
    if cs == "ok" and not REQUIRED <= listed:
        hit.append("R3")
    if cs == "ok" and CONTEXT in listed and CONTEXT not in present:
        hit.append("R4")
    return hit


def show(title: str, hits: list[dict]) -> None:
    print(f"\n{title}: {len(hits)}")
    print("  outcome today:", dict(Counter(r["outcome"] for r in hits)))
    for f, n in sorted(Counter(test_file(r) for r in hits).items()):
        print(f"    {n:4d}  {f}")


print(f"artifact loads: {len(loads)} (+ {len(unreadable)} of a directory without readable metadata.json)")
print("format_version x checksums:", dict(sorted(Counter(f"{r['format_version']}/{r['checksums']}" for r in loads).items())))
print("outcome today:", dict(Counter(r["outcome"] for r in loads)))
for rule in ("R1", "R2", "R3", "R4"):
    show(rule, [r for r in loads if rule in rules_hit(r)])
ignored = [
    r for r in loads
    if r["checksums"] == "ok" and CONTEXT in r["present"] and CONTEXT not in set(r["listed"] or ())
]
show("I (ignored, not refused)", ignored)
union = [r for r in loads if rules_hit(r)]
show("UNION of refusals R1-R4", union)
newly = [r for r in union if r["outcome"] == "ok"]
show("of which load OK today (would newly fail)", newly)
symlinked = [r for r in loads if r.get("symlinks")]
show("loads with a symlink in the directory", symlinked)
print(f"\nFiring rate: {len(union)}/{len(loads)} of artifact loads "
      f"({len(newly)} of them load OK today)")
print(f"unknown records: {len(unknown)}")
sys.exit(1 if unknown else 0)
