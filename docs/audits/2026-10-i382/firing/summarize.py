"""Summarise an H-0141 census log: the calls the alias rule acts on, by test file.

    python docs/audits/2026-10-i382/firing/summarize.py <census.jsonl>

Exits 1 when a call outside the H-0141 test files is aliased or refused
(acceptance 9), so the census is a check, not a reading.
"""

from __future__ import annotations

import json
import sys
from collections import Counter

H0141_FILES = (
    "tests/test_codegen/test_h0141_column_names.py",
    "tests/test_codegen/test_h0141_column_name_projects.py",
)

records = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
fired = [
    r
    for r in records
    if r["non_str_features"] or r.get("declared_remapped") or r.get("refused")
]
print(f"project_column_names calls: {len(records)}")
for site in ("export_code", "generate_code"):
    at = [r for r in records if r["site"] == site]
    hit = [r for r in fired if r["site"] == site]
    print(f"  {site}: {len(hit)}/{len(at)} aliased or refused")
by_file = Counter(r["test"].split("::", 1)[0] for r in fired)
for name, count in sorted(by_file.items()):
    print(f"    {count:4d}  {name}")
outside = [r for r in fired if not r["test"].startswith(H0141_FILES)]
print(f"  outside the H-0141 tests: {len(outside)}")
for r in outside[:20]:
    print("   ", r["test"][:110], r)
sys.exit(1 if outside else 0)
