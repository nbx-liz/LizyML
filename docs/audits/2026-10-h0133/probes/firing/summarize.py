"""Summarize the H-0133 firing log written by ``sitecustomize.py``.

Counts encoder fit records (one per fit and column) by process role and by
the category value types seen, and how many held a value LightGBM cannot
write -- the population D2-B would select and D4's allow-list would judge.
Records from the tests that build unwritable categories on purpose are
counted apart. Any ``error`` record makes the exit status 1.

Usage: python3 summarize.py <log.jsonl> [<log.jsonl> ...]
"""

from __future__ import annotations

import json
import sys
from collections import Counter

#: Test files that construct JSON-unsafe categories on purpose (#338).
DELIBERATE = (
    "tests/test_persistence/test_unwritable_categories.py::",
    "tests/test_codegen/test_export_contract.py::",
    "tests/test_persistence/test_h0120_records.py::",
)


def main(paths: list[str]) -> int:
    recs = [json.loads(line) for p in paths for line in open(p, encoding="utf-8")]
    c: Counter[str] = Counter()
    types: Counter[str] = Counter()
    unwritable_types: Counter[str] = Counter()
    elsewhere: Counter[str] = Counter()
    errors = 0
    for r in recs:
        if r.get("kind") == "error":
            errors += 1
            print("error:", r["error"], r.get("test"))
            continue
        c[f"fit columns[{r['role']}]"] += 1
        for t in r["types"]:
            types[t] += 1
        deliberate = r["test"].startswith(DELIBERATE)
        if "outside_v" not in r:
            errors += 1
            print("error: record without the V verdict", r.get("test"))
            continue
        impl = r.get("impl_outside_v")
        if impl is not None:
            c["implementation agrees with the reference" if bool(impl) == bool(r["outside_v"]) else "implementation DISAGREES"] += 1
            if bool(impl) != bool(r["outside_v"]):
                errors += 1
                print("disagreement:", r["outside_v"], impl, r.get("test"))
        if "stand_in" in r:
            if r["stand_in"] != bool(r["unwritable"]):
                errors += 1
                print("stand-in decision differs from the LightGBM predicate:", r.get("test"))
            c["stand-in column deliberate" if r["stand_in"] and deliberate else ("stand-in column elsewhere" if r["stand_in"] else "raw column")] += 1
        if r["outside_v"]:
            c["outside V deliberate" if deliberate else "outside V elsewhere"] += 1
            for reason in r["outside_v"]:
                c[f"outside V reason[{reason}]"] += 1
        else:
            c["inside V"] += 1
        if r["unwritable"]:
            c["unwritable deliberate" if deliberate else "unwritable elsewhere"] += 1
            for t in r["unwritable"]:
                unwritable_types[t] += 1
            if not deliberate:
                elsewhere[r["test"].split(" ")[0]] += 1
    for k in sorted(c):
        print(f"{k}: {c[k]}")
    print("category value types (column fits containing each):")
    for t, n in types.most_common():
        print(f"  {t}: {n}")
    print("unwritable types:", dict(unwritable_types.most_common()))
    print("tests outside the deliberate files with an unwritable column:")
    for t, n in elsewhere.most_common():
        print(f"  {t}: {n}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
