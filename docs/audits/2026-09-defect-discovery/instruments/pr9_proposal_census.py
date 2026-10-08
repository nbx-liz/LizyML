#!/usr/bin/env python3
"""PR 9 census: every HISTORY.md proposal, its Status spelling, and whether BLUEPRINT cites it.

Reuses the closed entry grammar of ``tests/test_docs/test_history_ids.py``
(``parse_entries``), so this instrument does not introduce a third HISTORY
parser. An entry without exactly one id is a failure, never a skip.

"Cited" is a full-token match of the id in ``BLUEPRINT.md`` (``H-0083`` is not
satisfied by ``H-00830``). Citation is reported for comparison with #271's
original 92 / 35 / 57 figures only: #271 established that id absence is not
content absence, so nothing here decides an obligation.

Usage (from the repository root)::

    python3 docs/audits/2026-09-defect-discovery/instruments/pr9_proposal_census.py \
        [--ref origin/develop]

With ``--ref`` the two documents are read from that git ref, not the checkout.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from tests.test_docs._history_grammar import (  # noqa: E402
    entry_texts,
    id_violations,
    parse_entries,
)

_STATUS = re.compile(r"^\s*-\s*(?:\*\*)?(?:Status|ステータス)(?:\*\*)?\s*[:：]\s*(.*)$")
_STATUS_CLASS = (
    ("implemented", re.compile(r"implemented|done", re.IGNORECASE)),
    ("accepted", re.compile(r"accepted", re.IGNORECASE)),
    ("proposed", re.compile(r"proposed", re.IGNORECASE)),
)

# #271's population as filed (2026-09-04): the ids absent from BLUEPRINT then.
ORIGINAL_REGISTER = 92


def read(name: str, ref: str | None) -> str:
    if ref is None:
        return (ROOT / name).read_text(encoding="utf-8")
    return subprocess.run(
        ["git", "show", f"{ref}:{name}"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout


def entry_bodies(text: str) -> dict[str, list[str]]:
    """Lines of each entry keyed by its id, from the shared grammar."""
    return {i: body.splitlines() for i, body in entry_texts(text).items()}


def status_of(lines: list[str]) -> tuple[str, str]:
    """``(raw spelling, class)`` of the first Status line; class is ``none`` if absent."""
    for line in lines:
        match = _STATUS.match(line)
        if match:
            raw = match.group(1).strip()
            for name, pattern in _STATUS_CLASS:
                if pattern.search(raw):
                    return raw, name
            return raw, "unclassified"
    return "", "none"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", default=None)
    args = parser.parse_args()

    history = read("HISTORY.md", args.ref)
    blueprint = read("BLUEPRINT.md", args.ref)

    entries = parse_entries(history)
    problems = id_violations(entries)
    ids = sorted({i for _, declared in entries for i in declared})
    bodies = entry_bodies(history)
    missing_bodies = sorted(set(ids) - set(bodies))

    cited = {i for i in ids if re.search(rf"(?<![\w-]){re.escape(i)}(?![\w-])", blueprint)}
    absent = [i for i in ids if i not in cited]
    stray = sorted(set(re.findall(r"(?<![\w-])H-\d{4}(?![\w-])", blueprint)) - set(ids))

    statuses = {i: status_of(bodies.get(i, [])) for i in ids}
    spelling = Counter(raw for raw, _ in statuses.values())
    classes = Counter(cls for _, cls in statuses.values())

    print(f"ref                        : {args.ref or 'working tree'}")
    print(f"HISTORY entries (## )      : {len(entries)}")
    print(f"entry/id violations        : {len(problems)}")
    for problem in problems:
        print(f"  ! {problem}")
    print(f"proposal ids               : {len(ids)}  ({ids[0]} .. {ids[-1]})")
    print(f"ids without a body         : {len(missing_bodies)} {missing_bodies}")
    print(f"cited in BLUEPRINT         : {len(cited)}")
    print(f"absent from BLUEPRINT      : {len(absent)}")
    print(f"BLUEPRINT ids not in HISTORY: {len(stray)} {stray}")
    print(f"ids beyond #271's register : {len(ids) - ORIGINAL_REGISTER} "
          f"(register was {ORIGINAL_REGISTER})")
    print()
    print("status class   count")
    for name, count in sorted(classes.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<13}{count}")
    print()
    print(f"status spellings: {len(spelling)}")
    for raw, count in sorted(spelling.items(), key=lambda kv: -kv[1]):
        print(f"  {count:>3}  {raw[:70]!r}")
    print()
    print("per proposal: id  cited  status-class")
    for i in ids:
        print(f"  {i}  {'Y' if i in cited else '-'}  {statuses[i][1]}")
    return 1 if problems or missing_bodies else 0


if __name__ == "__main__":
    raise SystemExit(main())
