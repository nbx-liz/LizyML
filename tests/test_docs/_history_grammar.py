"""The closed HISTORY.md entry grammar, shared by every test that reads the register.

An entry is a level-2 heading (``## ``) outside fenced code. It declares its id
in exactly one of two measured spellings: the heading itself (``## H-0093: ...``)
or a metadata line (``- ID: `H-0102` ...``). ``id_violations`` reports an entry
with no id or two ids, and an id declared twice, by name -- never skipped -- so
a third spelling cannot widen or shrink the population silently (H-0101).

One grammar, one module: ``test_history_ids.py`` checks the register's ids and
``test_proposal_blueprint_coverage.py`` checks each id's BLUEPRINT disposition
(H-0110). A second parser would be free to disagree with the first.
"""

from __future__ import annotations

import re
from collections import defaultdict

_FENCE = re.compile(r"^(```|~~~)")
_HEADING_ID = re.compile(r"^## (H-\d{4})\b")
_META_ID = re.compile(r"^- ID: `(H-\d{4})`")


def _split(text: str) -> list[tuple[str, set[str], list[str]]]:
    """``(heading, declared ids, lines)`` for every level-2 entry."""
    entries: list[tuple[str, set[str], list[str]]] = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            if entries:
                entries[-1][2].append(line)
            continue
        if in_fence:
            if entries:
                entries[-1][2].append(line)
            continue
        if line.startswith("## "):
            entries.append((line, set(), [line]))
            match = _HEADING_ID.match(line)
            if match:
                entries[-1][1].add(match.group(1))
            continue
        if not entries:
            continue
        entries[-1][2].append(line)
        match = _META_ID.match(line)
        if match:
            entries[-1][1].add(match.group(1))
    return entries


def parse_entries(text: str) -> list[tuple[str, set[str]]]:
    """Return ``(heading, declared ids)`` for every level-2 entry."""
    return [(heading, ids) for heading, ids, _ in _split(text)]


def entry_texts(text: str) -> dict[str, str]:
    """The full text of each entry that declares exactly one id, keyed by that id.

    Entries violating the grammar are left to ``id_violations``; callers check
    it first, so nothing here is skipped unreported.
    """
    return {
        next(iter(ids)): "\n".join(lines)
        for _, ids, lines in _split(text)
        if len(ids) == 1
    }


def id_violations(entries: list[tuple[str, set[str]]]) -> list[str]:
    """Entries without exactly one id, and ids declared by more than one entry."""
    problems: list[str] = []
    owners: dict[str, list[str]] = defaultdict(list)
    for heading, ids in entries:
        if len(ids) != 1:
            problems.append(f"{heading!r} declares {len(ids)} ids: {sorted(ids)}")
        for proposal_id in ids:
            owners[proposal_id].append(heading)
    for proposal_id, headings in sorted(owners.items()):
        if len(headings) > 1:
            problems.append(
                f"{proposal_id} is declared by {len(headings)} entries: {headings}"
            )
    return problems
