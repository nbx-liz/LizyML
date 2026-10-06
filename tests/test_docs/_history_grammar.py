"""The closed HISTORY.md entry grammar, shared by every test that reads the register.

An entry is a level-2 heading (``## ``) outside fenced code. It declares its id
in exactly one of two measured spellings: the heading itself (``## H-0093: ...``,
the id followed by ``:`` or the end of the line) or a metadata line
(``- ID: `H-0102` ...``). ``id_violations`` reports an entry with no id or two
ids, and an id declared twice, by name -- never skipped -- so a third spelling
cannot widen or shrink the population silently (H-0101).

Fenced code is tracked the way CommonMark closes it: a fence opens with three or
more backticks or tildes at any indentation (HISTORY.md indents fences 0, 3 and 4
spaces inside list items), and closes only on a line of the **same** character,
at least as long, with nothing after it. A ``~~~`` inside a backtick fence does
not close it. A fence still open at the end of the file is reported by
``grammar_violations``, because it would silently swallow every entry after it
(design review round 1 of H-0110 executed both failures).

One grammar, one module: ``test_history_ids.py`` checks the register's ids and
``test_proposal_blueprint_coverage.py`` checks each id's BLUEPRINT disposition
(H-0110). A second parser would be free to disagree with the first.
"""

from __future__ import annotations

import re
from collections import defaultdict

_FENCE_OPEN = re.compile(r"^\s*(`{3,}|~{3,})")
_HEADING_ID = re.compile(r"^## (H-\d{4})(?::|\s*$)")
_META_ID = re.compile(r"^- ID: `(H-\d{4})`")


def _fence_close(line: str, opener: str) -> bool:
    """``line`` closes a fence opened by ``opener`` (same character, no shorter)."""
    stripped = line.strip()
    return len(stripped) >= len(opener) and set(stripped) == {opener[0]}


def _scan(
    text: str,
) -> tuple[list[tuple[str, set[str], list[str]]], int | None, list[int]]:
    """``(entries, line of a fence left open or None, '- ID:' lines per entry)``.

    Each entry is ``(heading, declared ids, lines)``; ``lines`` includes fenced
    lines, the per-entry ``- ID:`` count does not.
    """
    entries: list[tuple[str, set[str], list[str]]] = []
    metas: list[int] = []
    opener: str | None = None
    opened_at: int | None = None
    for number, line in enumerate(text.splitlines(), start=1):
        if opener is not None:
            if _fence_close(line, opener):
                opener, opened_at = None, None
            if entries:
                entries[-1][2].append(line)
            continue
        fence = _FENCE_OPEN.match(line)
        if fence:
            opener, opened_at = fence.group(1), number
            if entries:
                entries[-1][2].append(line)
            continue
        if line.startswith("## "):
            entries.append((line, set(), [line]))
            metas.append(0)
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
            metas[-1] += 1
    return entries, opened_at, metas


def parse_entries(text: str) -> list[tuple[str, set[str]]]:
    """Return ``(heading, declared ids)`` for every level-2 entry."""
    return [(heading, ids) for heading, ids, _ in _scan(text)[0]]


def entry_texts(text: str) -> dict[str, str]:
    """The full text of each entry that declares exactly one id, keyed by that id.

    Entries violating the grammar are left to ``id_violations``; callers check
    it first, so nothing here is skipped unreported.
    """
    return {
        next(iter(ids)): "\n".join(lines)
        for _, ids, lines in _scan(text)[0]
        if len(ids) == 1
    }


def grammar_violations(text: str) -> list[str]:
    """A fence left open at the end of the file, and a repeated ``- ID:`` line.

    The heading and one metadata line may name the same id: that is a measured
    spelling (H-0054 .. H-0060 carry both). Two metadata lines in one entry are
    not, even when they name the same id -- a set of ids would collapse them
    silently (H-0110 design review round 2).
    """
    entries, opened_at, metas = _scan(text)
    problems = []
    if opened_at is not None:
        problems.append(f"fence opened at line {opened_at} is never closed")
    for (heading, _, _), count in zip(entries, metas, strict=True):
        if count > 1:
            problems.append(f"{heading!r} has {count} '- ID:' lines")
    return problems


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
