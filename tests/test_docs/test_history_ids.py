"""Every HISTORY.md entry declares exactly one proposal id, and no id twice (H-0101).

Two branches written in parallel each took the next free number and both merged:
the 2026-09-10 space-merge entry and PR 3c's calibration entry were both
``H-0100`` until 2026-10-01. Nothing compared them, because an id is allocated on
a branch and the collision only exists after both merge.

The grammar is closed rather than matched loosely. An entry is a level-2
heading (``## ``) outside fenced code. It declares its id in exactly one of two
measured spellings: the heading itself (``## H-0093: ...``) or a metadata line
(``- ID: `H-0102` ...``). An entry with no id, or with two different ids, is a
failure reported by name -- never skipped -- so a third spelling cannot widen
the population silently. The grammar lives in ``_history_grammar.py``, shared
with the BLUEPRINT coverage check (H-0110).
"""

from __future__ import annotations

from pathlib import Path

from tests.test_docs._history_grammar import id_violations, parse_entries

HISTORY = Path(__file__).resolve().parents[2] / "HISTORY.md"


def test_history_ids_are_unique_and_one_per_entry() -> None:
    entries = parse_entries(HISTORY.read_text(encoding="utf-8"))
    # Vacuity guard: the file has over a hundred entries; a parser that found
    # none would report no violations.
    assert len(entries) >= 100, f"only {len(entries)} entries parsed"
    problems = id_violations(entries)
    assert not problems, "\n".join(problems)


def test_a_duplicated_id_is_reported() -> None:
    text = "## H-0001: a\n\n## 2026-01-01: b\n\n- ID: `H-0001`\n"
    problems = id_violations(parse_entries(text))
    assert len(problems) == 1 and "H-0001 is declared by 2 entries" in problems[0]


def test_an_entry_without_an_id_is_reported() -> None:
    text = "## H-0001: a\n\n## 2026-01-01: no id here\n\n- Id: H-0002\n"
    problems = id_violations(parse_entries(text))
    assert problems == ["'## 2026-01-01: no id here' declares 0 ids: []"]


def test_an_entry_with_two_ids_is_reported() -> None:
    text = "## H-0001: a\n\n- ID: `H-0002`\n"
    problems = id_violations(parse_entries(text))
    assert problems == ["'## H-0001: a' declares 2 ids: ['H-0001', 'H-0002']"]


def test_headings_inside_code_fences_are_not_entries() -> None:
    text = "## H-0001: a\n\n```\n## H-0001: quoted\n```\n"
    assert id_violations(parse_entries(text)) == []
