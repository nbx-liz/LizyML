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

from tests.test_docs._history_grammar import (
    grammar_violations,
    id_violations,
    parse_entries,
)

HISTORY = Path(__file__).resolve().parents[2] / "HISTORY.md"


def test_history_ids_are_unique_and_one_per_entry() -> None:
    text = HISTORY.read_text(encoding="utf-8")
    entries = parse_entries(text)
    # Vacuity guard: the file has over a hundred entries; a parser that found
    # none would report no violations.
    assert len(entries) >= 100, f"only {len(entries)} entries parsed"
    problems = grammar_violations(text) + id_violations(entries)
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


def test_a_tilde_line_does_not_close_a_backtick_fence() -> None:
    # H-0110 design review round 1: the fence closed on "~~~", the fenced
    # example became an entry and the real H-0003 vanished with no violation.
    text = "## H-0001: a\n\n```\n~~~\n## H-0002: fake\n```\n\n## H-0003: real\n"
    entries = parse_entries(text)
    assert [ids for _, ids in entries] == [{"H-0001"}, {"H-0003"}]


def test_an_indented_fence_hides_its_headings() -> None:
    text = "## H-0001: a\n\n1. item\n   ```\n## H-0002: fake\n   ```\n"
    assert [ids for _, ids in parse_entries(text)] == [{"H-0001"}]


def test_a_longer_fence_closes_only_on_a_run_at_least_as_long() -> None:
    text = "## H-0001: a\n\n````\n```\n## H-0002: fake\n````\n\n## H-0003: real\n"
    assert [ids for _, ids in parse_entries(text)] == [{"H-0001"}, {"H-0003"}]


def test_a_heading_id_with_a_suffix_is_not_an_id() -> None:
    problems = id_violations(parse_entries("## H-0042-extra\n"))
    assert problems == ["'## H-0042-extra' declares 0 ids: []"]


def test_an_unclosed_fence_is_reported() -> None:
    assert grammar_violations("## H-0001: a\n\n```\nnever closed\n") == [
        "fence opened at line 3 is never closed"
    ]


def test_a_heading_and_one_id_line_may_name_the_same_id() -> None:
    # The measured spelling of H-0054 .. H-0060: both, naming one id.
    text = "## H-0055: a\n\n- ID: `H-0055`\n"
    assert grammar_violations(text) == []
    assert id_violations(parse_entries(text)) == []


def test_a_repeated_id_line_is_reported() -> None:
    # H-0110 design review round 2: a set of ids collapsed the repeat silently.
    text = "## 2026-01-01: a\n\n- ID: `H-0001`\n- ID: `H-0001`\n"
    assert grammar_violations(text) == ["'## 2026-01-01: a' has 2 '- ID:' lines"]


def test_an_id_line_inside_a_fence_is_not_counted() -> None:
    text = "## H-0001: a\n\n- ID: `H-0001`\n\n```\n- ID: `H-0001`\n```\n"
    assert grammar_violations(text) == []


def test_a_malformed_id_line_next_to_a_heading_id_is_reported() -> None:
    # H-0110 design review round 3: an unbackticked ID line was ignored silently.
    text = "## H-0001: migrated\n\n- ID: H-0002\n"
    assert grammar_violations(text) == [
        "'## H-0001: migrated' has a malformed ID line: '- ID: H-0002'"
    ]


def test_a_malformed_id_line_after_a_valid_one_is_reported() -> None:
    text = "## 2026-01-01: a\n\n- ID: `H-0001`\n- id : H-0002\n"
    assert grammar_violations(text) == [
        "'## 2026-01-01: a' has 2 '- ID:' lines",
        "'## 2026-01-01: a' has a malformed ID line: '- id : H-0002'",
    ]
