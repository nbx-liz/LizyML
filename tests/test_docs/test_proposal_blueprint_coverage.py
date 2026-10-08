"""Every HISTORY.md proposal has a BLUEPRINT.md disposition, checked by content.

#271 (H-0110): decisions were recorded in HISTORY.md, implemented, and never folded into
BLUEPRINT.md -- the canonical instance being H-0083's artifact checksum, which
BLUEPRINT did not mention at all. Folding them in by hand closes the instance;
this check is the durable repair.

It compares **content, not ids**. #271 showed id presence is the wrong proxy in
both directions: 33 proposals are fully specified without their id appearing,
and a bare ``(H-0083)`` added anywhere would satisfy an id check while the
checksum stayed unwritten. So each proposal has a row in
``docs/proposal_dispositions.toml`` with exactly one disposition:

``specified``
    ``anchors``: identifiers that state the decision. Each must occur, by full
    token, both in BLUEPRINT.md and in the proposal's own HISTORY entry -- the
    second ties the anchor to what was decided, so an unrelated word that merely
    happens to be in BLUEPRINT cannot stand in for it. A proposal id is never an
    anchor.
``no_obligation``
    ``reason``: the proposal decides nothing on the ``CLAUDE.md`` section 3
    surface that is still in force.
``superseded``
    ``superseded_by`` (a proposal id) and ``reason``.
``pending``
    ``reason``: not decided yet.

The population is the register, not a list: rows must equal the ids HISTORY.md
declares (``_history_grammar``), in both directions, and the per-proposal test is
parametrized over those ids, so its count is regenerated from HISTORY.md rather
than stored. A row with an unknown key or an unknown disposition is a failure,
never a skip.

``[names]`` covers the public names #271 found undocumented anywhere: each is
``documented`` (full token in the named document) or ``internal`` with a reason.

"Full token" means not adjacent to ``[A-Za-z0-9_]``: ``checksum`` is not found in
``checksum_algorithm``, nor ``TASK_TYPES`` in ``TASK_TYPES_LEGACY`` (DC2).
"""

from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

from tests.test_docs._history_grammar import (
    entry_texts,
    grammar_violations,
    id_violations,
    parse_entries,
)

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10; pytest itself depends on tomli there
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
HISTORY = ROOT / "HISTORY.md"
BLUEPRINT = ROOT / "BLUEPRINT.md"
DISPOSITIONS = ROOT / "docs" / "proposal_dispositions.toml"

_PROPOSAL_ID = re.compile(r"H-\d{4}")
_ROW_KEYS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    # disposition: (required keys, optional keys); "disposition" itself is implied
    "specified": (frozenset({"anchors"}), frozenset({"note"})),
    "no_obligation": (frozenset({"reason"}), frozenset()),
    "superseded": (frozenset({"superseded_by", "reason"}), frozenset()),
    "pending": (frozenset({"reason"}), frozenset()),
}
_NAME_KEYS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "documented": (frozenset({"where"}), frozenset({"note"})),
    "internal": (frozenset({"reason"}), frozenset()),
}
_NAME_DOCUMENTS = frozenset({"BLUEPRINT.md", "docs/api.md"})
# #271 found eight implemented public names documented nowhere; two
# (ErrorCode.EVALUATION_FAILED, ErrorCode.CALIBRATION_NOT_FITTED) were documented
# in docs/api.md before PR 9. The [names] table must hold exactly the other six,
# so dropping a row cannot turn the check green.
PUBLIC_NAMES_271 = frozenset(
    {
        "CHECKSUM_ALGORITHM",
        "SUPPORTED_CONFIG_VERSIONS",
        "TASK_TYPES",
        "DEFAULT_TEMPLATE",
        "DEFAULT_HEIGHT",
        "DEFAULT_WIDTH",
    }
)


def has_token(token: str, text: str) -> bool:
    """``token`` occurs in ``text`` with no identifier character on either side."""
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])"
    return re.search(pattern, text) is not None


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _shape_problems(
    where: str, row: Any, kinds: dict[str, tuple[frozenset[str], frozenset[str]]]
) -> list[str]:
    if not isinstance(row, dict):
        return [f"{where}: row is {type(row).__name__}, not a table"]
    kind = row.get("disposition")
    if kind not in kinds:
        return [f"{where}: disposition {kind!r} is not one of {sorted(kinds)}"]
    required, optional = kinds[kind]
    keys = set(row) - {"disposition"}
    problems = [f"{where}: missing key {k!r}" for k in sorted(required - keys)]
    problems += [
        f"{where}: unknown key {k!r}" for k in sorted(keys - required - optional)
    ]
    for key in sorted(keys & ({"reason", "note", "superseded_by", "where"})):
        if not _nonempty_str(row[key]):
            problems.append(f"{where}: {key!r} must be a non-empty string")
    return problems


def row_problems(
    proposal_id: str, row: Any, history_ids: set[str], entry: str, blueprint: str
) -> list[str]:
    """Everything wrong with one proposal's disposition row; empty when it holds."""
    problems = _shape_problems(proposal_id, row, _ROW_KEYS)
    if problems:
        return problems
    kind = row["disposition"]
    if kind == "superseded":
        target = row["superseded_by"]
        if target == proposal_id or target not in history_ids:
            problems.append(
                f"{proposal_id}: superseded_by {target!r} is not another proposal"
            )
    if kind == "specified":
        anchors = row["anchors"]
        if not isinstance(anchors, list) or not anchors:
            return [f"{proposal_id}: anchors must be a non-empty list"]
        if len(set(map(str, anchors))) != len(anchors):
            problems.append(f"{proposal_id}: anchors repeat")
        for anchor in anchors:
            if not _nonempty_str(anchor):
                problems.append(
                    f"{proposal_id}: anchor {anchor!r} is not a non-empty string"
                )
                continue
            if _PROPOSAL_ID.search(anchor):
                problems.append(f"{proposal_id}: anchor {anchor!r} names a proposal id")
                continue
            if not has_token(anchor, blueprint):
                problems.append(
                    f"{proposal_id}: anchor {anchor!r} is not in BLUEPRINT.md"
                )
            if not has_token(anchor, entry):
                problems.append(
                    f"{proposal_id}: anchor {anchor!r} is not in the proposal's "
                    "HISTORY entry"
                )
    return problems


def name_problems(name: str, row: Any, documents: dict[str, str]) -> list[str]:
    """Everything wrong with one public name's disposition row."""
    problems = _shape_problems(name, row, _NAME_KEYS)
    if problems or row["disposition"] != "documented":
        return problems
    where = row["where"]
    if where not in documents:
        return [f"{name}: where {where!r} is not one of {sorted(documents)}"]
    if not has_token(name, documents[where]):
        return [f"{name}: not found by full token in {where}"]
    return []


def name_coverage_problems(names: dict[str, Any]) -> list[str]:
    """The [names] rows must equal #271's six names in both directions."""
    problems = [
        f"{n}: #271 name has no row" for n in sorted(PUBLIC_NAMES_271 - set(names))
    ]
    problems += [
        f"{n}: row is not one of #271's names"
        for n in sorted(set(names) - PUBLIC_NAMES_271)
    ]
    return problems


def coverage_problems(history_ids: set[str], rows: dict[str, Any]) -> list[str]:
    """Rows must equal the register in both directions."""
    missing = sorted(history_ids - set(rows))
    extra = sorted(set(rows) - history_ids)
    problems = [f"{i}: proposal has no disposition row" for i in missing]
    problems += [f"{i}: row names no HISTORY.md proposal" for i in extra]
    return problems


@lru_cache(maxsize=1)
def _register() -> tuple[frozenset[str], dict[str, str]]:
    text = HISTORY.read_text(encoding="utf-8")
    entries = parse_entries(text)
    problems = grammar_violations(text) + id_violations(entries)
    assert not problems, "\n".join(problems)
    return frozenset(i for _, ids in entries for i in ids), entry_texts(text)


@lru_cache(maxsize=1)
def _dispositions() -> dict[str, Any]:
    with DISPOSITIONS.open("rb") as handle:
        data = tomllib.load(handle)
    assert set(data) == {"proposals", "names"}, (
        f"top-level tables must be exactly proposals and names, got {sorted(data)}"
    )
    return data


def _history_ids_at_collection() -> list[str]:
    entries = parse_entries(HISTORY.read_text(encoding="utf-8"))
    return sorted({i for _, ids in entries for i in ids})


def test_every_proposal_has_exactly_one_row() -> None:
    history_ids, _ = _register()
    # Vacuity guard: the register has over a hundred proposals.
    assert len(history_ids) >= 100, f"only {len(history_ids)} proposals parsed"
    problems = coverage_problems(set(history_ids), _dispositions()["proposals"])
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("proposal_id", _history_ids_at_collection())
def test_proposal_disposition_holds(proposal_id: str) -> None:
    history_ids, entries = _register()
    rows = _dispositions()["proposals"]
    assert proposal_id in rows, f"{proposal_id}: proposal has no disposition row"
    problems = row_problems(
        proposal_id,
        rows[proposal_id],
        set(history_ids),
        entries[proposal_id],
        BLUEPRINT.read_text(encoding="utf-8"),
    )
    assert not problems, "\n".join(problems)


def test_public_name_dispositions_hold() -> None:
    documents = {d: (ROOT / d).read_text(encoding="utf-8") for d in _NAME_DOCUMENTS}
    names = _dispositions()["names"]
    problems = name_coverage_problems(names)
    problems += [
        p for name, row in names.items() for p in name_problems(name, row, documents)
    ]
    assert not problems, "\n".join(problems)


# --- the matcher and the row grammar refuse what they must ---------------------


@pytest.mark.parametrize(
    ("token", "text"),
    [
        ("checksum", "the checksum_algorithm field"),
        ("TASK_TYPES", "TASK_TYPES_LEGACY is kept"),
        ("firm", "firmbogus"),
        ("price_confidence", "not_price_confidence"),
        ("sha256", "sha2567"),
    ],
)
def test_a_superstring_is_not_a_token_match(token: str, text: str) -> None:
    assert not has_token(token, text)


@pytest.mark.parametrize(
    ("token", "text"),
    [
        ("CHECKSUM_ALGORITHM", '`CHECKSUM_ALGORITHM = "sha256"`'),
        ("metadata.json", "written into metadata.json."),
        ("model.params_table()", "call model.params_table() after fit"),
        ("検証", "load 時に検証する"),
    ],
)
def test_a_bounded_occurrence_is_a_token_match(token: str, text: str) -> None:
    assert has_token(token, text)


_IDS = {"H-0001", "H-0002"}
_ENTRY = "## H-0002: x\n- decided CHECKSUM_ALGORITHM and metadata.json\n"
_BLUEPRINT = "the CHECKSUM_ALGORITHM constant; metadata.json; an unrelated word"


def test_a_well_formed_specified_row_holds() -> None:
    row = {
        "disposition": "specified",
        "anchors": ["CHECKSUM_ALGORITHM", "metadata.json"],
    }
    assert row_problems("H-0002", row, _IDS, _ENTRY, _BLUEPRINT) == []


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"disposition": "folded"}, "is not one of"),
        ({"disposition": "specified"}, "missing key 'anchors'"),
        ({"disposition": "specified", "anchors": []}, "non-empty list"),
        (
            {"disposition": "specified", "anchors": ["unrelated"]},
            "not in the proposal's HISTORY",
        ),
        ({"disposition": "specified", "anchors": ["decided"]}, "not in BLUEPRINT.md"),
        ({"disposition": "specified", "anchors": ["H-0002"]}, "names a proposal id"),
        (
            {"disposition": "specified", "anchors": ["metadata.json", "metadata.json"]},
            "repeat",
        ),
        ({"disposition": "specified", "anchors": [""]}, "not a non-empty string"),
        (
            {"disposition": "specified", "anchors": ["metadata.json"], "reason": "x"},
            "unknown key",
        ),
        ({"disposition": "no_obligation"}, "missing key 'reason'"),
        ({"disposition": "no_obligation", "reason": "  "}, "non-empty string"),
        (
            {"disposition": "superseded", "reason": "x", "superseded_by": "H-0009"},
            "not another",
        ),
        (
            {"disposition": "superseded", "reason": "x", "superseded_by": "H-0002"},
            "not another",
        ),
        ({"disposition": "pending"}, "missing key 'reason'"),
        ("specified", "not a table"),
    ],
)
def test_a_malformed_row_is_refused(row: Any, expected: str) -> None:
    problems = row_problems("H-0002", row, _IDS, _ENTRY, _BLUEPRINT)
    assert any(expected in p for p in problems), problems


def test_rows_must_equal_the_register_in_both_directions() -> None:
    problems = coverage_problems({"H-0001", "H-0002"}, {"H-0002": {}, "H-0003": {}})
    assert problems == [
        "H-0001: proposal has no disposition row",
        "H-0003: row names no HISTORY.md proposal",
    ]


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (
            {"disposition": "documented", "where": "BLUEPRINT.md"},
            "not found by full token",
        ),
        ({"disposition": "documented", "where": "README.md"}, "is not one of"),
        ({"disposition": "internal"}, "missing key 'reason'"),
        ({"disposition": "public"}, "is not one of"),
    ],
)
def test_a_malformed_name_row_is_refused(row: Any, expected: str) -> None:
    documents = {"BLUEPRINT.md": "TASK_TYPES_LEGACY only", "docs/api.md": ""}
    problems = name_problems("TASK_TYPES", row, documents)
    assert any(expected in p for p in problems), problems


def test_the_names_table_must_hold_exactly_the_six_names() -> None:
    # H-0110 design review round 1: deleting the CHECKSUM_ALGORITHM row left the
    # other five rows passing.
    six = {
        name: {"disposition": "internal", "reason": "x"} for name in PUBLIC_NAMES_271
    }
    del six["CHECKSUM_ALGORITHM"]
    six["EXTRA_NAME"] = {"disposition": "internal", "reason": "x"}
    assert name_coverage_problems(six) == [
        "CHECKSUM_ALGORITHM: #271 name has no row",
        "EXTRA_NAME: row is not one of #271's names",
    ]
