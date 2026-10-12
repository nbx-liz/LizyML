"""The documents state the search-space dimension format (H-0123 criterion 5).

BLUEPRINT §11.1 and docs/config-reference.md carry the accepted format of a
``tuning.optuna.space`` / ``tune(space=...)`` dimension, and the CHANGELOG lists
what is newly refused and accepted. The keys, types, categories and the number
literal are read from the code that enforces them (``lizyml.tuning._space_format``),
not copied here (DC3), so a key added to the code without the documents, or a
changed literal, fails here.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from lizyml.tuning._space_format import DIM_KEYS, NUMBER_LITERAL, REQUIRED_KEYS
from lizyml.tuning.lineage_codec import DIM_CATEGORIES

_ROOT = Path(__file__).resolve().parents[2]


def _read(rel: str) -> str:
    return (_ROOT / rel).read_text(encoding="utf-8")


def _section(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    return text[begin : text.index(end, begin + len(start))]


def _blueprint_space() -> str:
    return _section(_read("BLUEPRINT.md"), "## 11.1 ", "## 11.3 ")


def _config_reference_tuning() -> str:
    return _section(_read("docs/config-reference.md"), "## `tuning`", "\n## ")


def _changelog_unreleased() -> str:
    return _section(_read("CHANGELOG.md"), "## [Unreleased]", "## [0.18.0]")


_KEYS = sorted(set().union(*DIM_KEYS.values()))
_DOCS: list[tuple[str, Callable[[], str]]] = [
    ("blueprint", _blueprint_space),
    ("config_reference", _config_reference_tuning),
]


def test_the_format_was_read_from_the_code() -> None:
    assert set(DIM_KEYS) == {"float", "int", "categorical"}
    assert "choices" in _KEYS
    assert len(DIM_CATEGORIES) == 3


@pytest.mark.parametrize(("doc", "section"), _DOCS, ids=[d for d, _ in _DOCS])
def test_doc_states_every_key_type_and_category(
    doc: str, section: Callable[[], str]
) -> None:
    text = section()
    for token in (*_KEYS, *DIM_KEYS, *sorted(DIM_CATEGORIES)):
        assert f"`{token}`" in text or f'`"{token}"`' in text, (doc, token)


@pytest.mark.parametrize(("doc", "section"), _DOCS, ids=[d for d, _ in _DOCS])
def test_doc_states_the_number_literal_and_the_refused_step(
    doc: str, section: Callable[[], str]
) -> None:
    text = section()
    assert NUMBER_LITERAL.pattern in text, doc
    assert "`step`" in text, doc
    assert "H-0123" in text, doc


def _dimension_table() -> list[list[str]]:
    """The rows of docs/config-reference.md's dimension key table, as cells."""
    lines = _config_reference_tuning().splitlines()
    start = lines.index("| Key | Applies to | Required | Accepted values |")
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def _ticked(cell: str) -> list[str]:
    return [part.strip().strip("`") for part in cell.split(",")]


def test_config_reference_key_table_matches_the_code() -> None:
    # Both directions (DC3): a key the table lists but the code lacks, or a
    # required flag that differs from REQUIRED_KEYS, fails here too.
    allowed: dict[str, set[str]] = {t: set() for t in DIM_KEYS}
    required: dict[str, set[str]] = {t: set() for t in DIM_KEYS}
    for key_cell, applies, req, _ in _dimension_table():
        types = list(DIM_KEYS) if applies == "all" else _ticked(applies)
        for key in _ticked(key_cell):
            for dim_type in types:
                allowed[dim_type].add(key)
                if req == "Yes":
                    required[dim_type].add(key)
    assert allowed == {t: set(keys) for t, keys in DIM_KEYS.items()}
    assert required == {t: {"type", *keys} for t, keys in REQUIRED_KEYS.items()}


def test_changelog_lists_the_refused_and_accepted_forms() -> None:
    text = _changelog_unreleased()
    assert "H-0123" in text
    for form in ('`log: "false"`', "`step`", "`1.7`", '`"abc"`', '`"16.0"`', '`"1e2"`'):
        assert form in text, form
