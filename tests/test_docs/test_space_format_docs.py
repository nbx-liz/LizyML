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

from lizyml.tuning._space_format import DIM_KEYS, NUMBER_LITERAL
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


def test_changelog_lists_the_refused_and_accepted_forms() -> None:
    text = _changelog_unreleased()
    assert "H-0123" in text
    for form in ('`log: "false"`', "`step`", "`1.7`", '`"abc"`', '`"16.0"`', '`"1e2"`'):
        assert form in text, form
