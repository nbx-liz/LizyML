"""The documents state the search-space dimension format (H-0123 criterion 5).

BLUEPRINT §11.1 and docs/config-reference.md carry the accepted format of a
``tuning.optuna.space`` / ``tune(space=...)`` dimension, including the ``step``
grid of H-0124 (criterion 11), and the CHANGELOG lists what is newly refused
and accepted. The keys, types, categories and the number literal are read from
the code that enforces them (``lizyml.tuning._space_format``), and the step's
refusal reasons by triggering them,
not copied here (DC3), so a key added to the code without the documents, or a
changed literal, fails here.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from lizyml.core.exceptions import LizyMLError
from lizyml.tuning._space_format import DIM_KEYS, NUMBER_LITERAL, REQUIRED_KEYS
from lizyml.tuning.lineage_codec import DIM_CATEGORIES
from lizyml.tuning.search_space import parse_space

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
    return _section(_read("CHANGELOG.md"), "## [0.19.0]", "## [0.18.0]")


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


def _step_reasons() -> tuple[str, ...]:
    """The refusal reasons of a ``step`` and of its grid, read from the code."""
    reasons = []
    for spec in (
        {"type": "float", "low": 0.15, "high": 1.0, "step": 0.1},
        {"type": "float", "low": 0.0, "high": 1.0, "step": 0},
    ):
        with pytest.raises(LizyMLError) as caught:
            parse_space({"d": spec})
        reasons.append(caught.value.context["reason"])
    return tuple(reasons)


@pytest.mark.parametrize(("doc", "section"), _DOCS, ids=[d for d, _ in _DOCS])
def test_doc_states_the_number_literal_and_the_step_grid(
    doc: str, section: Callable[[], str]
) -> None:
    # H-0123 refused `step`; H-0124 accepts it on a grid with origin 0. The
    # documents state the number literal, the step's grid and its refusals.
    text = section()
    assert NUMBER_LITERAL.pattern in text, doc
    assert "`step`" in text, doc
    assert {"float", "int"} == {t for t, keys in DIM_KEYS.items() if "step" in keys}
    assert "H-0123" in text and "H-0124" in text, doc
    for reason in _step_reasons():
        assert f"`{reason}`" in text, (doc, reason)
    assert "0.30000000000000004" in text, doc
    assert "is refused (LizyML dimensions have no" not in text, doc
    assert "`step` は受け付けない" not in text, doc


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


def _changelog_entry(marker: str) -> str:
    """The one ``[0.19.0]`` bullet (a single line) containing *marker*."""
    (line,) = [ln for ln in _changelog_unreleased().splitlines() if marker in ln]
    return line


def test_changelog_lists_the_refused_and_accepted_forms() -> None:
    text = _changelog_unreleased()
    assert "H-0123" in text
    for form in ('`log: "false"`', "`1.7`", '`"abc"`', '`"16.0"`', '`"1e2"`'):
        assert form in text, form
    # H-0123's refusal of `step` was never released (H-0124 accepts it in the
    # same release), so its newly-refused list does not name it.
    assert "`step`" not in _changelog_entry("Newly refused:")


def test_changelog_states_the_step() -> None:
    entry = _changelog_entry("(H-0124,")
    for token in (
        "`step`",
        *(f"`{reason}`" for reason in _step_reasons()),
        "silently ignored",
        "half to even",
        "`0.30000000000000004`",
        "`Tuner.tune`",
        "LizyStudio#688",
    ):
        assert token in entry, token


def _magnitude_reasons() -> tuple[str, ...]:
    """The refusal reasons of H-0124 amendment 1 policy 9, read from the code."""
    reasons = []
    for spec in (
        {"type": "float", "low": -1e308, "high": 1e308},
        {"type": "int", "low": 0, "high": 2**52 + 1},
    ):
        with pytest.raises(LizyMLError) as caught:
            parse_space({"d": spec})
        reasons.append(caught.value.context["reason"])
    return tuple(reasons)


@pytest.mark.parametrize(
    ("doc", "section"),
    [*_DOCS, ("changelog", lambda: _changelog_entry("(H-0124,"))],
    ids=[*(d for d, _ in _DOCS), "changelog"],
)
def test_doc_states_the_magnitude_limits(doc: str, section: Callable[[], str]) -> None:
    text = section()
    assert len(_magnitude_reasons()) == 2
    for reason in _magnitude_reasons():
        assert f"`{reason}`" in text, (doc, reason)
    assert "2**52" in text, doc
