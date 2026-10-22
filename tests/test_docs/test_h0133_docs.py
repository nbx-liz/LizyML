"""H-0133 criterion 9: the documents state what the implementation does."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from lizyml.core.category_values import ADMITTED_TYPES, MAX_DEPTH
from lizyml.persistence.exporter import FORMAT_VERSION

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10; pytest itself depends on tomli there
    import tomli as tomllib

REPO = Path(__file__).resolve().parents[2]


def _read(name: str) -> str:
    return (REPO / name).read_text(encoding="utf-8")


def _unreleased(changelog: str) -> str:
    start = changelog.index("## [0.20.0]")
    return changelog[start : changelog.index("\n## [", start + 1)]


def test_the_layer_table_puts_category_values_in_layer_0() -> None:
    row = next(
        line for line in _read("BLUEPRINT.md").splitlines() if line.startswith("| 0 |")
    )
    assert "`core/category_values.py`" in row


@pytest.mark.parametrize(
    "fragment",
    [
        f"`FORMAT_VERSION = {FORMAT_VERSION}`",
        "`stand_in_columns`",
        '{"$lizyml": 印, ...}',
        f"入れ物は {MAX_DEPTH} 段まで",
        "`categories_checked` は H-0133 で非推奨",
    ],
)
def test_blueprint_states_the_mechanism(fragment: str) -> None:
    assert fragment in _read("BLUEPRINT.md")


def test_the_blueprint_value_set_names_every_admitted_python_and_pandas_type() -> None:
    text = _read("BLUEPRINT.md")
    section = text[text.index("**値の集合 V**") :]
    section = section[: section.index("\n  - **拒否**")]
    for kind in ADMITTED_TYPES:
        if kind.__module__ in ("builtins", "datetime", "decimal"):
            assert re.search(rf"`{kind.__name__}`", section), kind
        if kind.__module__.startswith("pandas"):
            assert f"`pd.{kind.__name__}`" in section, kind


def test_migration_and_changelog_describe_format_version_3() -> None:
    # v0.20.0 shipped H-0133 with format_version 4 (H-0134 raised it in the
    # same release); H-0138 raised the current version later.
    migration = _read("docs/migration.md")
    assert "## v0.20.0" in migration
    assert "`format_version` 4" in migration
    unreleased = _unreleased(_read("CHANGELOG.md"))
    assert "H-0133" in unreleased
    assert "`format_version` 4" in unreleased
    assert "categories_checked=True" in unreleased


def test_the_deprecation_registry_lists_categories_checked() -> None:
    assert "`lizyml.persistence.export(categories_checked=True)`" in _read(
        "docs/DEPRECATIONS.md"
    )


def test_the_add_estimator_guide_points_to_the_value_set() -> None:
    assert "core/category_values.py" in _read("docs/add-estimator-guide.md")


def test_h0133_is_specified() -> None:
    rows = tomllib.loads(_read("docs/proposal_dispositions.toml"))["proposals"]
    assert rows["H-0133"]["disposition"] == "specified"
