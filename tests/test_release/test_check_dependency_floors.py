"""``scripts/check_dependency_floors.py`` (H-0126, #295).

The lowest-direct CI lane runs the script after installing every declared floor.
These tests fix its verdicts on synthetic declarations, and run it once against
the real installed metadata so a requirement the grammar cannot read fails here
and not only in that lane.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import pathlib
import sys
from types import ModuleType

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "lizyml_check_dependency_floors",
        ROOT / "scripts" / "check_dependency_floors.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


floors = _load()

INSTALLED = {"numpy": "1.24.0", "pandas": "2.0.0", "optuna": "3.1.0", "scipy": "1.10.0"}


def _installed(name: str) -> str | None:
    return INSTALLED.get(name)


def test_a_floor_equal_to_the_installed_version_passes() -> None:
    row = floors.check_requirement("numpy>=1.24", _installed)
    assert row.ok
    assert (row.name, row.floor, row.installed, row.extra) == (
        "numpy",
        "1.24",
        "1.24.0",
        None,
    )


def test_an_extra_requirement_is_checked_too() -> None:
    row = floors.check_requirement('optuna>=3.1; extra == "tuning"', _installed)
    assert row.ok
    assert row.extra == "tuning"


@pytest.mark.parametrize(
    "requirement, reason",
    [
        ("numpy>=1.23", "installed version is not the floor"),  # installed above
        ("pandas>=2.1", "installed version is not the floor"),  # installed below
        ('scipy>=1.11; extra == "calibration"', "installed version is not the floor"),
        ("pyyaml>=6.0", "not installed"),
    ],
)
def test_a_version_other_than_the_floor_fails(requirement: str, reason: str) -> None:
    row = floors.check_requirement(requirement, _installed)
    assert not row.ok
    assert row.reason == reason


@pytest.mark.parametrize(
    "requirement",
    [
        "numpy>=1.24,<3",  # two specifiers
        "numpy<3",  # an upper bound only
        "numpy==1.24.0",  # a pin
        "numpy~=1.24",
        "numpy",  # no floor at all
        'numpy>=1.24; python_version >= "3.11"',  # another marker
        'numpy>=1.24; extra == "a" and python_version >= "3.11"',
        "numpy[extra]>=1.24",  # extras on the requirement
        "numpy @ https://example.org/numpy.whl",
        "not a requirement !!",
    ],
)
def test_a_requirement_outside_the_grammar_is_unreadable(requirement: str) -> None:
    row = floors.check_requirement(requirement, _installed)
    assert not row.ok
    assert row.reason.startswith("unreadable:")


def test_an_empty_declaration_fails() -> None:
    rows = floors.check([], _installed)
    assert len(rows) == 1
    assert not rows[0].ok


def test_one_failure_fails_the_whole_check(monkeypatch: pytest.MonkeyPatch) -> None:
    declared = ["numpy>=1.24", 'optuna>=3.4; extra == "tuning"']
    monkeypatch.setattr(floors.importlib.metadata, "requires", lambda _: declared)
    monkeypatch.setattr(floors, "installed_version", _installed)
    assert floors.main() == 1


def test_the_table_names_every_requirement(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    declared = ["numpy>=1.24", 'scipy>=1.10; extra == "calibration"']
    monkeypatch.setattr(floors.importlib.metadata, "requires", lambda _: declared)
    monkeypatch.setattr(floors, "installed_version", _installed)
    assert floors.main() == 0
    out = capsys.readouterr().out
    assert "numpy" in out and "1.24.0" in out
    assert "scipy" in out and "calibration" in out


def test_the_real_declaration_is_inside_the_grammar() -> None:
    """Every requirement LizyML declares is readable (not necessarily at its
    floor here: this environment holds the locked versions)."""
    declared = importlib.metadata.requires("lizyml")
    assert declared
    rows = floors.check(declared, floors.installed_version)
    unreadable = [
        row.requirement for row in rows if row.reason.startswith("unreadable")
    ]
    assert unreadable == []
    assert {row.extra for row in rows} >= {
        None,
        "tuning",
        "explain",
        "plots",
        "calibration",
    }
