"""H-0130 negative controls: each new test catches the defect it is for.

Every control patches one mechanism into a plausible wrong implementation, runs
the tests that guard it, and expects at least one named test to FAIL. A control
whose tests all pass means the tests do not guard that mechanism; the script
then exits 1.

Usage (from the repository root): uv run python docs/audits/2026-10-h0130/probes/negative_controls_h0130.py
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import tests.test_architecture._layer_rules as lr  # noqa: E402

LAYER = "tests/test_architecture/test_layer_rules.py"
CONTRACTS = "tests/test_architecture/test_h0130_contracts.py"


def _lenient_grammar() -> None:
    """Strip trailing spaces before matching (a 'tolerant' reader)."""
    original = lr._read_allowances

    def read(lines: list[str], *args: Any) -> Any:
        return original([line.rstrip() for line in lines], *args)

    lr._read_allowances = read  # type: ignore[assignment]


def _allow_every_layer_1_pair() -> None:
    """R2 ignores which pairs the table declares."""
    original = lr.read_table

    def read(blueprint: Path) -> Any:
        table = original(blueprint)
        leaf = [m for m in table.packages if table.paths[m] == 1 and m.count(".") == 1]
        pairs = frozenset((a, b) for a in leaf for b in leaf if a != b)
        return lr.Table(table.paths, table.packages, table.interfaces, pairs)

    lr.read_table = read  # type: ignore[assignment]


def _no_staleness() -> None:
    """An allowance no import uses is not reported."""
    original = lr.check

    def check(*args: Any, **kwargs: Any) -> list[str]:
        problems = original(*args, **kwargs)
        return [p for p in problems if "is used by no import" not in p]

    lr.check = check  # type: ignore[assignment]
    import tests.test_architecture.test_layer_rules as t

    t.check = check  # type: ignore[assignment]


def _no_dispatch_rule() -> None:
    """R6 switched off: no concrete dispatch or registry access is reported."""
    lr.dispatch_problems = lambda *args, **kwargs: []  # type: ignore[assignment]


def _registry_access_anywhere() -> None:
    """Registry access allowed in every module (the two-module rule dropped)."""
    import tests.test_architecture._dispatch_rules as dr

    original = dr.dispatch_problems

    def problems(*args: Any, **kwargs: Any) -> list[str]:
        return [p for p in original(*args, **kwargs) if " uses " not in p]

    lr.dispatch_problems = problems  # type: ignore[assignment]


def _plots_accept_anything() -> None:
    """The plots check a marker every object satisfies."""
    import lizyml.calibration.base as base

    base.BaseCalibrationResult = object  # type: ignore[misc, assignment]


CONTROLS: dict[str, tuple[Callable[[], None], list[str]]] = {
    "lenient allowance grammar": (
        _lenient_grammar,
        [
            f"{LAYER}::test_an_allowance_block_outside_the_grammar_fails"
            "[lead trailing space]",
            f"{LAYER}::test_an_allowance_block_outside_the_grammar_fails"
            "[row trailing space]",
        ],
    ),
    "R2 blind to the table": (
        _allow_every_layer_1_pair,
        [f"{LAYER}::test_an_undeclared_layer_1_pair_fails"],
    ),
    "no staleness check": (
        _no_staleness,
        [f"{LAYER}::test_an_allowance_no_import_uses_fails"],
    ),
    "R6 switched off": (
        _no_dispatch_rule,
        [
            f"{LAYER}::test_a_concrete_type_dispatch_outside_the_list_fails[isinstance]",
            f"{LAYER}::test_a_second_dispatch_of_a_listed_pair_fails",
        ],
    ),
    "registry access anywhere": (
        _registry_access_anywhere,
        [f"{LAYER}::test_registry_access_outside_the_two_registries_fails[get]"],
    ),
    "plots accept any calibrator": (
        _plots_accept_anything,
        [
            f"{CONTRACTS}::test_a_calibrator_without_the_marker_is_refused_as_before",
        ],
    ),
}


class _Patch:
    def __init__(self, apply: Callable[[], None]) -> None:
        self.apply = apply

    def pytest_configure(self, config: Any) -> None:
        self.apply()


def main() -> int:
    failures = 0
    for name, (apply, tests) in CONTROLS.items():
        code = pytest.main(
            ["-q", "-p", "no:cacheprovider", "-p", "no:randomly", *tests],
            plugins=[_Patch(apply)],
        )
        caught = code == pytest.ExitCode.TESTS_FAILED
        print(f"CONTROL {name}: {'caught (tests failed)' if caught else 'NOT CAUGHT'}")
        failures += not caught
    print(f"\n{len(CONTROLS) - failures}/{len(CONTROLS)} controls caught")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
