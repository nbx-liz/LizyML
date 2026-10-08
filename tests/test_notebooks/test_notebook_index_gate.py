"""The notebook index gate job's decision (H-0119 6., review round 1, finding 5).

The gate passes in exactly two cases: the scope job succeeded with ``run=true``
and both substantive jobs succeeded, or the scope job succeeded with
``run=false`` and both were skipped. Every other combination fails, so a scope
failure can no longer turn the substantive jobs into silent skips.
"""

from __future__ import annotations

import importlib.util
import itertools
import pathlib
import sys
from types import ModuleType

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ("success", "failure", "cancelled", "skipped")
RUNS = ("true", "false", "")


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "lizyml_notebook_index_gate", ROOT / "scripts" / "notebook_index_gate.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load()

PASSING = {
    ("success", "true", "success", "success"),
    ("success", "false", "skipped", "skipped"),
}


@pytest.mark.parametrize(
    "combination", list(itertools.product(RESULTS, RUNS, RESULTS, RESULTS))
)
def test_only_the_two_consistent_combinations_pass(
    combination: tuple[str, str, str, str],
) -> None:
    ok, reason = gate.decide(*combination)
    assert ok is (combination in PASSING), reason
    assert reason


@pytest.mark.parametrize(
    "combination",
    [
        ("success", "true", "skipped", "success"),  # run=true but a job skipped
        ("success", "true", "success", "skipped"),
        ("failure", "", "skipped", "skipped"),  # the scope failed: not a skip
        ("cancelled", "", "skipped", "skipped"),
        ("success", "false", "success", "success"),  # ran although run=false
        ("success", "TRUE", "success", "success"),  # unknown run value
        ("success", "true", "neutral", "success"),  # unknown result
    ],
)
def test_named_inconsistent_combinations_fail(
    combination: tuple[str, str, str, str],
) -> None:
    assert gate.decide(*combination)[0] is False


def test_main_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "SCOPE_RESULT": "success",
        "SCOPE_RUN": "true",
        "REGISTRY_RESULT": "success",
        "EXECUTION_RESULT": "success",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    assert gate.main() == 0
    monkeypatch.setenv("EXECUTION_RESULT", "failure")
    assert gate.main() == 1
    monkeypatch.delenv("SCOPE_RUN")
    with pytest.raises(KeyError):
        gate.main()
