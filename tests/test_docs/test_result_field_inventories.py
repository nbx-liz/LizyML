"""The prose inventories of result fields list exactly the dataclass fields (#326).

``BLUEPRINT.md`` §7 and ``docs/api.md`` each enumerate the fields of
``FitResult`` and ``PredictionResult``. The golden tests pin the dataclasses, so
a field added there and to neither document kept CI green while the documents
went stale (DC3). This derives each inventory's field set from the document and
compares it with ``dataclasses.fields``.
"""

from __future__ import annotations

import dataclasses
import pathlib
import re

import pytest

from lizyml.core.types.fit_result import FitResult
from lizyml.core.types.predict_result import PredictionResult

ROOT = pathlib.Path(__file__).resolve().parents[2]
BLUEPRINT = ROOT / "BLUEPRINT.md"
API = ROOT / "docs" / "api.md"


def _section(text: str, start: str, end: str) -> list[str]:
    lines = text.splitlines()
    begin = next(i for i, line in enumerate(lines) if line.startswith(start))
    stop = next(i for i in range(begin + 1, len(lines)) if lines[i].startswith(end))
    return lines[begin + 1 : stop]


def _blueprint_fields(start: str, end: str) -> set[str]:
    """Top-level bullets that open with a code span: ``- `a / b` ...``."""
    names: set[str] = set()
    for line in _section(BLUEPRINT.read_text(encoding="utf-8"), start, end):
        match = re.match(r"^- `([^`]+)`", line)
        if match:
            names.update(part.strip() for part in match.group(1).split("/"))
    return names


def _api_fields(heading: str) -> set[str]:
    """First-column code spans of the table under ``heading``, up to ``---``."""
    names: set[str] = set()
    for line in _section(API.read_text(encoding="utf-8"), heading, "---"):
        match = re.match(r"^\| `([A-Za-z_][A-Za-z0-9_]*)` \|", line)
        if match:
            names.add(match.group(1))
    return names


CASES = [
    pytest.param(
        FitResult,
        _blueprint_fields("## 7.1 FitResult", "公開の戻り値"),
        id="FitResult-BLUEPRINT-7.1",
    ),
    pytest.param(
        PredictionResult,
        _blueprint_fields("## 7.3 PredictionResult", "補足:"),
        id="PredictionResult-BLUEPRINT-7.3",
    ),
    pytest.param(FitResult, _api_fields("### FitResult"), id="FitResult-api.md"),
    pytest.param(
        PredictionResult,
        _api_fields("### PredictionResult"),
        id="PredictionResult-api.md",
    ),
]


@pytest.mark.parametrize(("cls", "documented"), CASES)
def test_the_inventory_lists_exactly_the_dataclass_fields(
    cls: type, documented: set[str]
) -> None:
    actual = {f.name for f in dataclasses.fields(cls)}
    assert documented, "the inventory parsed to nothing; the parser lost its anchor"
    assert documented == actual, (
        f"missing from the document: {sorted(actual - documented)}; "
        f"not a field: {sorted(documented - actual)}"
    )
