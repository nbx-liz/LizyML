"""The PR 9 fold-map check refuses an incomplete map (H-0110, acceptance r1).

``instruments/pr9_fold_map_check.py`` is the mechanical half of acceptance criterion
A6. Round 1 deleted an owed row in memory and the check stayed green; these tests
pin that the map is compared with the inventory in both directions.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
AUDIT = ROOT / "docs" / "audits" / "2026-09-defect-discovery"
_spec = importlib.util.spec_from_file_location(
    "pr9_fold_map_check", AUDIT / "instruments" / "pr9_fold_map_check.py"
)
assert _spec is not None and _spec.loader is not None
check = importlib.util.module_from_spec(_spec)
sys.modules["pr9_fold_map_check"] = check
_spec.loader.exec_module(check)


@pytest.fixture(scope="module")
def real() -> tuple[list[list[str]], set[str], str]:
    inventories = [
        json.loads((AUDIT / "results" / f"pr9_inventory_{b}.json").read_text("utf-8"))
        for b in ("A", "B")
    ]
    rows = check.parse_rows((AUDIT / "results" / "pr9_fold_map.md").read_text("utf-8"))
    blueprint = (ROOT / "BLUEPRINT.md").read_text(encoding="utf-8")
    return rows, check.expected_ids(inventories), blueprint


def test_the_shipped_map_is_complete(
    real: tuple[list[list[str]], set[str], str],
) -> None:
    rows, expected, blueprint = real
    # Vacuity guard: 127 owed clauses (one split in two) plus 13 decision rows.
    assert len(expected) == 141
    assert check.problems(rows, expected, blueprint) == []


def test_a_deleted_owed_row_is_reported(
    real: tuple[list[list[str]], set[str], str],
) -> None:
    rows, expected, blueprint = real
    kept = [cells for cells in rows if cells[0] != "H-0083#0"]
    assert len(kept) == len(rows) - 1
    assert check.problems(kept, expected, blueprint) == [
        "H-0083#0: owed but not mapped"
    ]


def test_a_duplicated_or_unowed_row_is_reported(
    real: tuple[list[list[str]], set[str], str],
) -> None:
    rows, expected, blueprint = real
    first = rows[0]
    extra = ["H-0024#0", *first[1:]]
    found = check.problems([*rows, first, extra], expected, blueprint)
    assert f"{first[0]}: mapped 2 times" in found
    assert "H-0024#0: mapped but not owed" in found


def test_a_quote_absent_from_blueprint_is_reported(
    real: tuple[list[list[str]], set[str], str],
) -> None:
    rows, expected, blueprint = real
    changed = [list(cells) for cells in rows]
    folded = next(c for c in changed if c[2].strip("`") == "folded")
    folded[3] = "`no such sentence anywhere`"
    found = check.problems(changed, expected, blueprint)
    assert found == [
        f"{folded[0]}: quote not in BLUEPRINT.md: 'no such sentence anywhere'"
    ]
