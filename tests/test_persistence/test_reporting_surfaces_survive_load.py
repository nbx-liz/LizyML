"""Every reading a reporting surface gives survives ``export -> load`` (H-0109).

The plan's permanent check for PR 8b, executed rather than argued. It is a
bounded regression contract, not a proof that every possible reading survives:
the population is the public ``Model`` names in ``_load_census.INVENTORY``, read
with their defaults, over four configurations and five lifecycles. The set of
cells whose readings differ must be **exactly** the declared exception set, so a
new difference fails here, and so does fixing a declared exception without
updating the declaration.

Two exceptions are declared, each measured (``results/pr8b_measurements.txt``):

* ``importance("gain")`` -- LightGBM's model text writes ``split_gain=`` with six
  significant digits and reads it back as binary32, so a loaded booster's gain
  moves by at most ``GAIN_RTOL`` relative plus ``GAIN_ATOL`` absolute (the
  subnormal range). Compared within that bound and counted as agreeing.
* ``tuning_table`` / ``tuning_plot`` / ``boundary_table`` -- H-0086 persists the
  tuning result's ``best_*`` overlay and not its trials, rounds or boundary
  report. How the surfaces present that is #315's decision.

Before H-0109 the 24 #281 cells differed too: ``params_table`` and
``export_code`` reported the configured inner-validation ratio after
``tune -> fit -> export -> load``, where the fit trained with the tuned one.
"""

from __future__ import annotations

from typing import Any

import pytest

from lizyml.core.model import Model
from tests.test_persistence import _load_census as census


@pytest.fixture(scope="module")
def cells(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, Any]]:
    return census.collect(tmp_path_factory.mktemp("survive_load"))


def test_the_inventory_classifies_every_public_name() -> None:
    """A new public method fails here until it is read or excluded with a reason."""
    public = {name for name in dir(Model) if not name.startswith("_")}
    assert set(census.INVENTORY) == public, sorted(set(census.INVENTORY) ^ public)
    read_through = {
        surface
        for entry in census.INVENTORY.values()
        if isinstance(entry, tuple)
        for surface in entry
    }
    assert read_through == set(census.SURFACES)


def test_every_reading_survives_load_except_the_declared(
    cells: list[dict[str, Any]],
) -> None:
    expected = (
        len(census.CONFIGURATIONS) * len(census.LIFECYCLES) * len(census.SURFACES)
    )
    assert len(cells) == expected
    differing = {
        entry["cell"]
        for entry in cells
        if not census.agrees(entry["cell"][2], entry["before"], entry["after"])
    }

    assert differing - census.DECLARED == set(), sorted(differing - census.DECLARED)
    # A declared cell that agrees means #315 changed: update DECLARED with it.
    assert census.DECLARED - differing == set(), sorted(census.DECLARED - differing)


def test_every_surface_was_exercised(cells: list[dict[str, Any]]) -> None:
    """A surface that raised everywhere would agree everywhere, and prove nothing."""
    exercised = {entry["cell"][2] for entry in cells if entry["before"][0] == "ok"}
    assert exercised == set(census.SURFACES), sorted(set(census.SURFACES) - exercised)

    tuned_ratio_cells = [
        entry
        for entry in cells
        if entry["cell"][2] == "params_table"
        and entry["before"][0] == "ok"
        and entry["before"][1].loc["validation_ratio", "value"] == census.TUNED_RATIO
    ]
    # tune_fit, tune_fit_reexport and tune_resume_fit, in four configurations:
    # the cells #281 was about, so the check is not passing on config values.
    assert len(tuned_ratio_cells) == 12, len(tuned_ratio_cells)
