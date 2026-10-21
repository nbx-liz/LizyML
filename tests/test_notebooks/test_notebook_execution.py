"""Notebook tests.

Two tiers:
- A **slow** tier executes every notebook end-to-end via nbconvert (skipped by
  ``-m "not slow"`` on develop PRs).
- A **fast** tier (runs on develop PRs) statically validates every notebook —
  it must parse, contain runnable code, and use LizyML. This catches missing,
  empty, or unreferenced notebooks without paying the execution cost (H-0178
  item 6: 3 tutorials were previously referenced by no test).

Notebooks are auto-discovered so a newly added tutorial cannot silently escape
coverage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.test_notebooks import network_guard

nbformat = pytest.importorskip("nbformat")
nbconvert_pp = pytest.importorskip(
    "nbconvert.preprocessors",
)
nbclient_exc = pytest.importorskip("nbclient.exceptions")

NOTEBOOKS_DIR = Path(__file__).resolve().parents[2] / "notebooks"

_ALL_NOTEBOOKS = sorted(p.name for p in NOTEBOOKS_DIR.glob("*.ipynb"))

# Tutorials that must always ship (guards against accidental deletion and
# pins the three that were previously unreferenced by any test).
_REQUIRED_NOTEBOOKS = {
    "tutorial_binary_lgbm.ipynb",
    "tutorial_multiclass_lgbm.ipynb",
    "tutorial_regression_lgbm.ipynb",
    "tutorial_regression_tuning_lgbm.ipynb",
    "tutorial_time_series_lgbm.ipynb",
    "tutorial_calibration.ipynb",
    "tutorial_codegen_export.ipynb",
    "tutorial_shap_explanations.ipynb",
}


def test_all_required_notebooks_present() -> None:
    """Every expected tutorial exists and is discovered (no orphan notebooks)."""
    missing = _REQUIRED_NOTEBOOKS - set(_ALL_NOTEBOOKS)
    assert not missing, f"Required notebooks missing: {sorted(missing)}"


@pytest.mark.parametrize("notebook_name", _ALL_NOTEBOOKS)
def test_notebook_is_valid_and_uses_lizyml(notebook_name: str) -> None:
    """Fast static check: parses, has runnable code, and imports LizyML."""
    nb = nbformat.read(str(NOTEBOOKS_DIR / notebook_name), as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code" and c.source.strip()]
    assert code_cells, f"{notebook_name} has no non-empty code cells"
    sources = "\n".join(c.source for c in code_cells)
    assert "lizyml" in sources, f"{notebook_name} does not use lizyml"


def execute_guarded(nb: Any, workdir: Path, cell_timeout: int) -> None:
    """Run ``nb`` in ``workdir`` with the network guard installed and checked.

    The guard (H-0137) goes in as the first cell and is checked by a last
    cell, so a remote load fails the run even when the notebook catches it.
    The file on disk is not changed.
    """
    nb.cells.insert(0, nbformat.v4.new_code_cell(network_guard.setup_source()))
    nb.cells.append(nbformat.v4.new_code_cell(network_guard.check_source()))
    ep = nbconvert_pp.ExecutePreprocessor(timeout=cell_timeout, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(workdir)}})


@pytest.mark.slow
@pytest.mark.parametrize("notebook_name", _ALL_NOTEBOOKS)
def test_notebook_executes(notebook_name: str) -> None:
    """Execute a notebook and assert no CellExecutionError.

    It reads its data from ``notebooks/data/``, so the guard does not fire;
    ``test_the_guard_fires_in_this_layer`` shows that it would.
    """
    path = NOTEBOOKS_DIR / notebook_name
    assert path.exists(), f"Notebook not found: {path}"

    nb = nbformat.read(str(path), as_version=4)
    # Tuning notebook runs Optuna trials — needs more time on CI
    cell_timeout = 600 if "tuning" in notebook_name else 180
    execute_guarded(nb, NOTEBOOKS_DIR, cell_timeout)


_URL = "https://lizyml-h0137.invalid/x.csv"  # never resolvable (RFC 2606)
_FETCH = f"import urllib.request\nurllib.request.urlopen({_URL!r})"
_SWALLOWED = (
    "import urllib.request\n"
    "try:\n"
    f"    urllib.request.urlopen({_URL!r})\n"
    "except Exception as exc:\n"
    "    assert 'H-0137' in str(exc)  # the call was made, and refused\n"
)
_LOCAL_READ = (
    "import pandas as pd\nassert pd.read_csv('data/vehicle.csv.gz').shape == (846, 19)"
)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("case", "source", "tags"),
    [
        ("a network call", _FETCH, []),
        ("a network call the notebook swallows", _SWALLOWED, []),
        ("a network call in a cell allowed to raise", _FETCH, ["raises-exception"]),
    ],
)
def test_the_guard_fires_in_this_layer(case: str, source: str, tags: list[str]) -> None:
    """The same wiring as test_notebook_executes fails a remote load (H-0137)."""
    del case
    cell = nbformat.v4.new_code_cell(source, metadata={"tags": tags})
    nb = nbformat.v4.new_notebook(cells=[cell])
    with pytest.raises(nbclient_exc.CellExecutionError, match=network_guard.MESSAGE):
        execute_guarded(nb, NOTEBOOKS_DIR, 120)


@pytest.mark.slow
def test_the_twin_without_a_network_call_runs_in_this_layer() -> None:
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(_LOCAL_READ)])
    execute_guarded(nb, NOTEBOOKS_DIR, 120)
