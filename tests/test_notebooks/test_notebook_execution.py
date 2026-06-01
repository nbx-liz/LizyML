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

import pytest

nbformat = pytest.importorskip("nbformat")
nbconvert_pp = pytest.importorskip(
    "nbconvert.preprocessors",
)
nbclient_exc = pytest.importorskip("nbclient.exceptions")

NOTEBOOKS_DIR = Path(__file__).resolve().parents[2] / "notebooks"

# Some tutorials fetch a remote dataset (e.g. OpenML credit-g). When the CI
# runner cannot reach the network, that is an environment outage, not a
# notebook regression — skip rather than fail the (release-gating) slow run.
_NETWORK_ERROR_MARKERS = (
    "HTTPError",
    "URLError",
    "OpenMLError",
    "api.openml.org",
    "Max retries",
    "ConnectionError",
    "Temporary failure in name resolution",
    "network error",
)

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


@pytest.mark.slow
@pytest.mark.parametrize("notebook_name", _ALL_NOTEBOOKS)
def test_notebook_executes(notebook_name: str) -> None:
    """Execute a notebook and assert no CellExecutionError."""
    path = NOTEBOOKS_DIR / notebook_name
    assert path.exists(), f"Notebook not found: {path}"

    nb = nbformat.read(str(path), as_version=4)
    # Tuning notebook runs Optuna trials — needs more time on CI
    cell_timeout = 600 if "tuning" in notebook_name else 180
    ep = nbconvert_pp.ExecutePreprocessor(
        timeout=cell_timeout,
        kernel_name="python3",
    )
    try:
        ep.preprocess(nb, {"metadata": {"path": str(NOTEBOOKS_DIR)}})
    except nbclient_exc.CellExecutionError as exc:
        message = str(exc)
        if any(marker in message for marker in _NETWORK_ERROR_MARKERS):
            pytest.skip(
                f"{notebook_name}: remote dataset fetch failed (network "
                f"unavailable in CI), not a notebook regression."
            )
        raise
