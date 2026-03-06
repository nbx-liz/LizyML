"""Notebook execution tests — verify notebooks run end-to-end without errors.

Uses nbconvert's ExecutePreprocessor to execute each notebook in-process.
Marked with ``pytest.mark.slow`` so they can be skipped with ``-m "not slow"``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbconvert_pp = pytest.importorskip(
    "nbconvert.preprocessors",
)

NOTEBOOKS_DIR = Path(__file__).resolve().parents[2] / "notebooks"

_NOTEBOOKS = [
    "tutorial_binary_lgbm.ipynb",
    "tutorial_multiclass_lgbm.ipynb",
    "tutorial_regression_lgbm.ipynb",
    "tutorial_regression_tuning_lgbm.ipynb",
]


@pytest.mark.slow
@pytest.mark.parametrize("notebook_name", _NOTEBOOKS)
def test_notebook_executes(notebook_name: str) -> None:
    """Execute a notebook and assert no CellExecutionError."""
    path = NOTEBOOKS_DIR / notebook_name
    assert path.exists(), f"Notebook not found: {path}"

    nb = nbformat.read(str(path), as_version=4)
    ep = nbconvert_pp.ExecutePreprocessor(
        timeout=180,
        kernel_name="python3",
    )
    ep.preprocess(nb, {"metadata": {"path": str(NOTEBOOKS_DIR)}})
