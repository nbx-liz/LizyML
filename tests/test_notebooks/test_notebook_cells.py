"""Static validation of notebook parameter cells.

Verifies that all tutorial notebooks use ``model.params_table()``
for parameter display (22-M / H-0035) and that Config cells contain
the required smart parameter keywords.
"""

from __future__ import annotations

from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

NOTEBOOKS_DIR = Path(__file__).resolve().parents[2] / "notebooks"

REGRESSION_NB = NOTEBOOKS_DIR / "tutorial_regression_lgbm.ipynb"
BINARY_NB = NOTEBOOKS_DIR / "tutorial_binary_lgbm.ipynb"
MULTICLASS_NB = NOTEBOOKS_DIR / "tutorial_multiclass_lgbm.ipynb"
TUNING_NB = NOTEBOOKS_DIR / "tutorial_regression_tuning_lgbm.ipynb"
TIME_SERIES_NB = NOTEBOOKS_DIR / "tutorial_time_series_lgbm.ipynb"


def _read_code_cells(path: Path) -> str:
    """Concatenate all code cell sources into a single string."""
    nb = nbformat.read(str(path), as_version=4)
    return "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")


# --- Config keywords required in all 3 main notebooks ---

CONFIG_KEYWORDS = [
    "min_data_in_leaf_ratio",
    "min_data_in_bin_ratio",
]


@pytest.mark.parametrize(
    "notebook_path",
    [REGRESSION_NB, BINARY_NB, MULTICLASS_NB],
    ids=["regression", "binary", "multiclass"],
)
@pytest.mark.parametrize("keyword", CONFIG_KEYWORDS)
def test_config_keywords_present(notebook_path: Path, keyword: str) -> None:
    code = _read_code_cells(notebook_path)
    assert keyword in code, f"'{keyword}' not found in {notebook_path.name}"


# --- params_table() call required in all 4 notebooks ---


@pytest.mark.parametrize(
    "notebook_path",
    [REGRESSION_NB, BINARY_NB, MULTICLASS_NB, TUNING_NB, TIME_SERIES_NB],
    ids=["regression", "binary", "multiclass", "tuning", "time_series"],
)
def test_params_table_present(notebook_path: Path) -> None:
    code = _read_code_cells(notebook_path)
    assert "params_table()" in code, (
        f"'params_table()' not found in {notebook_path.name}"
    )


# --- Task-specific keywords in config ---


def test_binary_has_scale_pos_weight_or_balanced() -> None:
    code = _read_code_cells(BINARY_NB)
    assert "balanced" in code


def test_multiclass_has_balanced() -> None:
    code = _read_code_cells(MULTICLASS_NB)
    assert "balanced" in code


# --- feature_weights resolved cell in tuning notebook ---


def test_time_series_has_split_summary() -> None:
    """Time series notebook must call split_summary()."""
    code = _read_code_cells(TIME_SERIES_NB)
    assert "split_summary()" in code, (
        f"'split_summary()' not found in {TIME_SERIES_NB.name}"
    )


def test_time_series_has_time_series_method() -> None:
    """Time series notebook must use time_series split method."""
    code = _read_code_cells(TIME_SERIES_NB)
    assert "time_series" in code, (
        f"'time_series' split method not found in {TIME_SERIES_NB.name}"
    )


def test_time_series_has_predict() -> None:
    """Time series notebook must include predict demonstration."""
    code = _read_code_cells(TIME_SERIES_NB)
    assert ".predict(" in code, f"'.predict(' not found in {TIME_SERIES_NB.name}"


def test_time_series_has_purged() -> None:
    """Time series notebook must include purged_time_series example."""
    code = _read_code_cells(TIME_SERIES_NB)
    assert "purged_time_series" in code, (
        f"'purged_time_series' not found in {TIME_SERIES_NB.name}"
    )


def test_tuning_has_feature_weights_resolved_cell() -> None:
    """Tuning notebook must have a cell checking resolved feature_weights."""
    code = _read_code_cells(TUNING_NB)
    assert "feature_weights" in code, (
        f"'feature_weights' resolved cell not found in {TUNING_NB.name}"
    )
