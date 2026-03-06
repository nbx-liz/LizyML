"""Tests for Phase 23-F — Logging output unification (H-0034).

Covers:
- setup_output_dir creates directory and log file
- Model with output_dir creates run directory on fit
- Model without output_dir has no side effects
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from lizyml import Model
from lizyml.core.logging import setup_output_dir

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reg_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {"feat_a": rng.uniform(0, 10, n), "feat_b": rng.uniform(-1, 1, n)}
    )
    df["target"] = df["feat_a"] * 2 + rng.normal(0, 0.5, n)
    return df


def _reg_config() -> dict:
    return {
        "config_version": 1,
        "task": "regression",
        "data": {"target": "target"},
        "split": {"method": "kfold", "n_splits": 3, "random_state": 42},
        "model": {"name": "lgbm", "params": {"n_estimators": 10}},
        "training": {"seed": 0},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSetupOutputDir:
    def test_creates_directory_and_log_file(self, tmp_path: Path) -> None:
        run_dir = setup_output_dir(tmp_path, "test-run-001")

        assert run_dir.exists()
        assert run_dir == tmp_path / "test-run-001"
        assert (run_dir / "run.log").exists()

        # Clean up handler to avoid affecting other tests
        root = logging.getLogger("lizyml")
        for h in list(root.handlers):
            if isinstance(h, logging.FileHandler):
                root.removeHandler(h)
                h.close()


class TestModelOutputDir:
    def test_fit_creates_run_dir(self, tmp_path: Path) -> None:
        df = _reg_df()
        model = Model(_reg_config(), data=df, output_dir=tmp_path)
        model.fit()

        assert model._run_dir is not None
        assert model._run_dir.exists()
        assert (model._run_dir / "run.log").exists()

        # Verify log file has content
        log_content = (model._run_dir / "run.log").read_text()
        assert len(log_content) > 0

        # Clean up handler
        root = logging.getLogger("lizyml")
        for h in list(root.handlers):
            if isinstance(h, logging.FileHandler):
                root.removeHandler(h)
                h.close()

    def test_no_output_dir_no_side_effects(self) -> None:
        df = _reg_df()
        model = Model(_reg_config(), data=df)
        model.fit()

        assert model._run_dir is None
