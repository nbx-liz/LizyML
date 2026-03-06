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

    def test_output_dir_from_config(self, tmp_path: Path) -> None:
        """output_dir specified in Config should create run directory."""
        df = _reg_df()
        cfg = _reg_config()
        cfg["output_dir"] = str(tmp_path)
        model = Model(cfg, data=df)
        model.fit()

        assert model._run_dir is not None
        assert model._run_dir.exists()

        # Clean up handler
        root = logging.getLogger("lizyml")
        for h in list(root.handlers):
            if isinstance(h, logging.FileHandler):
                root.removeHandler(h)
                h.close()

    def test_constructor_overrides_config(self, tmp_path: Path) -> None:
        """Constructor output_dir should take priority over Config."""
        df = _reg_df()
        cfg = _reg_config()
        cfg["output_dir"] = str(tmp_path / "config_dir")
        constructor_dir = tmp_path / "constructor_dir"
        model = Model(cfg, data=df, output_dir=constructor_dir)
        model.fit()

        assert model._run_dir is not None
        # Should be under constructor_dir, not config_dir
        assert str(model._run_dir).startswith(str(constructor_dir))

        # Clean up handler
        root = logging.getLogger("lizyml")
        for h in list(root.handlers):
            if isinstance(h, logging.FileHandler):
                root.removeHandler(h)
                h.close()


def _cleanup_file_handlers() -> None:
    root = logging.getLogger("lizyml")
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler):
            root.removeHandler(h)
            h.close()


class TestExportOutputDir:
    def test_export_explicit_path(self, tmp_path: Path) -> None:
        """Explicit path argument is used directly (backward compat)."""
        df = _reg_df()
        model = Model(_reg_config(), data=df)
        model.fit()

        export_dir = tmp_path / "custom_export"
        result = model.export(path=export_dir)

        assert result == export_dir
        assert (export_dir / "metadata.json").exists()

    def test_export_uses_run_dir_after_fit(self, tmp_path: Path) -> None:
        """export() with no path uses {run_dir}/export after fit."""
        df = _reg_df()
        model = Model(_reg_config(), data=df, output_dir=tmp_path)
        model.fit()

        result = model.export()

        assert result == model._run_dir / "export"
        assert (result / "metadata.json").exists()

        _cleanup_file_handlers()

    def test_export_creates_run_dir_from_output_dir(self, tmp_path: Path) -> None:
        """export() creates a new run dir when only output_dir is set."""
        df = _reg_df()
        model = Model(_reg_config(), data=df, output_dir=tmp_path)
        model.fit()
        # Reset run_dir to simulate export without prior run dir
        model._run_dir = None

        result = model.export()

        assert result.name == "export"
        assert model._run_dir is not None
        assert (result / "metadata.json").exists()

        _cleanup_file_handlers()

    def test_export_no_path_no_output_dir_raises(self) -> None:
        """export() with no path and no output_dir raises error."""
        import pytest

        from lizyml.core.exceptions import LizyMLError

        df = _reg_df()
        model = Model(_reg_config(), data=df)
        model.fit()

        with pytest.raises(LizyMLError, match="No export path"):
            model.export()

    def test_export_returns_path(self, tmp_path: Path) -> None:
        """export() returns a Path object."""
        df = _reg_df()
        model = Model(_reg_config(), data=df)
        model.fit()

        result = model.export(path=tmp_path / "out")
        assert isinstance(result, Path)
