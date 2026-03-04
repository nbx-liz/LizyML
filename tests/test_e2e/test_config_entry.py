"""Tests for T-7 — Config/Data entry-point contract.

Covers:
- Model(config=str)           JSON file path (str)
- Model(config=Path)          YAML file path (Path)
- Model(config=LizyMLConfig)  pre-validated instance (no double-validate)
- Model.fit() with data.path  data loaded from CSV referenced in config
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from lizyml import Model
from lizyml.config.loader import load_config
from lizyml.config.schema import LizyMLConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reg_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "feat_a": rng.uniform(0, 10, n),
            "feat_b": rng.uniform(-1, 1, n),
        }
    )
    df["target"] = df["feat_a"] * 2.0 + df["feat_b"]
    return df


def _base_cfg() -> dict:
    return {
        "config_version": 1,
        "task": "regression",
        "data": {"target": "target"},
        "split": {"method": "kfold", "n_splits": 3, "random_state": 42},
        "model": {"name": "lgbm", "params": {"n_estimators": 10}},
        "training": {"seed": 0},
    }


# ---------------------------------------------------------------------------
# T-7a: JSON file path
# ---------------------------------------------------------------------------


class TestJsonConfigFile:
    def test_model_fit_predict_via_json_path_str(self, tmp_path: Path) -> None:
        """Model(config=str) with JSON file: fit and predict work end-to-end."""
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(_base_cfg()), encoding="utf-8")

        df = _reg_df()
        m = Model(str(cfg_path))
        result = m.fit(data=df)
        assert result.feature_names == ["feat_a", "feat_b"]

        X = df.drop(columns=["target"])
        pred = m.predict(X)
        assert pred.pred.shape == (len(df),)

    def test_json_config_normalizes_aliases(self, tmp_path: Path) -> None:
        """Alias normalization (k-fold → kfold) works when loading from JSON."""
        cfg = _base_cfg()
        cfg["split"]["method"] = "k-fold"
        cfg_path = tmp_path / "config_alias.json"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

        m = Model(str(cfg_path))
        result = m.fit(data=_reg_df())
        assert len(result.splits.outer) == 3


# ---------------------------------------------------------------------------
# T-7b: YAML file path
# ---------------------------------------------------------------------------


class TestYamlConfigFile:
    def test_model_fit_predict_via_yaml_path(self, tmp_path: Path) -> None:
        """Model(config=Path) with YAML file: fit and predict work end-to-end."""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(_base_cfg()), encoding="utf-8")

        df = _reg_df()
        m = Model(cfg_path)
        result = m.fit(data=df)
        assert result.feature_names == ["feat_a", "feat_b"]

        X = df.drop(columns=["target"])
        pred = m.predict(X)
        assert pred.pred.shape == (len(df),)

    def test_yml_extension_works(self, tmp_path: Path) -> None:
        """Files with .yml extension are parsed as YAML."""
        cfg_path = tmp_path / "config.yml"
        cfg_path.write_text(yaml.safe_dump(_base_cfg()), encoding="utf-8")

        m = Model(cfg_path)
        result = m.fit(data=_reg_df())
        assert isinstance(result.feature_names, list)


# ---------------------------------------------------------------------------
# T-7c: LizyMLConfig instance (no double-validation)
# ---------------------------------------------------------------------------


class TestLizyMLConfigInstance:
    def test_model_accepts_lizymlconfig_directly(self) -> None:
        """Model(config=LizyMLConfig) must not re-validate (config passes through)."""
        cfg_obj = load_config(_base_cfg())
        assert isinstance(cfg_obj, LizyMLConfig)

        df = _reg_df()
        m = Model(cfg_obj)
        result = m.fit(data=df)
        assert result.feature_names == ["feat_a", "feat_b"]

    def test_lizymlconfig_instance_skips_load_config(self) -> None:
        """Passing a LizyMLConfig instance must use it directly (no re-parse).

        We verify this by constructing an already-validated config and
        confirming the model runs without error, even though load_config
        would normally be called for string/dict inputs.
        """
        cfg_obj = load_config(_base_cfg())
        m = Model(cfg_obj)
        # _cfg must be the same object (identity) - no re-wrapping
        assert m._cfg is cfg_obj


# ---------------------------------------------------------------------------
# T-7d: data.path in config (fit without explicit data= argument)
# ---------------------------------------------------------------------------


class TestDataPathInConfig:
    def test_fit_loads_data_from_csv_path(self, tmp_path: Path) -> None:
        """Model.fit() with no data= arg loads data from config.data.path (CSV)."""
        df = _reg_df()
        csv_path = tmp_path / "train.csv"
        df.to_csv(csv_path, index=False)

        cfg = _base_cfg()
        cfg["data"]["path"] = str(csv_path)

        m = Model(cfg)
        result = m.fit()  # no data= argument
        assert result.feature_names == ["feat_a", "feat_b"]

        X = df.drop(columns=["target"])
        pred = m.predict(X)
        assert pred.pred.shape == (len(df),)

    def test_explicit_data_arg_overrides_config_path(self, tmp_path: Path) -> None:
        """data= passed to fit() overrides config.data.path."""
        # Write a *different* dataset to the path
        other_df = _reg_df(n=50)
        csv_path = tmp_path / "other.csv"
        other_df.to_csv(csv_path, index=False)

        cfg = _base_cfg()
        cfg["data"]["path"] = str(csv_path)

        df = _reg_df(n=100)
        m = Model(cfg)
        result = m.fit(data=df)  # explicit data= should win
        # Verify we trained on 100 rows (from df), not 50 (from csv)
        all_valid = np.concatenate([v for _, v in result.splits.outer])
        assert len(all_valid) == 100
