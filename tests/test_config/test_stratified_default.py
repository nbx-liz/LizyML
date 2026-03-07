"""Tests for StratifiedKFold default for classification tasks (H-0013)."""

from __future__ import annotations

import warnings

import pytest

from lizyml.config.loader import load_config
from lizyml.core._model_factories import build_splitter


class TestStratifiedDefault:
    """Test that classification tasks default to stratified_kfold."""

    def test_binary_no_split_defaults_to_stratified(self) -> None:
        raw = {
            "config_version": 1,
            "task": "binary",
            "data": {"target": "y"},
            "model": {"name": "lgbm"},
        }
        cfg = load_config(raw)
        assert cfg.split.method == "stratified_kfold"

    def test_multiclass_no_split_defaults_to_stratified(self) -> None:
        raw = {
            "config_version": 1,
            "task": "multiclass",
            "data": {"target": "y"},
            "model": {"name": "lgbm"},
        }
        cfg = load_config(raw)
        assert cfg.split.method == "stratified_kfold"

    def test_regression_no_split_defaults_to_kfold(self) -> None:
        raw = {
            "config_version": 1,
            "task": "regression",
            "data": {"target": "y"},
            "model": {"name": "lgbm"},
        }
        cfg = load_config(raw)
        assert cfg.split.method == "kfold"

    def test_explicit_kfold_preserved(self) -> None:
        raw = {
            "config_version": 1,
            "task": "binary",
            "data": {"target": "y"},
            "model": {"name": "lgbm"},
            "split": {"method": "kfold", "n_splits": 3},
        }
        cfg = load_config(raw)
        assert cfg.split.method == "kfold"
        assert cfg.split.n_splits == 3

    def test_explicit_stratified_preserved(self) -> None:
        raw = {
            "config_version": 1,
            "task": "binary",
            "data": {"target": "y"},
            "model": {"name": "lgbm"},
            "split": {"method": "stratified_kfold", "n_splits": 3},
        }
        cfg = load_config(raw)
        assert cfg.split.method == "stratified_kfold"
        assert cfg.split.n_splits == 3


class TestKFoldWarning:
    """Test that kfold warning fires for classification tasks."""

    def test_binary_kfold_warns(self) -> None:
        raw = {
            "config_version": 1,
            "task": "binary",
            "data": {"target": "y"},
            "model": {"name": "lgbm"},
            "split": {"method": "kfold"},
        }
        cfg = load_config(raw)
        with pytest.warns(UserWarning, match="stratified_kfold"):
            build_splitter(cfg)

    def test_multiclass_kfold_warns(self) -> None:
        raw = {
            "config_version": 1,
            "task": "multiclass",
            "data": {"target": "y"},
            "model": {"name": "lgbm"},
            "split": {"method": "kfold"},
        }
        cfg = load_config(raw)
        with pytest.warns(UserWarning, match="stratified_kfold"):
            build_splitter(cfg)

    def test_regression_kfold_no_warn(self) -> None:
        raw = {
            "config_version": 1,
            "task": "regression",
            "data": {"target": "y"},
            "model": {"name": "lgbm"},
            "split": {"method": "kfold"},
        }
        cfg = load_config(raw)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            build_splitter(cfg)  # Should not raise

    def test_stratified_kfold_no_warn(self) -> None:
        raw = {
            "config_version": 1,
            "task": "binary",
            "data": {"target": "y"},
            "model": {"name": "lgbm"},
            "split": {"method": "stratified_kfold"},
        }
        cfg = load_config(raw)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            build_splitter(cfg)  # Should not raise
