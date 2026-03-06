"""Tests for T-1 — Split settings integration.

Verifies that Model.fit() correctly propagates split.method through
CVTrainer and records the right fold structure in FitResult.splits.outer.

Tests cover three non-kfold splitters:
- stratified_kfold: target class balance is preserved per fold
- group_kfold: no group leaks across train/valid within any fold
- time_series: temporal ordering (train_max < valid_min) is enforced
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lizyml import Model
from lizyml.core.types.fit_result import FitResult

# ---------------------------------------------------------------------------
# Synthetic datasets
# ---------------------------------------------------------------------------

N_SPLITS = 3


def _binary_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Imbalanced binary dataset: 70% class-0, 30% class-1."""
    rng = np.random.default_rng(seed)
    feat_a = rng.uniform(0, 10, n)
    feat_b = rng.uniform(-1, 1, n)
    # Deliberately imbalanced: positive only where feat_a > 7
    target = (feat_a > 7).astype(int)
    return pd.DataFrame({"feat_a": feat_a, "feat_b": feat_b, "target": target})


def _grouped_df(n: int = 300, n_groups: int = 15, seed: int = 1) -> pd.DataFrame:
    """Regression dataset with an explicit group column.

    Groups are non-overlapping ranges of rows so the group identity is clear.
    """
    rng = np.random.default_rng(seed)
    feat_a = rng.uniform(0, 10, n)
    feat_b = rng.uniform(-1, 1, n)
    target = feat_a * 2.0 + feat_b + rng.normal(0, 0.1, n)
    # Each group covers n // n_groups consecutive rows
    groups = np.repeat(np.arange(n_groups), n // n_groups)[:n]
    return pd.DataFrame(
        {"feat_a": feat_a, "feat_b": feat_b, "group": groups, "target": target}
    )


def _timeseries_df(n: int = 300, seed: int = 2) -> pd.DataFrame:
    """Regression dataset where row order implies time (index = time)."""
    rng = np.random.default_rng(seed)
    feat_a = np.linspace(0, 10, n) + rng.normal(0, 0.1, n)
    feat_b = rng.uniform(-1, 1, n)
    target = feat_a * 2.0 + feat_b + rng.normal(0, 0.1, n)
    return pd.DataFrame(
        {"feat_a": feat_a, "feat_b": feat_b, "time_idx": range(n), "target": target}
    )


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _stratified_config(n_splits: int = N_SPLITS) -> dict:
    return {
        "config_version": 1,
        "task": "binary",
        "data": {"target": "target"},
        "split": {
            "method": "stratified_kfold",
            "n_splits": n_splits,
            "random_state": 42,
        },
        "model": {"name": "lgbm", "params": {"n_estimators": 10}},
        "training": {"seed": 0},
    }


def _group_kfold_config(n_splits: int = N_SPLITS) -> dict:
    return {
        "config_version": 1,
        "task": "regression",
        "data": {"target": "target", "group_col": "group"},
        "split": {
            "method": "group_kfold",
            "n_splits": n_splits,
        },
        "model": {"name": "lgbm", "params": {"n_estimators": 10}},
        "training": {"seed": 0},
    }


def _time_series_config(n_splits: int = N_SPLITS) -> dict:
    return {
        "config_version": 1,
        "task": "regression",
        "data": {"target": "target", "time_col": "time_idx"},
        "split": {
            "method": "time_series",
            "n_splits": n_splits,
        },
        "model": {"name": "lgbm", "params": {"n_estimators": 10}},
        "training": {"seed": 0},
    }


# ---------------------------------------------------------------------------
# T-1a: stratified_kfold — target balance preserved
# ---------------------------------------------------------------------------


class TestStratifiedKFold:
    def test_fit_returns_fit_result(self) -> None:
        m = Model(_stratified_config())
        result = m.fit(data=_binary_df())
        assert isinstance(result, FitResult)

    def test_n_outer_folds_matches_config(self) -> None:
        m = Model(_stratified_config(n_splits=N_SPLITS))
        result = m.fit(data=_binary_df())
        assert len(result.splits.outer) == N_SPLITS

    def test_target_balance_preserved_per_fold(self) -> None:
        """Stratified folds: valid-set positive rate must be within ±20% of overall."""
        df = _binary_df()
        overall_rate = df["target"].mean()

        m = Model(_stratified_config(n_splits=N_SPLITS))
        result = m.fit(data=df)

        y = df["target"].to_numpy()
        for fold_idx, (_, valid_idx) in enumerate(result.splits.outer):
            fold_rate = y[valid_idx].mean()
            assert abs(fold_rate - overall_rate) < 0.20, (
                f"Fold {fold_idx}: valid positive rate {fold_rate:.3f} deviates "
                f"more than 20% from overall {overall_rate:.3f}"
            )

    def test_all_samples_covered_exactly_once(self) -> None:
        df = _binary_df()
        m = Model(_stratified_config(n_splits=N_SPLITS))
        result = m.fit(data=df)

        all_valid = np.concatenate([v for _, v in result.splits.outer])
        assert sorted(all_valid.tolist()) == list(range(len(df)))


# ---------------------------------------------------------------------------
# T-1b: group_kfold — no group leaks across folds
# ---------------------------------------------------------------------------


class TestGroupKFold:
    def test_fit_returns_fit_result(self) -> None:
        m = Model(_group_kfold_config())
        result = m.fit(data=_grouped_df())
        assert isinstance(result, FitResult)

    def test_n_outer_folds_matches_config(self) -> None:
        m = Model(_group_kfold_config(n_splits=N_SPLITS))
        result = m.fit(data=_grouped_df())
        assert len(result.splits.outer) == N_SPLITS

    def test_no_group_overlap_between_train_and_valid(self) -> None:
        """Groups in the valid fold must not appear in the train fold."""
        df = _grouped_df()
        groups = df["group"].to_numpy()

        m = Model(_group_kfold_config(n_splits=N_SPLITS))
        result = m.fit(data=df)

        for fold_idx, (train_idx, valid_idx) in enumerate(result.splits.outer):
            train_groups = set(groups[train_idx].tolist())
            valid_groups = set(groups[valid_idx].tolist())
            overlap = train_groups & valid_groups
            assert overlap == set(), (
                f"Fold {fold_idx}: groups {overlap} appear in both train and valid"
            )

    def test_all_samples_covered_exactly_once(self) -> None:
        df = _grouped_df()
        m = Model(_group_kfold_config(n_splits=N_SPLITS))
        result = m.fit(data=df)

        all_valid = np.concatenate([v for _, v in result.splits.outer])
        assert sorted(all_valid.tolist()) == list(range(len(df)))


# ---------------------------------------------------------------------------
# T-1c: time_series — train_max < valid_min in every fold
# ---------------------------------------------------------------------------


class TestTimeSeries:
    def test_fit_returns_fit_result(self) -> None:
        m = Model(_time_series_config())
        result = m.fit(data=_timeseries_df())
        assert isinstance(result, FitResult)

    def test_n_outer_folds_matches_config(self) -> None:
        m = Model(_time_series_config(n_splits=N_SPLITS))
        result = m.fit(data=_timeseries_df())
        assert len(result.splits.outer) == N_SPLITS

    def test_temporal_ordering_enforced(self) -> None:
        """All train indices must precede all valid indices in every fold."""
        m = Model(_time_series_config(n_splits=N_SPLITS))
        result = m.fit(data=_timeseries_df())

        for fold_idx, (train_idx, valid_idx) in enumerate(result.splits.outer):
            train_max = int(train_idx.max())
            valid_min = int(valid_idx.min())
            assert train_max < valid_min, (
                f"Fold {fold_idx}: train_max={train_max} >= valid_min={valid_min} "
                "— temporal ordering violated"
            )

    def test_valid_sets_are_non_overlapping(self) -> None:
        """Time series folds must not reuse the same valid samples."""
        m = Model(_time_series_config(n_splits=N_SPLITS))
        result = m.fit(data=_timeseries_df())

        seen: set[int] = set()
        for fold_idx, (_, valid_idx) in enumerate(result.splits.outer):
            fold_set = set(valid_idx.tolist())
            overlap = seen & fold_set
            assert overlap == set(), (
                f"Fold {fold_idx}: valid samples {overlap} were already used "
                "in a previous fold"
            )
            seen |= fold_set
