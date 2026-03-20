"""End-to-end tests for blocked_group_kfold (H-0060).

Tests Model.fit() and Model.tune() with blocked_group_kfold split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lizyml import Model

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def _make_df(
    n_users: int = 20,
    n_months: int = 4,
    task: str = "binary",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic DataFrame with date and user_id columns."""
    rng = np.random.RandomState(seed)
    rows = []
    for month in range(1, n_months + 1):
        for user in range(n_users):
            if rng.random() < 0.8:  # 80% presence
                if task == "binary":
                    target = rng.randint(0, 2)
                elif task == "multiclass":
                    target = rng.randint(0, 3)
                else:
                    target = rng.randn()
                rows.append(
                    {
                        "date": f"2025-{month:02d}",
                        "user_id": f"user_{user}",
                        "feat_a": rng.randn(),
                        "feat_b": rng.randn(),
                        "target": target,
                    }
                )
    return pd.DataFrame(rows)


def _make_config(
    task: str = "binary",
    cutoffs: list[str] | None = None,
    mode: str = "expanding",
    n_splits: int = 2,
) -> dict:
    """Create a minimal config dict for blocked_group_kfold."""
    return {
        "config_version": 1,
        "task": task,
        "data": {"target": "target"},
        "split": {
            "method": "blocked_group_kfold",
            "blocks": {
                "col": "date",
                "cutoffs": cutoffs or ["2025-03"],
                "mode": mode,
            },
            "groups": {
                "col": "user_id",
                "n_splits": n_splits,
            },
            "min_train_rows": 3,
            "min_valid_rows": 2,
        },
        "model": {"name": "lgbm"},
        "training": {"seed": 42},
    }


# ---------------------------------------------------------------------------
# E2E: fit
# ---------------------------------------------------------------------------


class TestBlockedGroupKFoldFit:
    """Model.fit() with blocked_group_kfold."""

    def test_binary_fit_succeeds(self) -> None:
        df = _make_df(task="binary")
        model = Model(config=_make_config(task="binary"))
        result = model.fit(df)
        assert result is not None
        assert result.oof_pred is not None
        assert len(result.models) > 0

    def test_regression_fit_succeeds(self) -> None:
        df = _make_df(task="regression")
        model = Model(config=_make_config(task="regression"))
        result = model.fit(df)
        assert result is not None

    def test_multiclass_fit_succeeds(self) -> None:
        df = _make_df(task="multiclass")
        model = Model(config=_make_config(task="multiclass", n_splits=2))
        result = model.fit(df)
        assert result is not None

    def test_no_user_leak_in_splits(self) -> None:
        """Verify group isolation in stored splits."""
        df = _make_df(task="binary", n_users=15)
        model = Model(config=_make_config(task="binary"))
        result = model.fit(df)
        groups = df.sort_values("date")["user_id"].to_numpy()
        for train_idx, valid_idx in result.splits.outer:
            train_users = set(groups[train_idx])
            valid_users = set(groups[valid_idx])
            assert train_users.isdisjoint(valid_users), (
                f"User leak: {train_users & valid_users}"
            )

    def test_sliding_mode(self) -> None:
        df = _make_df(task="binary", n_months=5)
        config = _make_config(task="binary", cutoffs=["2025-03", "2025-04"])
        config["split"]["blocks"]["mode"] = "sliding"
        config["split"]["blocks"]["train_window"] = 2
        model = Model(config=config)
        result = model.fit(df)
        assert result is not None
        assert len(result.models) > 0


# ---------------------------------------------------------------------------
# E2E: evaluate
# ---------------------------------------------------------------------------


class TestBlockedGroupKFoldEvaluate:
    """Model.evaluate() after fit with blocked_group_kfold."""

    def test_evaluate_returns_metrics(self) -> None:
        df = _make_df(task="binary")
        model = Model(config=_make_config(task="binary"))
        model.fit(df)
        metrics = model.evaluate()
        assert "raw" in metrics
        assert "oof" in metrics["raw"]
