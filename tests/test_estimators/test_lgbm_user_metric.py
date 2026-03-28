"""Tests for H-0061 — user-specified metric in _build_params() and params_summary().

Covers:
- User metric overrides task default
- Fallback to task default when no user metric
- String metric normalised to list
- Invalid metric produces LizyMLError with context
- params_summary() includes metric
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.estimators.lgbm import LGBMAdapter
from lizyml.estimators.lgbm.defaults import _TASK_METRIC
from lizyml.estimators.lgbm.provider import LGBMProvider

# ---------------------------------------------------------------------------
# _build_params: user metric
# ---------------------------------------------------------------------------


class TestUserMetricOverride:
    """User-specified metric should reach Booster params."""

    def test_user_metric_list_overrides_default(self) -> None:
        adapter = LGBMAdapter(task="binary", params={"metric": ["auc"]})
        params, _ = adapter._build_params()
        assert params["metric"] == ["auc"]

    def test_user_metric_string_normalised_to_list(self) -> None:
        adapter = LGBMAdapter(task="binary", params={"metric": "auc"})
        params, _ = adapter._build_params()
        assert params["metric"] == ["auc"]

    def test_fallback_to_task_default_when_no_user_metric(self) -> None:
        adapter = LGBMAdapter(task="binary")
        params, _ = adapter._build_params()
        assert params["metric"] == _TASK_METRIC["binary"]

    def test_fallback_when_empty_list(self) -> None:
        adapter = LGBMAdapter(task="binary", params={"metric": []})
        params, _ = adapter._build_params()
        assert params["metric"] == _TASK_METRIC["binary"]

    @pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
    def test_each_task_fallback(self, task: str) -> None:
        kwargs: dict = {"task": task}
        if task == "multiclass":
            kwargs["num_class"] = 3
        adapter = LGBMAdapter(**kwargs)
        params, _ = adapter._build_params()
        assert params["metric"] == _TASK_METRIC[task]


class TestUserMetricBehavioral:
    """User-specified metric actually changes eval history keys during training."""

    def test_user_metric_appears_in_eval_history(self) -> None:
        """Train with metric=['auc'] and verify eval_history contains only auc."""
        rng = np.random.default_rng(42)
        n = 200
        X_train = pd.DataFrame(
            {"f1": rng.standard_normal(n), "f2": rng.standard_normal(n)}
        )
        y_train = pd.Series((rng.standard_normal(n) > 0).astype(int))
        X_valid = pd.DataFrame(
            {"f1": rng.standard_normal(50), "f2": rng.standard_normal(50)}
        )
        y_valid = pd.Series((rng.standard_normal(50) > 0).astype(int))

        adapter = LGBMAdapter(
            task="binary",
            params={"metric": ["auc"], "n_estimators": 10},
            early_stopping_rounds=5,
        )
        adapter.fit(X_train, y_train, X_valid, y_valid)

        history = adapter.eval_results
        # With metric=["auc"], eval_history should contain auc but not binary_logloss
        valid_keys = set(history.get("valid_0", {}).keys())
        assert "auc" in valid_keys
        assert "binary_logloss" not in valid_keys

    def test_default_metric_shows_all_task_metrics(self) -> None:
        """Default binary metric should produce both auc and binary_logloss."""
        rng = np.random.default_rng(42)
        n = 200
        X_train = pd.DataFrame(
            {"f1": rng.standard_normal(n), "f2": rng.standard_normal(n)}
        )
        y_train = pd.Series((rng.standard_normal(n) > 0).astype(int))
        X_valid = pd.DataFrame(
            {"f1": rng.standard_normal(50), "f2": rng.standard_normal(50)}
        )
        y_valid = pd.Series((rng.standard_normal(50) > 0).astype(int))

        adapter = LGBMAdapter(
            task="binary",
            params={"n_estimators": 10},
            early_stopping_rounds=5,
        )
        adapter.fit(X_train, y_train, X_valid, y_valid)

        history = adapter.eval_results
        valid_keys = set(history.get("valid_0", {}).keys())
        assert "auc" in valid_keys
        assert "binary_logloss" in valid_keys


class TestInvalidMetricError:
    """Invalid metric should produce LizyMLError with descriptive context."""

    def test_invalid_metric_raises_with_context(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        X_train = pd.DataFrame({"f1": rng.standard_normal(n)})
        y_train = pd.Series((rng.standard_normal(n) > 0).astype(int))
        X_valid = pd.DataFrame({"f1": rng.standard_normal(20)})
        y_valid = pd.Series((rng.standard_normal(20) > 0).astype(int))

        adapter = LGBMAdapter(
            task="binary",
            params={"metric": ["totally_invalid_metric_xyz"], "n_estimators": 5},
            early_stopping_rounds=3,
        )
        with pytest.raises(LizyMLError) as exc_info:
            adapter.fit(X_train, y_train, X_valid, y_valid)

        err = exc_info.value
        assert err.code == ErrorCode.CONFIG_INVALID
        assert "metric" in err.context
        assert "totally_invalid_metric_xyz" in str(err.context["metric"])

    def test_invalid_metric_without_early_stopping_warns(self) -> None:
        """Invalid metric without early stopping should emit a warning."""
        rng = np.random.default_rng(42)
        n = 100
        X_train = pd.DataFrame({"f1": rng.standard_normal(n)})
        y_train = pd.Series((rng.standard_normal(n) > 0).astype(int))
        X_valid = pd.DataFrame({"f1": rng.standard_normal(20)})
        y_valid = pd.Series((rng.standard_normal(20) > 0).astype(int))

        adapter = LGBMAdapter(
            task="binary",
            params={
                "metric": ["totally_invalid_metric_xyz"],
                "n_estimators": 5,
            },
            early_stopping_rounds=None,
        )
        with pytest.warns(UserWarning, match="no eval results"):
            adapter.fit(X_train, y_train, X_valid, y_valid)

    def test_empty_string_metric_falls_back(self) -> None:
        """metric=[''] should be filtered and fall back to task default."""
        adapter = LGBMAdapter(
            task="binary",
            params={"metric": [""]},
        )
        params, _ = adapter._build_params()
        from lizyml.estimators.lgbm.defaults import _TASK_METRIC

        assert params["metric"] == _TASK_METRIC["binary"]

    def test_string_metric_with_empty_falls_back(self) -> None:
        """metric='' should be treated as empty and fall back."""
        adapter = LGBMAdapter(
            task="binary",
            params={"metric": ""},
        )
        params, _ = adapter._build_params()
        from lizyml.estimators.lgbm.defaults import _TASK_METRIC

        assert params["metric"] == _TASK_METRIC["binary"]


# ---------------------------------------------------------------------------
# params_summary: metric included
# ---------------------------------------------------------------------------


class TestParamsSummaryMetric:
    """params_summary() should include metric in output."""

    def test_metric_in_params_summary(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        X_train = pd.DataFrame(
            {"f1": rng.standard_normal(n), "f2": rng.standard_normal(n)}
        )
        y_train = pd.Series((rng.standard_normal(n) > 0).astype(int))
        X_valid = pd.DataFrame(
            {"f1": rng.standard_normal(30), "f2": rng.standard_normal(30)}
        )
        y_valid = pd.Series((rng.standard_normal(30) > 0).astype(int))

        adapter = LGBMAdapter(
            task="binary",
            params={"n_estimators": 10},
            early_stopping_rounds=5,
        )
        adapter.fit(X_train, y_train, X_valid, y_valid)

        provider = LGBMProvider()
        # Create a minimal model_cfg mock with required smart param attributes
        from types import SimpleNamespace

        model_cfg = SimpleNamespace(
            params={},
            auto_num_leaves=True,
            num_leaves_ratio=1.0,
            min_data_in_leaf_ratio=0.01,
            min_data_in_bin_ratio=0.01,
            feature_weights=None,
            balanced=None,
        )

        rows = provider.params_summary(adapter, model_cfg)
        param_names = [r["parameter"] for r in rows]
        assert "metric" in param_names

        # Find metric value
        metric_row = next(r for r in rows if r["parameter"] == "metric")
        assert isinstance(metric_row["value"], (str, list))
