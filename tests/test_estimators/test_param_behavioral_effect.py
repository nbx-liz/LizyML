"""Tests for H-0063 — Config parameter propagation AND behavioral effect.

Two levels of verification:
A. **Propagation**: value reaches Booster params dict
B. **Behavioral effect**: different values produce different model outputs

Uses the 2-value comparison pattern: fit with val_a vs val_b, verify
predictions differ (proving the parameter actually affects training).
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from lizyml.estimators.lgbm import LGBMAdapter
from lizyml.estimators.lgbm.defaults import _TASK_METRIC
from lizyml.estimators.lgbm.param_names import LGBM_PARAM_NAMES

# ---------------------------------------------------------------------------
# Test data helpers (minimal, fast)
# ---------------------------------------------------------------------------


def _regression_data(
    seed: int = 42,
    n: int = 200,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X_train = pd.DataFrame({"f1": rng.standard_normal(n), "f2": rng.standard_normal(n)})
    y_train = pd.Series(X_train["f1"] * 2 + rng.normal(0, 0.1, n))
    X_valid = pd.DataFrame(
        {"f1": rng.standard_normal(50), "f2": rng.standard_normal(50)}
    )
    y_valid = pd.Series(X_valid["f1"] * 2 + rng.normal(0, 0.1, 50))
    return X_train, y_train, X_valid, y_valid


def _binary_data(
    seed: int = 42,
    n: int = 200,
    imbalance_ratio: float = 0.5,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X_train = pd.DataFrame({"f1": rng.standard_normal(n), "f2": rng.standard_normal(n)})
    y_train = pd.Series((rng.random(n) < imbalance_ratio).astype(int))
    X_valid = pd.DataFrame(
        {"f1": rng.standard_normal(50), "f2": rng.standard_normal(50)}
    )
    y_valid = pd.Series((rng.random(50) < imbalance_ratio).astype(int))
    return X_train, y_train, X_valid, y_valid


def _fit_and_predict(
    task: str,
    params: dict[str, Any],
    *,
    seed: int = 42,
    early_stopping_rounds: int = 5,
    num_class: int | None = None,
    sample_weight: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """Fit adapter and return predictions on validation set."""
    if task == "binary":
        X_train, y_train, X_valid, y_valid = _binary_data(seed=seed)
    else:
        X_train, y_train, X_valid, y_valid = _regression_data(seed=seed)

    adapter = LGBMAdapter(
        task=task,  # type: ignore[arg-type]
        params={"n_estimators": 30, **params},
        num_class=num_class,
        early_stopping_rounds=early_stopping_rounds,
        random_state=seed,
    )
    kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        kwargs["sample_weight"] = sample_weight
    adapter.fit(X_train, y_train, X_valid, y_valid, **kwargs)
    return adapter.predict(X_valid)


# ---------------------------------------------------------------------------
# A. Propagation tests — value reaches Booster params
# ---------------------------------------------------------------------------


class TestBoosterParamPropagation:
    """Verify Config params reach the LightGBM Booster params dict."""

    @pytest.mark.parametrize(
        "param_name,param_value",
        [
            ("bagging_fraction", 0.5),
            ("bagging_freq", 5),
            ("lambda_l1", 1.0),
            ("lambda_l2", 1.0),
            ("max_bin", 127),
            ("boosting", "gbdt"),
            ("first_metric_only", True),
            ("path_smooth", 0.5),  # arbitrary user param passthrough
        ],
    )
    def test_param_reaches_booster(self, param_name: str, param_value: Any) -> None:
        """The value propagates, and the name is one LightGBM actually defines.

        Two assertions, because propagation alone is not the interesting claim.

        ``Booster.params`` is an echo of the dict handed to ``lgb.train``, not a
        report of what LightGBM parsed: an invented key is retained there
        verbatim (measured -- ``not_a_lightgbm_parameter`` comes back as ``9``),
        while a parameter LightGBM defaulted is absent. So asserting only that
        the value arrives would hold for a name LightGBM silently discards,
        which is precisely the defect H-0093 is about.

        The second assertion is what closes that: the names in the list above
        are hand-written, and each is checked against LightGBM's own registry.
        Whether the parameter *changes the model* is a separate question, and
        the behavioural section below is where it is asked.
        """
        assert param_name in LGBM_PARAM_NAMES, (
            f"{param_name!r} is not a name LightGBM {lgb.__version__} defines, "
            "so asserting that it propagates asserts nothing -- LightGBM would "
            "discard it without error."
        )

        X_train, y_train, X_valid, y_valid = _regression_data()
        adapter = LGBMAdapter(
            task="regression",
            params={"n_estimators": 10, param_name: param_value},
            random_state=42,
        )
        adapter.fit(X_train, y_train, X_valid, y_valid)
        booster = adapter._model
        assert booster is not None, "adapter.fit did not produce a Booster"
        assert booster.params.get(param_name) == param_value, (
            f"{param_name}={param_value!r} did not reach the trained Booster; "
            f"it holds {booster.params.get(param_name)!r}."
        )

    def test_objective_locked_to_task(self) -> None:
        # H-0079: cross-task objective injection now raises CONFIG_INVALID
        # (was silently stripped pre-H-0079 — same defensive intent, but
        # explicit failure instead of silent override).
        from lizyml.core.exceptions import LizyMLError

        adapter = LGBMAdapter(task="binary", params={"objective": "regression"})
        with pytest.raises(LizyMLError) as excinfo:
            adapter._build_params()
        assert excinfo.value.code.name == "CONFIG_INVALID"

    def test_verbosity_forced_negative(self) -> None:
        adapter = LGBMAdapter(task="regression")
        params, *_ = adapter._build_params()
        assert params["verbosity"] == -1

    def test_num_class_injected_for_multiclass(self) -> None:
        adapter = LGBMAdapter(task="multiclass", num_class=5)
        params, *_ = adapter._build_params()
        assert params["num_class"] == 5

    def test_num_class_absent_for_regression(self) -> None:
        adapter = LGBMAdapter(task="regression")
        params, *_ = adapter._build_params()
        assert "num_class" not in params

    def test_metric_default_per_task(self) -> None:
        for task in ("regression", "binary"):
            kwargs: dict[str, Any] = {"task": task}
            adapter = LGBMAdapter(**kwargs)
            params, *_ = adapter._build_params()
            assert params["metric"] == _TASK_METRIC[task]

    def test_user_metric_reaches_booster(self) -> None:
        adapter = LGBMAdapter(task="binary", params={"metric": ["auc"]})
        params, *_ = adapter._build_params()
        assert params["metric"] == ["auc"]

    def test_arbitrary_param_passthrough(self) -> None:
        """Params not in defaults should pass through untouched."""
        adapter = LGBMAdapter(
            task="regression",
            params={"extra_trees": True, "linear_tree": True},
        )
        params, *_ = adapter._build_params()
        assert params["extra_trees"] is True
        assert params["linear_tree"] is True


# ---------------------------------------------------------------------------
# B. Behavioral effect tests — different values produce different predictions
# ---------------------------------------------------------------------------


class TestBehavioralEffect:
    """Different param values must produce different predictions.

    Uses parametrized 2-value comparison pattern.
    """

    @pytest.mark.parametrize(
        "param_name,val_a,val_b",
        [
            ("learning_rate", 0.01, 0.5),
            ("max_depth", 2, 8),
            ("n_estimators", 5, 50),
            ("max_bin", 15, 511),
            ("lambda_l1", 0.0, 50.0),
            ("lambda_l2", 0.0, 50.0),
            ("feature_fraction", 0.3, 1.0),
            ("num_leaves", 4, 64),
            ("min_data_in_leaf", 1, 50),
        ],
    )
    def test_regression_param_effect(
        self, param_name: str, val_a: Any, val_b: Any
    ) -> None:
        pred_a = _fit_and_predict("regression", {param_name: val_a})
        pred_b = _fit_and_predict("regression", {param_name: val_b})
        assert not np.allclose(pred_a, pred_b, atol=1e-6), (
            f"{param_name}={val_a} vs {val_b} produced identical predictions"
        )

    def test_bagging_fraction_with_freq_effect(self) -> None:
        pred_a = _fit_and_predict(
            "regression", {"bagging_fraction": 0.5, "bagging_freq": 1}
        )
        pred_b = _fit_and_predict(
            "regression", {"bagging_fraction": 1.0, "bagging_freq": 0}
        )
        assert not np.allclose(pred_a, pred_b, atol=1e-6)

    def test_boosting_type_effect(self) -> None:
        pred_gbdt = _fit_and_predict(
            "regression",
            {"boosting": "gbdt", "bagging_fraction": 0.7, "bagging_freq": 1},
        )
        pred_rf = _fit_and_predict(
            "regression",
            {"boosting": "rf", "bagging_fraction": 0.7, "bagging_freq": 1},
        )
        assert not np.allclose(pred_gbdt, pred_rf, atol=1e-6)

    def test_metric_changes_eval_history(self) -> None:
        """Different metric configs produce different eval history keys."""
        X_train, y_train, X_valid, y_valid = _binary_data()

        adapter_a = LGBMAdapter(
            task="binary",
            params={"metric": ["auc"], "n_estimators": 10},
            early_stopping_rounds=5,
        )
        adapter_a.fit(X_train, y_train, X_valid, y_valid)
        keys_a = set(adapter_a.eval_results.get("valid_0", {}).keys())

        adapter_b = LGBMAdapter(
            task="binary",
            params={"metric": ["binary_logloss"], "n_estimators": 10},
            early_stopping_rounds=5,
        )
        adapter_b.fit(X_train, y_train, X_valid, y_valid)
        keys_b = set(adapter_b.eval_results.get("valid_0", {}).keys())

        assert keys_a != keys_b
        assert "auc" in keys_a
        assert "binary_logloss" in keys_b

    def test_scale_pos_weight_changes_predictions(self) -> None:
        """scale_pos_weight should shift binary prediction distribution."""
        X_train, y_train, X_valid, y_valid = _binary_data(imbalance_ratio=0.2)

        adapter_a = LGBMAdapter(
            task="binary",
            params={"n_estimators": 30, "scale_pos_weight": 1.0},
            early_stopping_rounds=5,
        )
        adapter_a.fit(X_train, y_train, X_valid, y_valid)
        proba_a = adapter_a.predict_proba(X_valid)

        adapter_b = LGBMAdapter(
            task="binary",
            params={"n_estimators": 30, "scale_pos_weight": 10.0},
            early_stopping_rounds=5,
        )
        adapter_b.fit(X_train, y_train, X_valid, y_valid)
        proba_b = adapter_b.predict_proba(X_valid)

        assert not np.allclose(proba_a, proba_b, atol=1e-4)
        # Higher weight on positive class should shift predictions upward
        assert proba_b[:, 1].mean() > proba_a[:, 1].mean()


# ---------------------------------------------------------------------------
# C. Smart params behavioral effect
# ---------------------------------------------------------------------------


class TestSmartParamsBehavior:
    """Smart params must produce observable behavioral changes."""

    def test_balanced_binary_shifts_predictions(self) -> None:
        """balanced=True should apply scale_pos_weight on imbalanced data."""
        from lizyml.estimators.lgbm.smart_params import resolve_smart_params

        X_train, y_train, _, _ = _binary_data(imbalance_ratio=0.1)

        # balanced=True
        smart_true = {
            "auto_num_leaves": True,
            "num_leaves_ratio": 1.0,
            "min_data_in_leaf_ratio": None,
            "min_data_in_bin_ratio": None,
            "feature_weights": None,
            "balanced": True,
        }
        resolved_true, _ = resolve_smart_params(
            smart=smart_true,
            effective_params={"max_depth": 5},
            n_rows=len(X_train),
            feature_names=list(X_train.columns),
            y=y_train,
            task="binary",
        )

        # balanced=False
        smart_false = {**smart_true, "balanced": False}
        resolved_false, _ = resolve_smart_params(
            smart=smart_false,
            effective_params={"max_depth": 5},
            n_rows=len(X_train),
            feature_names=list(X_train.columns),
            y=y_train,
            task="binary",
        )

        # balanced=True should set scale_pos_weight based on class ratio
        assert "scale_pos_weight" in resolved_true
        assert resolved_true["scale_pos_weight"] > 1.0  # minority positive class
        assert "scale_pos_weight" not in resolved_false

    def test_feature_weights_changes_importance(self) -> None:
        """feature_weights must change what the booster learns (BLUEPRINT 14.4).

        Rewritten for H-0093. The previous version compared the two *resolved
        parameter dicts* and never trained anything, so it held whether or not
        LightGBM honoured the key -- and LightGBM did not, because the emitted
        name was ``feature_weights`` rather than ``feature_contri``. The
        invariant BLUEPRINT declares is about importance ordering, so that is
        what is asserted: suppress the informative feature and it must stop
        leading the importance ranking.
        """
        from lizyml.estimators.lgbm.smart_params import resolve_smart_params

        X_train, y_train, X_valid, y_valid = _regression_data()
        feature_names = list(X_train.columns)
        base_smart: dict[str, Any] = {
            "auto_num_leaves": None,
            "num_leaves_ratio": None,
            "min_data_in_leaf_ratio": None,
            "min_data_in_bin_ratio": None,
            "feature_weights": None,
            "balanced": None,
        }

        def gains(weights: dict[str, float] | None) -> dict[str, float]:
            resolved, _ = resolve_smart_params(
                smart={**base_smart, "feature_weights": weights},
                effective_params={},
                n_rows=len(X_train),
                feature_names=feature_names,
                y=y_train,
                task="regression",
            )
            adapter = LGBMAdapter(
                task="regression",
                params={"n_estimators": 30, **resolved},
                random_state=42,
            )
            adapter.fit(X_train, y_train, X_valid, y_valid)
            booster = adapter._model
            return dict(
                zip(
                    booster.feature_name(),
                    booster.feature_importance("gain"),
                    strict=True,
                )
            )

        # f1 carries the signal (y = 2*f1 + noise), so it leads unweighted.
        unweighted = gains(None)
        assert max(unweighted, key=lambda k: unweighted[k]) == "f1"

        # Suppressing f1 must dislodge it. Under the old emitted key this was
        # byte-identical to `unweighted`.
        suppressed = gains({"f1": 0.0001, "f2": 1.0})
        assert suppressed["f1"] < unweighted["f1"], (
            "suppressing f1 did not reduce its gain, so the weights never "
            f"reached the booster: unweighted={unweighted}, "
            f"suppressed={suppressed}"
        )
        assert max(suppressed, key=lambda k: suppressed[k]) == "f2", (
            "f1 still leads the importance ranking after being suppressed; "
            f"got {suppressed}"
        )

    def test_feature_weights_resolves_to_an_ordered_list(self) -> None:
        """The dict is positional over the training feature order."""
        from lizyml.estimators.lgbm.smart_params import resolve_smart_params

        X_train, y_train, X_valid, y_valid = _regression_data()
        feature_names = list(X_train.columns)

        # Resolve with f1 heavily weighted
        smart_a = {
            "auto_num_leaves": True,
            "num_leaves_ratio": 1.0,
            "min_data_in_leaf_ratio": None,
            "min_data_in_bin_ratio": None,
            "feature_weights": {"f1": 10.0, "f2": 0.01},
            "balanced": None,
        }
        resolved_a, _ = resolve_smart_params(
            smart=smart_a,
            effective_params={"max_depth": 5},
            n_rows=len(X_train),
            feature_names=feature_names,
            y=y_train,
            task="regression",
        )

        # Resolve with f2 heavily weighted
        smart_b = {**smart_a, "feature_weights": {"f1": 0.01, "f2": 10.0}}
        resolved_b, _ = resolve_smart_params(
            smart=smart_b,
            effective_params={"max_depth": 5},
            n_rows=len(X_train),
            feature_names=feature_names,
            y=y_train,
            task="regression",
        )

        # Emitted under LightGBM's own name (H-0093), positional over the
        # training feature order.
        fw_a = resolved_a["feature_contri"]
        fw_b = resolved_b["feature_contri"]
        assert "feature_weights" not in resolved_a
        assert fw_a != fw_b
        assert isinstance(fw_a, list)
        assert fw_a[0] > fw_a[1]  # f1 heavier
        assert fw_b[1] > fw_b[0]  # f2 heavier

    def test_min_data_in_leaf_ratio_scales_with_nrows(self) -> None:
        """min_data_in_leaf_ratio * n_rows should produce proportional leaf size."""
        from lizyml.estimators.lgbm.smart_params import resolve_ratio_params

        result_small = resolve_ratio_params(0.05, None, 100)
        result_large = resolve_ratio_params(0.05, None, 1000)

        assert result_small["min_data_in_leaf"] == 5
        assert result_large["min_data_in_leaf"] == 50

    def test_min_data_in_bin_ratio_scales_with_nrows(self) -> None:
        """min_data_in_bin_ratio * n_rows should produce proportional bin size."""
        from lizyml.estimators.lgbm.smart_params import resolve_ratio_params

        result_small = resolve_ratio_params(None, 0.01, 200)
        result_large = resolve_ratio_params(None, 0.01, 2000)

        assert result_small["min_data_in_bin"] == 2
        assert result_large["min_data_in_bin"] == 20


# ---------------------------------------------------------------------------
# D. Training / Feature config behavioral effect
# ---------------------------------------------------------------------------


class TestTrainingConfigBehavior:
    """Training config params must produce observable effects."""

    def test_early_stopping_random_state_deterministic(self) -> None:
        """Same random_state should produce same inner valid split."""
        from lizyml import Model
        from tests._helpers import make_config, make_regression_df

        df = make_regression_df(n=200)
        cfg1 = make_config("regression", seed=42)
        cfg2 = make_config("regression", seed=42)

        r1 = Model(cfg1).fit(data=df)
        r2 = Model(cfg2).fit(data=df)

        # Same seed → same predictions (implies same inner split)
        np.testing.assert_array_almost_equal(r1.oof_pred, r2.oof_pred)

    def test_different_seed_different_results(self) -> None:
        """Different seeds should produce different results."""
        from lizyml import Model
        from tests._helpers import make_config, make_regression_df

        df = make_regression_df(n=200)
        r1 = Model(make_config("regression", seed=42)).fit(data=df)
        r2 = Model(make_config("regression", seed=99)).fit(data=df)

        assert not np.allclose(r1.oof_pred, r2.oof_pred, atol=1e-6)

    def test_verbosity_suppresses_output(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """LightGBM should not print to stdout with verbosity=-1."""
        X_train, y_train, X_valid, y_valid = _regression_data()
        adapter = LGBMAdapter(
            task="regression",
            params={"n_estimators": 10},
            early_stopping_rounds=5,
        )
        adapter.fit(X_train, y_train, X_valid, y_valid)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_n_estimators_controls_iterations(self) -> None:
        """n_estimators should control the max number of boosting rounds."""
        X_train, y_train, X_valid, y_valid = _regression_data()

        adapter = LGBMAdapter(
            task="regression",
            params={"n_estimators": 5},
            early_stopping_rounds=None,
        )
        adapter.fit(X_train, y_train, X_valid, y_valid)
        booster = adapter.get_native_model()
        assert booster.current_iteration() == 5


class TestFeatureConfigBehavior:
    """Feature config params must produce observable effects."""

    def test_auto_categorical_detects_string_columns(self) -> None:
        """auto_categorical should detect string/category columns."""
        from lizyml import Model
        from tests._helpers import make_config

        rng = np.random.default_rng(42)
        n = 100
        df = pd.DataFrame(
            {
                "feat_num": rng.standard_normal(n),
                "feat_cat": pd.Categorical(rng.choice(["a", "b", "c"], n)),
                "target": rng.standard_normal(n),
            }
        )
        cfg = make_config("regression")
        m = Model(cfg)
        result = m.fit(data=df)
        # Should complete without error with auto-categorical
        assert result is not None
        assert len(result.oof_pred) == n

    def test_exclude_removes_columns_from_training(self) -> None:
        """Excluded columns should not appear in feature names."""
        from lizyml import Model
        from tests._helpers import make_config

        rng = np.random.default_rng(42)
        n = 100
        df = pd.DataFrame(
            {
                "feat_a": rng.standard_normal(n),
                "feat_b": rng.standard_normal(n),
                "feat_c": rng.standard_normal(n),
                "target": rng.standard_normal(n),
            }
        )
        cfg = make_config("regression")
        cfg["features"] = {"exclude": ["feat_c"]}
        m = Model(cfg)
        result = m.fit(data=df)

        # feat_c should not be in the model's features
        feature_names = result.models[0].get_native_model().feature_name()
        assert "feat_c" not in feature_names
        assert "feat_a" in feature_names
        assert "feat_b" in feature_names


# ---------------------------------------------------------------------------
# E. Multiclass specific
# ---------------------------------------------------------------------------


class TestMulticlassParams:
    """Multiclass-specific parameter propagation and behavior."""

    def test_num_class_in_booster_params(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        X_train = pd.DataFrame(
            {"f1": rng.standard_normal(n), "f2": rng.standard_normal(n)}
        )
        y_train = pd.Series(rng.integers(0, 3, n))
        X_valid = pd.DataFrame(
            {"f1": rng.standard_normal(50), "f2": rng.standard_normal(50)}
        )
        y_valid = pd.Series(rng.integers(0, 3, 50))

        adapter = LGBMAdapter(
            task="multiclass",
            params={"n_estimators": 10},
            num_class=3,
            early_stopping_rounds=5,
        )
        adapter.fit(X_train, y_train, X_valid, y_valid)
        booster = adapter.get_native_model()
        assert int(booster.params.get("num_class", 0)) == 3

    def test_multiclass_predict_proba_shape(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        X_train = pd.DataFrame(
            {"f1": rng.standard_normal(n), "f2": rng.standard_normal(n)}
        )
        y_train = pd.Series(rng.integers(0, 4, n))
        X_valid = pd.DataFrame(
            {"f1": rng.standard_normal(50), "f2": rng.standard_normal(50)}
        )
        y_valid = pd.Series(rng.integers(0, 4, 50))

        adapter = LGBMAdapter(
            task="multiclass",
            params={"n_estimators": 10},
            num_class=4,
            early_stopping_rounds=5,
        )
        adapter.fit(X_train, y_train, X_valid, y_valid)
        proba = adapter.predict_proba(X_valid)
        assert proba.shape == (50, 4)
        # Rows should sum to 1
        np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0)
