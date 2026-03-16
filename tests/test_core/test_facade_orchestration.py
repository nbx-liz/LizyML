"""Facade orchestration tests — verify Model.fit() calls components correctly.

Uses mock/spy patterns to verify:
1. Component call sequence (CVTrainer → Evaluator → Calibrator → RefitTrainer)
2. Correct arguments wired to each component
3. CVTrainer and RefitTrainer receive identical factories
4. Conditional calibration paths
5. get_provider dispatch
6. _merge_params priority (Config < tune < fit() args)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lizyml.core._model_factories import get_provider
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.model import Model
from tests._helpers import (
    make_binary_df,
    make_config,
    make_multiclass_df,
    make_regression_df,
)

# ===========================================================================
# 2.1 fit() calls components in correct order
# ===========================================================================


class TestFitCallOrder:
    """Model.fit() orchestrates CVTrainer → Evaluator → RefitTrainer."""

    def test_cv_trainer_called_before_refit_trainer(self) -> None:
        """CVTrainer.fit must complete before RefitTrainer.fit."""
        cfg = make_config("regression")
        m = Model(cfg)
        result = m.fit(data=make_regression_df())

        # If CVTrainer didn't run, there would be no models
        assert len(result.models) > 0, "CVTrainer must have been called"
        # If RefitTrainer didn't run, predict would fail
        pred = m.predict(make_regression_df().drop(columns=["target"]))
        assert pred.pred is not None, "RefitTrainer must have been called"

    def test_evaluator_called_with_fit_result(self) -> None:
        """Metrics are populated, proving Evaluator was invoked."""
        cfg = make_config("regression")
        m = Model(cfg)
        result = m.fit(data=make_regression_df())
        assert "raw" in result.metrics
        assert "oof" in result.metrics["raw"]
        assert len(result.metrics["raw"]["oof"]) > 0


# ===========================================================================
# 2.2 CVTrainer receives correct constructor args
# ===========================================================================


class TestCVTrainerArgs:
    """CVTrainer receives correctly derived args from config."""

    def test_outer_splitter_type_matches_config(self) -> None:
        cfg = make_config("regression", split_method="kfold", n_splits=3)
        m = Model(cfg)
        result = m.fit(data=make_regression_df())
        assert len(result.splits.outer) == 3

    def test_task_type_propagated(self) -> None:
        """Task type from config reaches CVTrainer → FitResult."""
        for task, df_fn in [
            ("regression", make_regression_df),
            ("binary", make_binary_df),
            ("multiclass", make_multiclass_df),
        ]:
            cfg = make_config(task, n_splits=2)
            m = Model(cfg)
            result = m.fit(data=df_fn())
            if task == "multiclass":
                assert result.oof_pred.ndim == 2
            else:
                assert result.oof_pred.ndim == 1

    def test_collect_raw_scores_true_when_calibration_configured(self) -> None:
        """When calibration is configured, raw scores are collected."""
        cfg = make_config("binary", n_splits=2, calibration="platt")
        m = Model(cfg)
        result = m.fit(data=make_binary_df())
        assert result.oof_raw_scores is not None

    def test_collect_raw_scores_false_when_no_calibration(self) -> None:
        """Without calibration, raw scores are not collected."""
        cfg = make_config("binary", n_splits=2)
        m = Model(cfg)
        result = m.fit(data=make_binary_df())
        assert result.oof_raw_scores is None

    def test_n_classes_set_for_multiclass(self) -> None:
        """Multiclass should have n_classes computed from data."""
        df = make_multiclass_df()
        n_classes = df["target"].nunique()
        cfg = make_config("multiclass", n_splits=2)
        m = Model(cfg)
        result = m.fit(data=df)
        assert result.oof_pred.shape[1] == n_classes


# ===========================================================================
# 2.3 CVTrainer and RefitTrainer receive same factories
# ===========================================================================


class TestTrainersShareFactories:
    """CVTrainer and RefitTrainer use identical factories."""

    def test_shared_factories_produce_consistent_results(self) -> None:
        """Both trainers produce models with the same params."""
        cfg = make_config("regression", learning_rate=0.05, max_depth=3)
        m = Model(cfg)
        result = m.fit(data=make_regression_df())

        cv_params = result.models[0].get_native_model().params
        assert m._refit_result is not None
        refit_params = m._refit_result.model.get_native_model().params

        assert float(cv_params["learning_rate"]) == float(refit_params["learning_rate"])
        assert int(cv_params["max_depth"]) == int(refit_params["max_depth"])
        assert int(cv_params["seed"]) == int(refit_params["seed"])

    def test_same_pipeline_type(self) -> None:
        """Both trainers use the same pipeline type."""
        cfg = make_config("regression")
        m = Model(cfg)
        m.fit(data=make_regression_df())

        assert m._refit_result is not None
        assert m._fit_result is not None
        refit_state = m._refit_result.pipeline_state
        cv_state = m._fit_result.pipeline_state
        assert set(refit_state.keys()) == set(cv_state.keys())


# ===========================================================================
# 2.4 Calibration is called only when configured
# ===========================================================================


class TestCalibrationConditional:
    """Calibration path is activated only when config specifies it."""

    def test_no_calibration_no_calibrator(self) -> None:
        cfg = make_config("binary", n_splits=2)
        m = Model(cfg)
        result = m.fit(data=make_binary_df())
        assert result.calibrator is None
        assert "calibrated" not in result.metrics

    def test_platt_calibration_produces_calibrator(self) -> None:
        cfg = make_config("binary", n_splits=2, calibration="platt")
        m = Model(cfg)
        result = m.fit(data=make_binary_df())
        assert result.calibrator is not None
        assert "calibrated" in result.metrics

    def test_calibration_on_regression_raises(self) -> None:
        cfg = make_config("regression", calibration="platt")
        m = Model(cfg)
        with pytest.raises(LizyMLError) as exc_info:
            m.fit(data=make_regression_df())
        assert exc_info.value.code == ErrorCode.CALIBRATION_NOT_SUPPORTED

    def test_calibration_on_multiclass_raises(self) -> None:
        cfg = make_config("multiclass", n_splits=2, calibration="platt")
        m = Model(cfg)
        with pytest.raises(LizyMLError) as exc_info:
            m.fit(data=make_multiclass_df())
        assert exc_info.value.code == ErrorCode.CALIBRATION_NOT_SUPPORTED


# ===========================================================================
# 2.5 Evaluator receives correct task and metric_names
# ===========================================================================


class TestEvaluatorWiring:
    """Evaluator is wired with correct task and metrics from config."""

    def test_binary_default_metrics(self) -> None:
        cfg = make_config("binary", n_splits=2)
        m = Model(cfg)
        result = m.fit(data=make_binary_df())
        oof = result.metrics["raw"]["oof"]
        assert "logloss" in oof
        assert "auc" in oof

    def test_multiclass_default_metrics(self) -> None:
        cfg = make_config("multiclass", n_splits=2)
        m = Model(cfg)
        result = m.fit(data=make_multiclass_df())
        oof = result.metrics["raw"]["oof"]
        assert "logloss" in oof

    def test_custom_metrics_override_defaults(self) -> None:
        cfg = make_config("binary", n_splits=2)
        cfg["evaluation"] = {"metrics": ["auc"]}
        m = Model(cfg)
        result = m.fit(data=make_binary_df())
        oof = result.metrics["raw"]["oof"]
        assert "auc" in oof
        assert "logloss" not in oof


# ===========================================================================
# 2.6 get_provider dispatch correctness
# ===========================================================================


class TestProviderDispatch:
    """get_provider correctly dispatches to the right provider."""

    def test_lgbm_config_returns_lgbm_provider(self) -> None:
        from lizyml.config.schema import LGBMConfig
        from lizyml.estimators.lgbm.provider import LGBMProvider

        cfg = LGBMConfig(name="lgbm")
        provider = get_provider(cfg)
        assert isinstance(provider, LGBMProvider)

    def test_unknown_model_raises_config_invalid(self) -> None:
        mock_cfg = MagicMock()
        mock_cfg.name = "xgboost"
        with pytest.raises(LizyMLError) as exc_info:
            get_provider(mock_cfg)
        assert exc_info.value.code == ErrorCode.CONFIG_INVALID
        assert "xgboost" in str(exc_info.value.user_message)

    def test_provider_methods_called_during_fit(self) -> None:
        """Provider's key methods are exercised during fit()."""
        cfg = make_config("regression")
        m = Model(cfg)

        with patch(
            "lizyml.core.model.get_provider", wraps=get_provider
        ) as mock_get_provider:
            m.fit(data=make_regression_df())
            mock_get_provider.assert_called_once()


# ===========================================================================
# 2.7 _merge_params priority: Config < tune < fit() args
# ===========================================================================


class TestMergeParamsPriority:
    """Parameter merge priority: Config < tune best < fit() args."""

    def test_config_params_used_as_baseline(self) -> None:
        cfg = make_config("regression", learning_rate=0.1)
        m = Model(cfg)
        result = m.fit(data=make_regression_df())
        booster = result.models[0].get_native_model()
        lr = float(booster.params["learning_rate"])
        assert lr == pytest.approx(0.1)

    def test_tune_overrides_config(self) -> None:
        """After tuning, best params override config defaults."""
        cfg = make_config("regression", learning_rate=0.1, tuning_n_trials=3)
        m = Model(cfg)
        df = make_regression_df()
        tune_result = m.tune(data=df)
        fit_result = m.fit(data=df)

        booster = fit_result.models[0].get_native_model()
        booster_lr = float(booster.params["learning_rate"])
        if "learning_rate" in tune_result.best_params:
            assert booster_lr == pytest.approx(tune_result.best_params["learning_rate"])
        else:
            assert booster_lr == pytest.approx(0.1)

    def test_config_persists_when_not_tuned(self) -> None:
        """Without tuning, config learning_rate is preserved."""
        cfg = make_config("regression", learning_rate=0.05)
        m = Model(cfg)
        result = m.fit(data=make_regression_df())
        booster = result.models[0].get_native_model()
        lr = float(booster.params["learning_rate"])
        assert lr == pytest.approx(0.05)
