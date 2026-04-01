"""Issue #76: Tune → Fit identity tests.

Verifies that:
1. default_fixed_params() does NOT contain smart params (auto_num_leaves)
2. best_training_params are applied during fit() after tune()
3. Tune best trial and subsequent fit() produce identical OOF scores
"""

from __future__ import annotations

import pytest

from lizyml.config.schema import LizyMLConfig
from lizyml.core.model import Model
from lizyml.estimators.lgbm.defaults import default_fixed_params
from tests._helpers import make_binary_df, make_config, make_regression_df

# ===================================================================
# Phase 1: default_fixed_params must NOT contain smart params
# ===================================================================


class TestDefaultFixedParamsNoSmartParams:
    """auto_num_leaves is a smart param and must not be in fixed params."""

    @pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
    def test_auto_num_leaves_not_in_fixed_params(self, task: str) -> None:
        fp = default_fixed_params(task)
        assert "auto_num_leaves" not in fp, (
            f"auto_num_leaves is a smart param, not a fixed model param; "
            f"found in default_fixed_params('{task}')"
        )

    @pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
    def test_fixed_params_contain_only_model_keys(self, task: str) -> None:
        """All keys in default_fixed_params must be valid LightGBM params."""
        fp = default_fixed_params(task)
        # These are the only expected fixed keys
        allowed = {"first_metric_only", "metric"}
        assert set(fp.keys()) <= allowed, (
            f"Unexpected keys in default_fixed_params: {set(fp.keys()) - allowed}"
        )


# ===================================================================
# Phase 2: best_training_params applied during fit after tune
# ===================================================================


class TestTrainingParamsApplied:
    """After tune(), fit() must use best_training_params (esr, validation_ratio).

    Verified by comparing fit-after-tune (which applies best_training_params)
    against fit-without-tune (which uses config defaults).  If training params
    are actually applied, the OOF scores should differ.
    """

    def test_training_params_change_fit_behaviour(self) -> None:
        """fit() after tune() produces different OOF than fit() without tune()."""
        df = make_regression_df(n=300, seed=0)
        cfg_dict = make_config(
            "regression", n_estimators=50, n_splits=2, tuning_n_trials=3, seed=42
        )
        cfg = LizyMLConfig(**cfg_dict)

        # fit without tune (config defaults)
        m_no_tune = Model(cfg)
        fr_no_tune = m_no_tune.fit(data=df)
        score_no_tune = fr_no_tune.metrics["raw"]["oof"]["rmse"]

        # tune then fit (training params applied)
        m_tuned = Model(cfg)
        tune_result = m_tuned.tune(data=df)
        assert "early_stopping_rounds" in tune_result.best_training_params
        assert "validation_ratio" in tune_result.best_training_params

        fr_tuned = m_tuned.fit(data=df)
        score_tuned = fr_tuned.metrics["raw"]["oof"]["rmse"]

        # Scores should differ because tuned training params (ESR,
        # validation_ratio) override the config defaults.
        assert score_no_tune != pytest.approx(score_tuned, rel=1e-6), (
            "fit() after tune() should produce different OOF than "
            "fit() without tune() (training params not applied?)"
        )


# ===================================================================
# Phase 3: Tune → Fit OOF identity
# ===================================================================


class TestTuneFitIdentity:
    """Tune best trial score must match fit() OOF score after applying all params.

    A small tolerance (rel=1e-3) is used because LightGBM's internal
    thread-local state may cause minor floating-point differences even
    with identical parameters and seed.  The critical assertion is that
    scores are practically identical — not off by orders of magnitude as
    they were before the #76 fix.
    """

    def test_tune_then_fit_same_oof_score_regression(self) -> None:
        """Regression: tune best score ≈ fit OOF score on same metric."""
        df = make_regression_df(n=300, seed=0)
        cfg_dict = make_config(
            "regression", n_estimators=50, n_splits=2, tuning_n_trials=3, seed=42
        )
        cfg = LizyMLConfig(**cfg_dict)

        m = Model(cfg)
        tune_result = m.tune(data=df)
        fit_result = m.fit(data=df)

        metric_name = tune_result.metric_name
        tune_best_score = tune_result.best_score
        fit_oof_score = fit_result.metrics["raw"]["oof"][metric_name]

        assert fit_oof_score == pytest.approx(tune_best_score, rel=1e-3), (
            f"Tune best score ({tune_best_score}) != Fit OOF score ({fit_oof_score}) "
            f"for metric '{metric_name}'. "
            f"best_training_params may not be applied during fit."
        )

    def test_tune_then_fit_same_oof_score_binary(self) -> None:
        """Binary classification: tune best score ≈ fit OOF score."""
        df = make_binary_df(n=300, seed=0)
        cfg_dict = make_config(
            "binary", n_estimators=50, n_splits=2, tuning_n_trials=3, seed=42
        )
        cfg = LizyMLConfig(**cfg_dict)

        m = Model(cfg)
        tune_result = m.tune(data=df)
        fit_result = m.fit(data=df)

        metric_name = tune_result.metric_name
        tune_best_score = tune_result.best_score
        fit_oof_score = fit_result.metrics["raw"]["oof"][metric_name]

        assert fit_oof_score == pytest.approx(tune_best_score, rel=1e-3), (
            f"Tune best score ({tune_best_score}) != Fit OOF score ({fit_oof_score}) "
            f"for metric '{metric_name}'."
        )
