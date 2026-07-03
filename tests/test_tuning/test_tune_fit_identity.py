"""Issue #76: Tune → Fit identity tests.

Verifies that:
1. default_fixed_params() does NOT contain smart params (auto_num_leaves)
2. best_training_params are applied during fit() after tune()
3. Tune best trial and subsequent fit() produce identical OOF scores
"""

from __future__ import annotations

import logging

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
    """Tune best trial score must exactly match fit() OOF score.

    After the code-path unification (#76 follow-up), tune objective and
    fit() both go through ``_build_train_components`` with identical
    parameters.  The OOF scores must be bit-for-bit identical because the
    same seed, splitter, and resolved params are used.
    """

    def test_tune_then_fit_same_oof_score_regression(self) -> None:
        """Regression: tune best score == fit OOF score (exact)."""
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

        assert fit_oof_score == pytest.approx(tune_best_score, rel=1e-10), (
            f"Tune best score ({tune_best_score}) != Fit OOF score ({fit_oof_score}) "
            f"for metric '{metric_name}'. "
            f"Tune and fit code paths may diverge."
        )

    def test_tune_then_fit_same_oof_score_binary(self) -> None:
        """Binary classification: tune best score == fit OOF score (exact)."""
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

        assert fit_oof_score == pytest.approx(tune_best_score, rel=1e-10), (
            f"Tune best score ({tune_best_score}) != Fit OOF score ({fit_oof_score}) "
            f"for metric '{metric_name}'. "
            f"Tune and fit code paths may diverge."
        )


# ===================================================================
# Phase 4: H-0079 — tune-sampled objective reaches refit booster
# ===================================================================


class TestTuneSampledObjectiveIdentity:
    """L2 of H-0079: when tune samples ``objective`` from default_space,
    the refit booster must train with the **sampled** value, not the
    task-locked default.

    Pre-H-0079 the score-equality test (``TestTuneFitIdentity``) passed
    even with the silent strip, because both tune trial and refit threw
    away the sampled objective the same way. This test plugs the gap
    by inspecting the booster's actual ``params["objective"]`` after
    fit, independent of any score comparison.
    """

    def test_tune_sampled_objective_matches_refit_booster_regression(self) -> None:
        """Refit booster's objective must equal ``best_params["objective"]``."""
        df = make_regression_df(n=300, seed=0)
        cfg_dict = make_config(
            "regression", n_estimators=30, n_splits=2, tuning_n_trials=5, seed=42
        )
        cfg = LizyMLConfig(**cfg_dict)

        m = Model(cfg)
        tune_result = m.tune(data=df)

        # default_space("regression") samples objective ∈ {"huber", "fair"}.
        best_objective = tune_result.best_params.get("objective")
        assert best_objective is not None, (
            "tune.best_params must include 'objective' when default_space "
            "is used (regression)."
        )

        m.fit(data=df)
        refit_booster = m._refit_result.model.get_native_model()  # type: ignore[union-attr]
        actual = refit_booster.params["objective"]
        assert actual == best_objective, (
            f"Tune sampled objective='{best_objective}' but refit booster "
            f"trained with '{actual}'. _build_params() may be stripping "
            f"the sampled value (H-0079 silent override)."
        )


# ===================================================================
# Phase 5: #218 — post-tune fit() emits an optimistic-bias warning
# ===================================================================


class TestPostTuneFitWarning:
    """fit() after tune() reuses the tuning CV splits, so its OOF metrics are
    optimistically biased. The facade must surface this once per fit (#218).
    """

    def test_fit_after_tune_logs_bias_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        df = make_regression_df(n=300, seed=0)
        cfg_dict = make_config(
            "regression", n_estimators=30, n_splits=2, tuning_n_trials=3, seed=42
        )
        cfg = LizyMLConfig(**cfg_dict)

        m = Model(cfg)
        m.tune(data=df)
        with caplog.at_level(logging.WARNING, logger="lizyml.model"):
            m.fit(data=df)

        post_tune = [r for r in caplog.records if "fit.post_tune" in r.getMessage()]
        assert len(post_tune) == 1, (
            "fit() after tune() must emit exactly one post_tune bias warning"
        )

    def test_fit_without_tune_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        df = make_regression_df(n=300, seed=0)
        cfg_dict = make_config("regression", n_estimators=30, n_splits=2, seed=42)
        cfg = LizyMLConfig(**cfg_dict)

        m = Model(cfg)
        with caplog.at_level(logging.WARNING, logger="lizyml.model"):
            m.fit(data=df)

        assert not [r for r in caplog.records if "fit.post_tune" in r.getMessage()], (
            "fit() without a prior tune() must not emit the post_tune warning"
        )
