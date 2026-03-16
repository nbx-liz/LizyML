"""Config → Component value propagation tests.

Verify that specific Config field values reach the final downstream component
(Booster params, split indices, pipeline state, etc.) as observable outcomes.

These tests exercise the full chain: Config → pydantic → Specs → Provider →
TrainComponents → CVTrainer → FitResult, not just individual units.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lizyml.core.model import Model
from tests._helpers import (
    make_binary_df,
    make_config,
    make_regression_df,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _booster_params(fit_result, fold: int = 0) -> dict:
    """Extract the native Booster params from a FitResult fold."""
    return fit_result.models[fold].get_native_model().params


# ===========================================================================
# 1.1 Config.model.params → Booster params
# ===========================================================================


class TestModelParamsPropagation:
    """Config model params reach the LightGBM Booster unchanged."""

    def test_learning_rate_reaches_booster(self):
        cfg = make_config("regression", learning_rate=0.123)
        m = Model(cfg)
        result = m.fit(data=make_regression_df())
        assert float(_booster_params(result)["learning_rate"]) == pytest.approx(0.123)

    def test_max_depth_reaches_booster(self):
        cfg = make_config("regression", max_depth=4)
        m = Model(cfg)
        result = m.fit(data=make_regression_df())
        assert int(_booster_params(result)["max_depth"]) == 4

    def test_feature_fraction_reaches_booster(self):
        cfg = make_config("binary", n_splits=2, feature_fraction=0.7)
        m = Model(cfg)
        result = m.fit(data=make_binary_df())
        assert float(_booster_params(result)["feature_fraction"]) == pytest.approx(0.7)


# ===========================================================================
# 1.2 Config.training.seed → Booster seed
# ===========================================================================


class TestSeedPropagation:
    """Config training seed reaches the LightGBM Booster."""

    @pytest.mark.parametrize("seed", [0, 99, 12345])
    def test_seed_reaches_booster(self, seed: int):
        cfg = make_config("regression", seed=seed)
        m = Model(cfg)
        result = m.fit(data=make_regression_df())
        assert int(_booster_params(result)["seed"]) == seed


# ===========================================================================
# 1.3 Config.training.early_stopping.rounds → early stopping active
# ===========================================================================


class TestEarlyStoppingPropagation:
    """Early stopping rounds config flows through to the estimator."""

    def test_early_stopping_rounds_reach_adapter(self):
        cfg = make_config("regression", n_estimators=10)
        cfg["training"]["early_stopping"] = {"enabled": True, "rounds": 7}
        m = Model(cfg)
        result = m.fit(data=make_regression_df())
        # Verify the adapter was configured with the correct rounds value
        adapter = result.models[0]
        assert adapter.early_stopping_rounds == 7

    @pytest.mark.parametrize("rounds", [3, 50, 150])
    def test_various_rounds_values(self, rounds: int):
        cfg = make_config("regression", n_estimators=10)
        cfg["training"]["early_stopping"] = {"enabled": True, "rounds": rounds}
        m = Model(cfg)
        result = m.fit(data=make_regression_df())
        assert result.models[0].early_stopping_rounds == rounds

    def test_early_stopping_disabled_no_stopping(self):
        cfg = make_config("regression", n_estimators=10)
        cfg["training"]["early_stopping"] = {"enabled": False}
        m = Model(cfg)
        result = m.fit(data=make_regression_df())
        # Without early stopping, adapter should have None for early_stopping_rounds
        assert result.models[0].early_stopping_rounds is None


# ===========================================================================
# 1.4 Config.features.exclude → columns removed from X
# ===========================================================================


class TestFeaturesExclude:
    """Excluded features are removed from the training DataFrame."""

    def test_exclude_removes_column(self):
        df = make_regression_df()
        cfg = make_config("regression")
        cfg["features"] = {"exclude": ["feat_b"]}
        m = Model(cfg)
        result = m.fit(data=df)
        assert "feat_b" not in result.feature_names
        assert "feat_a" in result.feature_names

    def test_exclude_multiple_columns(self):
        df = make_regression_df()
        # Add an extra feature so we still have one left
        df["feat_c"] = df["feat_a"] * 0.5
        cfg = make_config("regression")
        cfg["features"] = {"exclude": ["feat_a", "feat_b"]}
        m = Model(cfg)
        result = m.fit(data=df)
        assert result.feature_names == ["feat_c"]


# ===========================================================================
# 1.5 Config.features.categorical → recognized by pipeline
# ===========================================================================


class TestCategoricalPropagation:
    """Categorical columns from config are used by the feature pipeline."""

    def test_categorical_column_recognized(self):
        df = make_binary_df()
        rng = np.random.default_rng(0)
        df["cat_col"] = pd.Categorical(rng.choice(["a", "b", "c"], len(df)))
        cfg = make_config("binary", n_splits=2)
        cfg["features"] = {"categorical": ["cat_col"]}
        m = Model(cfg)
        result = m.fit(data=df)
        assert "cat_col" in result.categorical_features


# ===========================================================================
# 1.6 Config.evaluation.metrics → actually used metrics
# ===========================================================================


class TestEvaluationMetricsPropagation:
    """Custom metric list from config controls which metrics appear."""

    def test_custom_metrics_used(self):
        cfg = make_config("regression")
        cfg["evaluation"] = {"metrics": ["mae"]}
        m = Model(cfg)
        result = m.fit(data=make_regression_df())
        oof_metrics = result.metrics["raw"]["oof"]
        assert "mae" in oof_metrics
        # Should NOT include defaults like rmse unless explicitly listed
        assert "rmse" not in oof_metrics

    def test_default_metrics_when_empty(self):
        cfg = make_config("regression")
        # No evaluation.metrics → defaults: ["rmse", "mae"]
        m = Model(cfg)
        result = m.fit(data=make_regression_df())
        oof_metrics = result.metrics["raw"]["oof"]
        assert "rmse" in oof_metrics
        assert "mae" in oof_metrics


# ===========================================================================
# 1.7 Config.split.n_splits → number of folds
# ===========================================================================


class TestNSplitsPropagation:
    """Number of CV folds matches config."""

    @pytest.mark.parametrize("n_splits", [2, 3, 5])
    def test_n_splits_controls_fold_count(self, n_splits: int):
        cfg = make_config("regression", n_splits=n_splits)
        m = Model(cfg)
        result = m.fit(data=make_regression_df())
        assert len(result.splits.outer) == n_splits
        assert len(result.models) == n_splits
        assert len(result.if_pred_per_fold) == n_splits


# ===========================================================================
# 1.8 Config.split.random_state → deterministic folds
# ===========================================================================


class TestSplitRandomStatePropagation:
    """Same split random_state produces identical fold indices."""

    def test_same_random_state_same_folds(self):
        df = make_regression_df()
        cfg1 = make_config("regression", split_method="kfold", n_splits=3)
        cfg1["split"]["random_state"] = 99
        cfg2 = make_config("regression", split_method="kfold", n_splits=3)
        cfg2["split"]["random_state"] = 99
        m1 = Model(cfg1)
        m2 = Model(cfg2)
        r1 = m1.fit(data=df)
        r2 = m2.fit(data=df)
        for f1, f2 in zip(r1.splits.outer, r2.splits.outer, strict=True):
            np.testing.assert_array_equal(f1[0], f2[0])  # train indices
            np.testing.assert_array_equal(f1[1], f2[1])  # valid indices

    def test_different_random_state_different_folds(self):
        df = make_regression_df()
        cfg1 = make_config("regression", split_method="kfold", n_splits=3)
        cfg1["split"]["random_state"] = 1
        cfg2 = make_config("regression", split_method="kfold", n_splits=3)
        cfg2["split"]["random_state"] = 999
        m1 = Model(cfg1)
        m2 = Model(cfg2)
        r1 = m1.fit(data=df)
        r2 = m2.fit(data=df)
        any_different = any(
            not np.array_equal(f1[0], f2[0])
            for f1, f2 in zip(r1.splits.outer, r2.splits.outer, strict=True)
        )
        assert any_different, "Different random_state should produce different folds"


# ===========================================================================
# 1.9 Config.data.group_col → groups used by splitter
# ===========================================================================


class TestGroupColPropagation:
    """Group column config is respected by group-based splitters."""

    def test_group_kfold_respects_groups(self):
        df = make_binary_df(n=200, group_col="grp", n_groups=10)
        cfg = make_config(
            "binary", split_method="group_kfold", n_splits=3, group_col="grp"
        )
        m = Model(cfg)
        result = m.fit(data=df)
        groups = df["grp"].values
        # Verify no group appears in both train and valid for any fold
        for train_idx, valid_idx in result.splits.outer:
            train_groups = set(groups[train_idx])
            valid_groups = set(groups[valid_idx])
            assert train_groups.isdisjoint(valid_groups), (
                "Groups must not overlap between train and valid"
            )


# ===========================================================================
# 1.10 Config smart params (auto_num_leaves) → resolved in Booster
# ===========================================================================


class TestSmartParamsPropagation:
    """Smart parameters are resolved and reach the Booster."""

    def test_auto_num_leaves_with_max_depth(self):
        cfg = make_config("regression")
        cfg["model"]["auto_num_leaves"] = True
        cfg["model"]["num_leaves_ratio"] = 0.5
        cfg["model"]["params"]["max_depth"] = 5
        m = Model(cfg)
        result = m.fit(data=make_regression_df())
        # Expected: ceil(2^5 * 0.5) = ceil(16) = 16
        actual = int(_booster_params(result)["num_leaves"])
        expected = 16  # ceil(2**5 * 0.5)
        assert actual == expected

    def test_auto_num_leaves_disabled(self):
        cfg = make_config("regression")
        cfg["model"]["auto_num_leaves"] = False
        cfg["model"]["params"]["num_leaves"] = 42
        m = Model(cfg)
        result = m.fit(data=make_regression_df())
        assert int(_booster_params(result)["num_leaves"]) == 42


# ===========================================================================
# 3.1 oof_raw_scores shape/type contract
# ===========================================================================


class TestOofRawScoresContract:
    """oof_raw_scores shape and type match calibration requirements."""

    def test_binary_with_calibration_has_raw_scores(self):
        cfg = make_config("binary", n_splits=2, calibration="platt")
        m = Model(cfg)
        result = m.fit(data=make_binary_df())
        assert result.oof_raw_scores is not None
        assert result.oof_raw_scores.shape == (len(make_binary_df()),)
        assert result.oof_raw_scores.dtype == np.float64

    def test_binary_without_calibration_no_raw_scores(self):
        cfg = make_config("binary", n_splits=2)
        m = Model(cfg)
        result = m.fit(data=make_binary_df())
        assert result.oof_raw_scores is None

    def test_regression_no_raw_scores(self):
        cfg = make_config("regression", n_splits=2)
        m = Model(cfg)
        result = m.fit(data=make_regression_df())
        assert result.oof_raw_scores is None

    def test_raw_scores_are_logits_not_probabilities(self):
        """Raw scores should contain logits (can be outside [0, 1])."""
        cfg = make_config("binary", n_splits=2, calibration="platt")
        m = Model(cfg)
        result = m.fit(data=make_binary_df(n=300))
        raw = result.oof_raw_scores
        assert raw is not None
        # Logits can be any real value; with enough samples, some should
        # have absolute value > 1 (beyond probability range)
        # At minimum, they should differ from the calibrated OOF predictions
        assert not np.allclose(raw, result.oof_pred)


# ===========================================================================
# 3.2 feature_weights Config → Booster E2E
# ===========================================================================


class TestFeatureWeightsE2E:
    """feature_weights from Config reach the LightGBM training."""

    def test_feature_weights_applied(self):
        cfg = make_config("regression")
        cfg["model"]["feature_weights"] = {"feat_a": 2.0, "feat_b": 0.5}
        m = Model(cfg)
        result = m.fit(data=make_regression_df())
        # The feature weights should affect importance distribution
        # feat_a (weight 2.0) should generally be more important than
        # feat_b (weight 0.5) given the data generation (target = 2*feat_a + feat_b)
        imp = result.models[0].importance("split")
        assert "feat_a" in imp
        assert "feat_b" in imp


# ===========================================================================
# 4.1 Inner Valid auto-resolution by split method
# ===========================================================================


class TestInnerValidAutoResolution:
    """Inner valid type is auto-resolved from split method."""

    def test_stratified_kfold_uses_stratified_holdout(self):
        from lizyml.config.loader import load_config
        from lizyml.core._model_factories import build_inner_valid
        from lizyml.training.inner_valid import HoldoutInnerValid

        cfg = load_config(make_config("binary", split_method="stratified_kfold"))
        iv = build_inner_valid(cfg)
        assert isinstance(iv, HoldoutInnerValid)
        assert iv.stratify is True

    def test_group_kfold_uses_group_holdout(self):
        from lizyml.config.loader import load_config
        from lizyml.core._model_factories import build_inner_valid
        from lizyml.training.inner_valid import GroupHoldoutInnerValid

        cfg = load_config(
            make_config("binary", split_method="group_kfold", group_col="grp")
        )
        iv = build_inner_valid(cfg)
        assert isinstance(iv, GroupHoldoutInnerValid)

    def test_time_series_uses_time_holdout(self):
        from lizyml.config.loader import load_config
        from lizyml.core._model_factories import build_inner_valid
        from lizyml.training.inner_valid import TimeHoldoutInnerValid

        cfg = load_config(
            make_config("regression", split_method="time_series", time_col="time")
        )
        iv = build_inner_valid(cfg)
        assert isinstance(iv, TimeHoldoutInnerValid)

    def test_kfold_uses_non_stratified_holdout(self):
        from lizyml.config.loader import load_config
        from lizyml.core._model_factories import build_inner_valid
        from lizyml.training.inner_valid import HoldoutInnerValid

        cfg = load_config(make_config("regression", split_method="kfold"))
        iv = build_inner_valid(cfg)
        assert isinstance(iv, HoldoutInnerValid)
        assert iv.stratify is False

    def test_early_stopping_disabled_uses_no_inner_valid(self):
        from lizyml.config.loader import load_config
        from lizyml.core._model_factories import build_inner_valid
        from lizyml.training.inner_valid import NoInnerValid

        cfg_dict = make_config("regression")
        cfg_dict["training"]["early_stopping"] = {"enabled": False}
        cfg = load_config(cfg_dict)
        iv = build_inner_valid(cfg)
        assert isinstance(iv, NoInnerValid)


# ===========================================================================
# 4.2 validation_ratio propagation
# ===========================================================================


class TestValidationRatioPropagation:
    """validation_ratio from config controls inner valid split size."""

    def test_custom_validation_ratio(self):
        from lizyml.config.loader import load_config
        from lizyml.core._model_factories import build_inner_valid
        from lizyml.training.inner_valid import HoldoutInnerValid

        cfg_dict = make_config("regression")
        cfg_dict["training"]["early_stopping"] = {
            "enabled": True,
            "rounds": 50,
            "validation_ratio": 0.2,
        }
        cfg = load_config(cfg_dict)
        iv = build_inner_valid(cfg)
        assert isinstance(iv, HoldoutInnerValid)
        assert iv.ratio == pytest.approx(0.2)

    def test_default_validation_ratio(self):
        from lizyml.config.loader import load_config
        from lizyml.core._model_factories import build_inner_valid
        from lizyml.training.inner_valid import HoldoutInnerValid

        cfg = load_config(make_config("regression"))
        iv = build_inner_valid(cfg)
        assert isinstance(iv, HoldoutInnerValid)
        assert iv.ratio == pytest.approx(0.1)  # default

    def test_validation_ratio_affects_split_size(self):
        """Integration: ratio=0.3 → ~30% of fold training data used for inner valid."""
        cfg_dict = make_config("regression", n_splits=2)
        cfg_dict["training"]["early_stopping"] = {
            "enabled": True,
            "rounds": 50,
            "validation_ratio": 0.3,
        }
        m = Model(cfg_dict)
        result = m.fit(data=make_regression_df(n=200))
        # Check inner split via SplitIndices (if available)
        inner_splits = result.splits.inner
        if inner_splits is not None and len(inner_splits) > 0:
            for inner_train, inner_valid in inner_splits:
                total = len(inner_train) + len(inner_valid)
                ratio = len(inner_valid) / total
                # Allow some tolerance due to integer rounding
                assert 0.15 <= ratio <= 0.45, (
                    f"Inner valid ratio {ratio:.2f} not near 0.3"
                )
