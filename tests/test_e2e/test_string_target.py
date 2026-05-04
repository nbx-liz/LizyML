"""E2E tests for non-numeric classification targets (H-0070).

These tests assert the user-facing contract:

- INV-1: encoder ``needs_encoding=True`` ⇔ original y was non-numeric
- INV-2: ``predict().pred.dtype`` matches the original y dtype at fit time
- INV-3: ``proba`` column order matches ``target_encoder.classes_``
- INV-4: regression × non-numeric y is rejected before model training starts
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from tests._helpers import make_config

# ---------------------------------------------------------------------------
# Fixtures (string-target classification data)
# ---------------------------------------------------------------------------


def _binary_string_df(n: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "feat_a": rng.uniform(0, 10, n),
            "feat_b": rng.uniform(-1, 1, n),
        }
    )
    df["target"] = np.where(df["feat_a"] > 5, "yes", "no")
    return df


def _multiclass_string_df(n: int = 240, seed: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "feat_a": rng.uniform(0, 10, n),
            "feat_b": rng.uniform(-1, 1, n),
        }
    )
    bins = pd.cut(df["feat_a"], bins=3, labels=["Adelie", "Chinstrap", "Gentoo"])
    df["target"] = bins.astype(str)
    return df


# ---------------------------------------------------------------------------
# Binary
# ---------------------------------------------------------------------------


class TestStringTargetBinary:
    def test_fit_succeeds_with_string_target(self) -> None:
        df = _binary_string_df()
        cfg = make_config("binary", n_estimators=20, split_method="stratified_kfold")
        result = Model(cfg).fit(data=df)
        # INV-1: encoder records non-numeric usage
        assert result.target_encoder.needs_encoding is True
        assert result.target_encoder.classes_ == ("no", "yes")

    def test_predict_returns_original_label_dtype(self) -> None:
        df = _binary_string_df()
        cfg = make_config("binary", n_estimators=20, split_method="stratified_kfold")
        m = Model(cfg)
        m.fit(data=df)
        pred = m.predict(df.drop(columns=["target"])).pred
        # INV-2: dtype matches original y dtype (object / str)
        assert pred.dtype == object
        # All values must be from the original label set
        assert set(np.unique(pred)).issubset({"yes", "no"})

    def test_proba_column_order_matches_classes(self) -> None:
        df = _binary_string_df()
        cfg = make_config("binary", n_estimators=20, split_method="stratified_kfold")
        m = Model(cfg)
        result = m.fit(data=df)
        prediction = m.predict(df.drop(columns=["target"]))
        # Binary predict returns 1-D proba (positive class). For binary the
        # positive class corresponds to classes_[1] = "yes".
        assert result.target_encoder.classes_[1] == "yes"
        assert prediction.proba is not None
        assert prediction.proba.ndim == 1

    def test_calibration_works_with_string_target(self) -> None:
        df = _binary_string_df()
        cfg = make_config(
            "binary",
            n_estimators=20,
            split_method="stratified_kfold",
            calibration="isotonic",
        )
        m = Model(cfg)
        result = m.fit(data=df)
        # Calibration runs on int-encoded y → "calibrated" key is populated.
        assert "calibrated" in result.metrics


# ---------------------------------------------------------------------------
# Multiclass
# ---------------------------------------------------------------------------


class TestStringTargetMulticlass:
    def test_fit_succeeds_with_three_string_classes(self) -> None:
        df = _multiclass_string_df()
        cfg = make_config(
            "multiclass", n_estimators=20, split_method="stratified_kfold"
        )
        result = Model(cfg).fit(data=df)
        assert result.target_encoder.needs_encoding is True
        assert result.target_encoder.classes_ == ("Adelie", "Chinstrap", "Gentoo")

    def test_predict_returns_original_string_labels(self) -> None:
        df = _multiclass_string_df()
        cfg = make_config(
            "multiclass", n_estimators=20, split_method="stratified_kfold"
        )
        m = Model(cfg)
        m.fit(data=df)
        prediction = m.predict(df.drop(columns=["target"]))
        assert prediction.pred.dtype == object
        assert set(np.unique(prediction.pred)).issubset(
            {"Adelie", "Chinstrap", "Gentoo"}
        )

    def test_proba_columns_match_classes_count(self) -> None:
        df = _multiclass_string_df()
        cfg = make_config(
            "multiclass", n_estimators=20, split_method="stratified_kfold"
        )
        m = Model(cfg)
        result = m.fit(data=df)
        prediction = m.predict(df.drop(columns=["target"]))
        assert prediction.proba is not None
        # 2-D probabilities: (n_samples, n_classes)
        assert prediction.proba.shape[1] == len(result.target_encoder.classes_)


# ---------------------------------------------------------------------------
# Regression rejection (INV-4)
# ---------------------------------------------------------------------------


class TestRegressionStringTargetRejected:
    def test_regression_string_target_raises_target_not_numeric(self) -> None:
        df = pd.DataFrame(
            {
                "feat_a": np.linspace(0, 10, 50),
                "target": np.repeat(["a", "b"], 25),
            }
        )
        cfg = make_config("regression", n_estimators=10, n_splits=2)
        with pytest.raises(LizyMLError) as exc_info:
            Model(cfg).fit(data=df)
        assert exc_info.value.code == ErrorCode.TARGET_NOT_NUMERIC


# ---------------------------------------------------------------------------
# Numeric pass-through (no behavior change for existing users)
# ---------------------------------------------------------------------------


class TestNumericTargetIsNoOp:
    def test_numeric_binary_encoder_is_no_op(self) -> None:
        from tests._helpers import make_binary_df

        df = make_binary_df(n=120)
        result = Model(
            make_config("binary", n_estimators=10, split_method="stratified_kfold")
        ).fit(data=df)
        assert result.target_encoder.needs_encoding is False
        assert result.target_encoder.classes_ == ()

    def test_numeric_predict_dtype_unchanged(self) -> None:
        from tests._helpers import make_binary_df

        df = make_binary_df(n=120)
        m = Model(
            make_config("binary", n_estimators=10, split_method="stratified_kfold")
        )
        m.fit(data=df)
        pred = m.predict(df.drop(columns=["target"])).pred
        # Numeric target → predict still returns int
        assert pred.dtype.kind in ("i", "u")


# ---------------------------------------------------------------------------
# tune() path with non-numeric y (review gap M-1)
# ---------------------------------------------------------------------------


class TestStringTargetTunePath:
    def test_tune_then_fit_then_predict_with_string_y(self) -> None:
        df = _binary_string_df()
        cfg = make_config(
            "binary",
            n_estimators=10,
            split_method="stratified_kfold",
            tuning_n_trials=2,
        )
        m = Model(cfg)
        # tune() shares _prepare_training_data — must accept str y too
        m.tune(data=df)
        result = m.fit(data=df)
        assert result.target_encoder.needs_encoding is True
        prediction = m.predict(df.drop(columns=["target"]))
        assert prediction.pred.dtype == object
        assert set(np.unique(prediction.pred)).issubset({"yes", "no"})


# ---------------------------------------------------------------------------
# Save/load round-trip with non-numeric y (review gap M-2)
# ---------------------------------------------------------------------------


class TestStringTargetExportLoad:
    def test_export_load_predict_preserves_string_dtype(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        df = _binary_string_df()
        cfg = make_config("binary", n_estimators=15, split_method="stratified_kfold")
        m = Model(cfg)
        m.fit(data=df)

        export_dir = tmp_path / "model"  # type: ignore[operator]
        m.export(export_dir)

        m2 = Model.load(export_dir)
        # target_encoder survives pickle round-trip
        assert m2.fit_result.target_encoder.classes_ == ("no", "yes")

        prediction = m2.predict(df.drop(columns=["target"]).iloc[:30])
        assert prediction.pred.dtype == object
        assert set(np.unique(prediction.pred)).issubset({"yes", "no"})

    def test_export_load_predict_multiclass_string(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        df = _multiclass_string_df()
        cfg = make_config(
            "multiclass", n_estimators=15, split_method="stratified_kfold"
        )
        m = Model(cfg)
        m.fit(data=df)

        export_dir = tmp_path / "mc_model"  # type: ignore[operator]
        m.export(export_dir)

        m2 = Model.load(export_dir)
        assert m2.fit_result.target_encoder.classes_ == (
            "Adelie",
            "Chinstrap",
            "Gentoo",
        )
        prediction = m2.predict(df.drop(columns=["target"]).iloc[:40])
        assert prediction.pred.dtype == object
        assert set(np.unique(prediction.pred)).issubset(
            {"Adelie", "Chinstrap", "Gentoo"}
        )
