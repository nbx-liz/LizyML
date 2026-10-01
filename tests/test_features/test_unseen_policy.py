"""Unseen categories: the configured policy applies, and substitutions are reported
(H-0104, #260).

Before H-0104 an unseen category at predict time was replaced by the training
mode with no warning anywhere, and the policy could not be chosen from Config.
Every value of ``UnseenPolicy`` is read from the type, not listed, and each is
asserted for what the caller can observe: a warning plus the substituted value,
or ``DATA_SCHEMA_INVALID``.
"""

from __future__ import annotations

import typing
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml.config.loader import load_config
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.model import Model
from lizyml.features.encoders.categorical_encoder import UnseenPolicy
from tests._helpers import make_config

POLICIES: tuple[str, ...] = typing.get_args(UnseenPolicy)


def test_the_policy_population_is_the_type() -> None:
    assert set(POLICIES) == {"mode", "nan", "error"}


def _df(n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cat = rng.choice(["a", "b", "c"], size=n, p=[0.6, 0.25, 0.15])
    df = pd.DataFrame(
        {"num": rng.normal(size=n), "cat": pd.Series(cat).astype("category")}
    )
    effect = pd.Series(cat).map({"a": 0.0, "b": 3.0, "c": -3.0}).to_numpy()
    df["target"] = df["num"] + effect + rng.normal(scale=0.1, size=n)
    return df


def _config(policy: str | None) -> dict[str, Any]:
    cfg = make_config("regression", n_estimators=30)
    if policy is not None:
        cfg["features"] = {"unseen_policy": policy}
    return cfg


def _with_cat(X: pd.DataFrame, values: list[Any]) -> pd.DataFrame:
    out = X.copy()
    out["cat"] = pd.Series(values, index=out.index, dtype="object").astype("category")
    return out


def _new_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["target"]).iloc[:3].copy()


@pytest.mark.parametrize("policy", POLICIES)
def test_every_policy_is_observable_end_to_end(policy: str) -> None:
    df = _df()
    model = Model(_config(policy))
    model.fit(data=df)
    X = _new_rows(df)
    unseen = _with_cat(X, ["TYPO", "a", "b"])

    if policy == "error":
        with pytest.raises(LizyMLError) as exc:
            model.predict(unseen)
        assert exc.value.code is ErrorCode.DATA_SCHEMA_INVALID
        assert "TYPO" in str(exc.value)
        return

    result = model.predict(unseen)
    assert len(result.warnings) == 1, result.warnings
    assert "cat" in result.warnings[0] and "TYPO" in result.warnings[0]

    # The substituted row predicts exactly as if the caller had written the
    # replacement: the training mode, or a missing value.
    mode = df["cat"].mode().iloc[0]
    as_mode = model.predict(_with_cat(X, [mode, "a", "b"])).pred
    as_missing = model.predict(_with_cat(X, [None, "a", "b"])).pred
    # Discriminating fixture: the two replacements predict differently, so
    # matching one of them identifies which policy ran.
    assert not np.isclose(as_mode[0], as_missing[0]), (as_mode[0], as_missing[0])
    expected = as_mode if policy == "mode" else as_missing
    np.testing.assert_allclose(result.pred, expected)


def test_default_policy_reports_the_substitution() -> None:
    """#260's regression: the default is "mode", and it no longer stays silent."""
    df = _df()
    model = Model(_config(None))
    model.fit(data=df)

    result = model.predict(_with_cat(_new_rows(df), ["TYPO", "a", "b"]))
    assert result.warnings != [], "an unseen category was replaced silently"
    assert any("TYPO" in w for w in result.warnings)
    assert load_config(_config(None)).features.unseen_policy == "mode"


def test_no_warning_without_unseen_categories() -> None:
    df = _df()
    model = Model(_config(None))
    model.fit(data=df)
    assert model.predict(_new_rows(df)).warnings == []


def test_predict_follows_the_saved_policy_not_the_current_config(
    tmp_path: Path,
) -> None:
    """The policy the fit applied is recorded in the pipeline state, and predict
    follows that record even when the config now says something else (H-0104
    decision 6; code review round 1, blocking 2). Checked on the live model and
    on a loaded one, against a discriminating fixture: "nan" and "mode" predict
    differently, and "error" would raise."""
    df = _df()
    model = Model(_config("nan"))
    model.fit(data=df)
    X = _new_rows(df)
    unseen = _with_cat(X, ["TYPO", "a", "b"])
    as_missing = model.predict(_with_cat(X, [None, "a", "b"])).pred
    as_mode = model.predict(_with_cat(X, [df["cat"].mode().iloc[0], "a", "b"])).pred
    assert not np.isclose(as_missing[0], as_mode[0])

    for conflicting in ("mode", "error"):
        model._cfg.features.unseen_policy = conflicting  # type: ignore[assignment]
        result = model.predict(unseen)
        np.testing.assert_allclose(result.pred, as_missing)
        assert len(result.warnings) == 1 and "unseen_policy='nan'" in result.warnings[0]

    model._cfg.features.unseen_policy = "nan"
    path = model.export(tmp_path / "artifact")
    loaded = Model.load(path)
    for conflicting in ("mode", "error"):
        loaded._cfg.features.unseen_policy = conflicting  # type: ignore[assignment]
        np.testing.assert_allclose(loaded.predict(unseen).pred, as_missing)


def test_policy_survives_save_and_load(tmp_path: Path) -> None:
    df = _df()
    model = Model(_config("nan"))
    model.fit(data=df)
    assert model._refit_result is not None
    assert model._refit_result.pipeline_state["encoder"]["unseen_policy"] == "nan"

    path = model.export(tmp_path / "artifact")
    loaded = Model.load(path)
    unseen = _with_cat(_new_rows(df), ["TYPO", "a", "b"])
    before = model.predict(unseen)
    after = loaded.predict(unseen)
    assert after.warnings == before.warnings and len(after.warnings) == 1
    np.testing.assert_allclose(after.pred, before.pred)


def test_error_policy_applies_to_cv_valid_folds() -> None:
    """H-0104 decision 7: a value only one outer-valid fold contains is unseen by
    that fold's training rows, so ``"error"`` stops ``fit`` itself.

    This needs a string column the data builder leaves alone. With the default
    ``auto_categorical: true`` (or the column listed in ``features.categorical``)
    the builder casts it to ``category`` over **all** rows before CV, the
    encoder learns that declared list, and every fold knows every value -- so
    nothing is unseen during CV. Only with ``auto_categorical: false`` does the
    pipeline learn categories per fold.
    """
    df = _df()
    df["cat"] = df["cat"].astype(str)
    df.loc[0, "cat"] = "RARE"

    def cfg(policy: str | None, auto_categorical: bool) -> dict[str, Any]:
        raw = _config(policy)
        raw.setdefault("features", {})["auto_categorical"] = auto_categorical
        return raw

    with pytest.raises(LizyMLError) as exc:
        Model(cfg("error", auto_categorical=False)).fit(data=df)
    assert exc.value.code is ErrorCode.DATA_SCHEMA_INVALID
    assert "RARE" in str(exc.value)

    # Declared over all rows, the value is known to every fold.
    Model(cfg("error", auto_categorical=True)).fit(data=df)
    # And the default policy trains through the per-fold case.
    Model(cfg(None, auto_categorical=False)).fit(data=df)


def test_config_literal_is_the_encoder_type() -> None:
    """``config/`` cannot import ``features/`` (layer rule), so the accepted
    values are written twice; this keeps the two spellings one set (DC3)."""
    from lizyml.config.schema import FeaturesConfig

    annotation = FeaturesConfig.model_fields["unseen_policy"].annotation
    assert set(typing.get_args(annotation)) == set(POLICIES)


@pytest.mark.parametrize("policy", POLICIES)
def test_tune_applies_the_configured_policy(
    policy: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every pipeline the study builds carries the configured policy."""
    from lizyml.features import pipelines_native

    built: list[str] = []
    real_init = pipelines_native.NativeFeaturePipeline.__init__

    def spy(self: Any, unseen_policy: Any = "mode") -> None:
        built.append(unseen_policy)
        real_init(self, unseen_policy=unseen_policy)

    monkeypatch.setattr(pipelines_native.NativeFeaturePipeline, "__init__", spy)
    raw = make_config("regression", n_estimators=5, n_splits=2, tuning_n_trials=2)
    raw["features"] = {"unseen_policy": policy}
    Model(raw, data=_df()).tune()

    assert built, "tune built no pipeline"
    assert set(built) == {policy}, built


def _sliding_window_df(n: int = 60) -> pd.DataFrame:
    """A string column whose value "early" exists only in the first rows."""
    rng = np.random.default_rng(2)
    cat = np.array(["a"] * n, dtype=object)
    cat[:3] = "early"
    df = pd.DataFrame(
        {
            "t": pd.date_range("2024-01-01", periods=n, freq="D"),
            "num": rng.normal(size=n),
            "cat": cat,
        }
    )
    df["target"] = df["num"] + rng.normal(scale=0.1, size=n)
    return df


def _sliding_window_config(policy: str) -> dict[str, Any]:
    raw = make_config(
        "regression",
        n_estimators=5,
        n_splits=2,
        split_method="time_series",
        time_col="t",
        split_overrides={"train_size_max": 20},
    )
    raw["features"] = {"unseen_policy": policy, "auto_categorical": False}
    return raw


def test_shap_importance_applies_the_stored_policy_outside_the_last_fold() -> None:
    """H-0104 decision 8 (design review round 1, blocking 1).

    SHAP importance transforms all training rows with the **last** CV fold's
    pipeline. Under a sliding window some rows belong to no part of that fold,
    so a value only they hold is unseen there: ``"error"`` refuses SHAP
    importance although ``fit`` succeeded, and ``"mode"`` substitutes without a
    report. Pinned as the documented behaviour; the last-fold pipeline itself
    is a separate issue.
    """
    df = _sliding_window_df()

    strict = Model(_sliding_window_config("error"))
    strict.fit(data=df)
    with pytest.raises(LizyMLError) as exc:
        strict.importance(kind="shap")
    assert exc.value.code is ErrorCode.DATA_SCHEMA_INVALID
    assert "early" in str(exc.value)

    lenient = Model(_sliding_window_config("mode"))
    lenient.fit(data=df)
    assert set(lenient.importance(kind="shap")) == {"num", "cat"}
