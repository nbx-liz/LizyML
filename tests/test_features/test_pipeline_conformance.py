"""A pipeline written against ``BaseFeaturePipeline`` works end to end (H-0104, #259).

The base class declares four abstract methods. The predict path used to call a
fifth, ``transform_with_warnings``, which only ``NativeFeaturePipeline`` had, so
a pipeline implemented exactly as declared trained and then raised
``AttributeError`` at the first ``predict``.

There is no public way to register a provider, so a custom pipeline reaches
``Model`` only through a provider's ``build_pipeline_factory`` -- the route a new
in-tree estimator takes. These tests inject the minimal pipeline there; that is
the closest stand-in for the extension point, not a public API.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.model import Model
from lizyml.estimators.lgbm.provider import LGBMProvider
from lizyml.features.pipeline_base import BaseFeaturePipeline
from lizyml.features.pipelines_native import NativeFeaturePipeline
from tests._helpers import make_config


class MinimalPipeline(BaseFeaturePipeline):
    """Implements only the four abstract methods, and checks no columns itself.

    ``get_state`` deliberately omits ``categorical_cols``: the key is optional
    (H-0104 decision 2), and the trainers must treat its absence as "no
    categorical columns".
    """

    def __init__(self) -> None:
        self._cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MinimalPipeline:
        self._cols = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Positional on purpose: no name lookup, so a missing or extra column
        # would pass through here unnoticed. Catching that is the facade's job.
        return X.copy()

    def get_state(self) -> dict[str, Any]:
        return {"cols": list(self._cols)}

    def load_state(self, state: dict[str, Any]) -> MinimalPipeline:
        self._cols = list(state["cols"])
        return self


@pytest.fixture
def minimal_pipeline(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Make the LightGBM provider hand out ``MinimalPipeline``."""

    def factory(self: LGBMProvider, *args: Any, **kwargs: Any) -> Any:
        return MinimalPipeline

    monkeypatch.setattr(LGBMProvider, "build_pipeline_factory", factory)
    yield


def _numeric_df(n: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["target"] = df["a"] * 2.0 + rng.normal(scale=0.1, size=n)
    return df


def _fitted(df: pd.DataFrame) -> Model:
    model = Model(make_config("regression", n_estimators=10))
    model.fit(data=df)
    return model


@pytest.mark.usefixtures("minimal_pipeline")
def test_minimal_pipeline_survives_fit_and_predict() -> None:
    df = _numeric_df()
    model = _fitted(df)
    assert model.fit_result.categorical_features == []

    result = model.predict(df.drop(columns=["target"]).iloc[:5])
    assert len(result.pred) == 5
    assert result.warnings == []


@pytest.mark.usefixtures("minimal_pipeline")
def test_minimal_pipeline_survives_shap() -> None:
    df = _numeric_df()
    model = _fitted(df)

    result = model.predict(df.drop(columns=["target"]).iloc[:5], return_shap=True)
    assert result.shap_values is not None
    assert result.shap_values.shape == (5, 2)

    importance = model.importance(kind="shap")
    assert set(importance) == {"a", "b"}


@pytest.mark.usefixtures("minimal_pipeline")
def test_facade_refuses_missing_columns_for_any_pipeline() -> None:
    """The pipeline never checks columns; the refusal must come from the facade."""
    df = _numeric_df()
    model = _fitted(df)

    with pytest.raises(LizyMLError) as exc:
        model.predict(df.drop(columns=["target", "b"]).iloc[:5])
    assert exc.value.code is ErrorCode.DATA_SCHEMA_INVALID
    assert "b" in str(exc.value)


class ReportingPipeline(MinimalPipeline):
    """Overrides ``transform_with_warnings`` to report its own correction."""

    SENTINEL = "ReportingPipeline: clipped 1 value"

    def transform_with_warnings(
        self, X: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        return self.transform(X), [self.SENTINEL]


def test_a_pipeline_reports_its_own_warnings_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The facade adds its column warnings; it must not drop or rewrite the
    pipeline's own."""

    def factory(self: LGBMProvider, *args: Any, **kwargs: Any) -> Any:
        return ReportingPipeline

    monkeypatch.setattr(LGBMProvider, "build_pipeline_factory", factory)
    df = _numeric_df()
    model = _fitted(df)

    clean = model.predict(df.drop(columns=["target"]).iloc[:5])
    assert clean.warnings == [ReportingPipeline.SENTINEL]

    drifted = model.predict(df.drop(columns=["target"]).iloc[:5].assign(extra=0.0))
    assert len(drifted.warnings) == 2
    assert ReportingPipeline.SENTINEL in drifted.warnings
    assert any(
        "extra" in w for w in drifted.warnings if w != ReportingPipeline.SENTINEL
    )


@pytest.mark.parametrize("pipeline", ["custom", "native"])
def test_extra_columns_warn_exactly_once(
    pipeline: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One extra column, one warning -- whichever pipeline runs.

    The facade selects the training columns before the pipeline sees the frame,
    so ``NativeFeaturePipeline``'s own check must not report the column again.
    """
    if pipeline == "custom":

        def factory(self: LGBMProvider, *args: Any, **kwargs: Any) -> Any:
            return MinimalPipeline

        monkeypatch.setattr(LGBMProvider, "build_pipeline_factory", factory)
    df = _numeric_df()
    model = _fitted(df)
    X_new = df.drop(columns=["target"]).iloc[:5].assign(surprise_col=1.0)

    result = model.predict(X_new)
    assert len(result.warnings) == 1, result.warnings
    assert "surprise_col" in result.warnings[0]
    # The native pipeline still reports the drift when used on its own.
    if pipeline == "native":
        native = NativeFeaturePipeline().fit(df.drop(columns=["target"]), df["target"])
        _, own = native.transform_with_warnings(X_new)
        assert len(own) == 1 and "surprise_col" in own[0]
