"""Boundary pin for ``RefitTrainer`` pipeline fit (H-0085 / #208).

H-0085 unifies the pipeline fit boundary to the **full outer-train** set — for
``RefitTrainer`` that is the whole dataset, fitted **exactly once**. The previous
implementation fitted the pipeline on the inner-train slice only and then fitted
a *second* pipeline on all data for ``pipeline_state`` (a double fit whose
early-stopping boundary was stricter than the CV folds it mirrors).

These tests fail closed on a regression to either the old inner-train boundary
or the double fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lizyml.estimators.lgbm import LGBMAdapter
from lizyml.features.encoders.categorical_encoder import UnseenPolicy
from lizyml.features.pipelines_native import NativeFeaturePipeline
from lizyml.training.inner_valid import HoldoutInnerValid, TimeHoldoutInnerValid
from lizyml.training.refit_trainer import RefitTrainer


class _FitCountingPipeline(NativeFeaturePipeline):
    """Records the row count of every ``fit`` call into a shared log."""

    def __init__(self, log: list[int], unseen_policy: UnseenPolicy = "mode") -> None:
        super().__init__(unseen_policy=unseen_policy)
        self._log = log

    def fit(self, X: pd.DataFrame, y: pd.Series) -> _FitCountingPipeline:
        self._log.append(len(X))
        super().fit(X, y)
        return self


def _reg_data(n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(rng.normal(size=n), name="target")
    return X, y


def test_refit_fits_pipeline_once_on_full_data() -> None:
    """Pipeline is fit exactly once, on the full dataset (no inner-train fit,
    no second full-data refit)."""
    X, y = _reg_data(n=200)
    fit_rows: list[int] = []
    trainer = RefitTrainer(
        inner_valid=HoldoutInnerValid(ratio=0.15, random_state=0),
        pipeline_factory=lambda: _FitCountingPipeline(fit_rows),
        estimator_factory=lambda: LGBMAdapter(
            task="regression",
            params={"n_estimators": 30, "learning_rate": 0.1},
            early_stopping_rounds=5,
            random_state=0,
        ),
        task="regression",
    )
    result = trainer.fit(X, y)

    # H-0085: single fit, on all rows — matches CVTrainer's outer-train boundary.
    assert fit_rows == [len(X)]
    assert result.pipeline_state is not None
    assert result.best_iteration is not None


def test_refit_pipeline_sees_inner_valid_categories() -> None:
    """A category present only in the inner-valid tail must be known to the
    pipeline — proving the fit boundary is the full data, not inner-train.

    With ``unseen_policy="error"`` the old inner-train boundary raised while
    transforming the full ``X`` (the tail category was unseen); the unified
    full-data boundary completes without error.
    """
    n = 200
    rng = np.random.default_rng(1)
    cat = np.array(["common"] * n, dtype=object)
    cat[-1] = "TAIL_ONLY"  # last row → falls in the time-holdout inner-valid tail
    X = pd.DataFrame({"num": rng.normal(size=n), "cat": cat})
    y = pd.Series(rng.normal(size=n), name="target")

    trainer = RefitTrainer(
        inner_valid=TimeHoldoutInnerValid(ratio=0.1),
        pipeline_factory=lambda: NativeFeaturePipeline(unseen_policy="error"),
        estimator_factory=lambda: LGBMAdapter(
            task="regression",
            params={"n_estimators": 20},
            early_stopping_rounds=5,
            random_state=0,
        ),
        task="regression",
    )

    # Must not raise: the full-data pipeline has learned "TAIL_ONLY".
    result = trainer.fit(X, y)
    assert result.pipeline_state is not None
    assert "cat" in result.pipeline_state["categorical_cols"]
