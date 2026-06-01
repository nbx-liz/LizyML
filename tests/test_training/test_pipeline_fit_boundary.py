"""Leakage trap: the feature pipeline must be fit on the train fold only.

The existing guard (``test_leakage_all.py``) only asserts ``pipeline_state is
not None`` — it would still pass if ``pipeline.fit`` were changed to use the
full dataset. This drives ``CVTrainer`` with a *controlled* split and a
categorical value that appears **only in the validation fold**. With
``unseen_policy="error"`` the encoder raises *iff* the pipeline was fit on the
train rows alone (so the valid-only category is genuinely unseen). A pipeline
leaking the full data would have learned the category and would NOT raise — so
this test fails closed on a leakage regression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.artifacts import RunMeta
from lizyml.data.fingerprint import compute as fp_compute
from lizyml.estimators.lgbm import LGBMAdapter
from lizyml.features.pipelines_native import NativeFeaturePipeline
from lizyml.splitters.base import BaseSplitter
from lizyml.training.cv_trainer import CVTrainer
from lizyml.training.inner_valid import NoInnerValid


class _FixedSplitter(BaseSplitter):
    """Yields a single, caller-controlled ``(train_idx, valid_idx)`` fold."""

    def __init__(self, train_idx: list[int], valid_idx: list[int]) -> None:
        self._train = np.asarray(train_idx, dtype=np.intp)
        self._valid = np.asarray(valid_idx, dtype=np.intp)

    def split(self, n_samples, y=None, groups=None):  # type: ignore[override]
        yield self._train, self._valid


def _data_with_rare_category() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 40
    cat = np.array(["common"] * n, dtype=object)
    cat[0] = "RARE"  # the single valid-fold-only category
    cat[1] = "RARE"
    X = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "cat": cat,
        }
    )
    y = pd.Series(rng.normal(size=n), name="target")
    return X, y


def _run_cv(X: pd.DataFrame, y: pd.Series, splitter: BaseSplitter):
    trainer = CVTrainer(
        outer_splitter=splitter,
        inner_valid=NoInnerValid(),
        pipeline_factory=lambda: NativeFeaturePipeline(unseen_policy="error"),
        estimator_factory=lambda: LGBMAdapter(
            task="regression",
            params={"n_estimators": 10},
            random_state=0,
        ),
        task="regression",
    )
    rm = RunMeta(
        lizyml_version="0.0.0",
        python_version="3.11",
        deps_versions={},
        config_normalized={},
        config_version=1,
        run_id="trap",
        timestamp="2026-01-01T00:00:00",
    )
    return trainer.fit(
        X, y, data_fingerprint=fp_compute(X, file_path=None), run_meta=rm
    )


def test_valid_only_category_raises_proving_train_only_fit() -> None:
    X, y = _data_with_rare_category()
    # RARE rows (0, 1) are in the validation fold, absent from the train fold.
    splitter = _FixedSplitter(train_idx=list(range(2, 40)), valid_idx=[0, 1])

    with pytest.raises(LizyMLError) as excinfo:
        _run_cv(X, y, splitter)
    assert excinfo.value.code == ErrorCode.DATA_SCHEMA_INVALID
    assert excinfo.value.context.get("column") == "cat"


def test_category_present_in_train_does_not_raise() -> None:
    # Control: the same data, but RARE rows are in the TRAIN fold, so the
    # validation fold has only known categories — no error, fit completes.
    X, y = _data_with_rare_category()
    splitter = _FixedSplitter(
        train_idx=list(range(0, 30)), valid_idx=list(range(30, 40))
    )

    result = _run_cv(X, y, splitter)
    assert result.pipeline_state is not None
