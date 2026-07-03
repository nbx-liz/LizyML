"""Unit tests for the shared plots data helper ``collect_is_data`` (#218).

The non-empty path is also exercised via the ROC / residual plot tests; these
lock the dtype-cast and empty-fold branches directly.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from lizyml import Model
from lizyml.plots._helpers import collect_is_data
from tests._helpers import make_config, make_regression_df


def _empty_fit_result() -> SimpleNamespace:
    return SimpleNamespace(splits=SimpleNamespace(outer=[]), if_pred_per_fold=[])


class TestCollectIsData:
    def test_concatenates_all_fold_train_rows(self) -> None:
        m = Model(make_config("regression", n_splits=2))
        fr = m.fit(data=make_regression_df(n=120, seed=0))
        y = np.asarray(m._y)  # type: ignore[arg-type]

        pred, actual = collect_is_data(fr, y)

        n_train_rows = sum(len(tr) for tr, _ in fr.splits.outer)
        assert pred.shape[0] == n_train_rows
        assert actual.shape[0] == n_train_rows

    def test_dtype_cast_applied(self) -> None:
        m = Model(make_config("regression", n_splits=2))
        fr = m.fit(data=make_regression_df(n=120, seed=0))
        y = np.asarray(m._y)  # type: ignore[arg-type]

        pred, actual = collect_is_data(fr, y, dtype=np.float64)

        assert pred.dtype == np.float64
        assert actual.dtype == np.float64

    def test_empty_folds_return_empty_arrays(self) -> None:
        y = np.array([1.0, 2.0, 3.0])

        pred, actual = collect_is_data(_empty_fit_result(), y)

        assert pred.size == 0
        assert actual.size == 0
        assert pred.dtype == y.dtype
