"""Firing rate of H-0127's two conditions, measured on the suite before implementation.

Loaded as a pytest plugin (``-p firing_rate_h0127``) with this directory on
``PYTHONPATH``. Nothing is changed: each wrapper records, then calls the
original.

1. #338 -- the export refusal and the fit warning. For every column
   ``CategoricalEncoder.fit`` records, the categories LightGBM would receive
   (``list(pd.CategoricalDtype(cats).categories)``, what ``set_categories``
   builds and ``_data_from_pandas`` reads) are checked against LightGBM's JSON
   writer: numpy integer / floating / bool scalars become ``.item()``, numpy
   arrays ``.tolist()``, anything else must be JSON-native. Population: columns
   recorded by ``CategoricalEncoder.fit`` in the suite.
   Also, every call of ``lizyml.persistence.exporter.export``: whether any
   pipeline state it saves holds such a column (the planned refusal), and
   whether the call raised (today's behaviour).
2. #309 -- the upcast. For every column ``transform_with_warnings`` encodes:
   whether the arrival dtype is ``float16`` or ``longdouble`` (when longdouble
   is wider than float64), and whether a value changes under float64.

Writes one tab-separated row per event to ``$FIRING_RATE_H0127_OUT``.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml.features.encoders import categorical_encoder as ce
from lizyml.persistence import exporter

_current = {"nodeid": "<none>"}
_rows: list[str] = []
_WIDE = {np.dtype(np.float16)}
if np.dtype(np.longdouble).itemsize > 8:
    _WIDE.add(np.dtype(np.longdouble))


def _lgb_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        plain = obj.item()
        if isinstance(plain, np.generic):  # np.longdouble: LightGBM recurses
            raise TypeError(type(obj).__name__)
        return plain
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def _unwritable(cats: list[Any]) -> list[str]:
    bad = []
    for value in list(pd.CategoricalDtype(cats).categories):
        try:
            json.dumps(value, default=_lgb_default)
        except (TypeError, ValueError, RecursionError):
            bad.append(type(value).__name__)
    return bad


def _state_columns(state: Any) -> dict[str, list[Any]]:
    if not isinstance(state, dict):
        return {}
    enc = state.get("encoder")
    if not isinstance(enc, dict):
        return {}
    cats = enc.get("categories")
    return cats if isinstance(cats, dict) else {}


_orig_fit = ce.CategoricalEncoder.fit
_orig_transform = ce.CategoricalEncoder.transform_with_warnings
_orig_export = exporter.export


def _fit(self: ce.CategoricalEncoder, X: pd.DataFrame, categorical_cols: list[str]) -> Any:
    out = _orig_fit(self, X, categorical_cols)
    for col, cats in self._categories.items():
        bad = _unwritable(cats)
        _rows.append(f"fit\t{_current['nodeid']}\t{col}\t{','.join(sorted(set(bad)))}")
    return out


def _transform(self: ce.CategoricalEncoder, X: pd.DataFrame) -> Any:
    for col in self._categories:
        if col not in X.columns:
            continue
        series = X[col]
        dtype = series.dtype
        wide = isinstance(dtype, np.dtype) and dtype in _WIDE
        lossy = False
        if wide:
            vals = series.dropna().to_numpy()
            lossy = bool((vals.astype(np.float64).astype(dtype) != vals).any())
        _rows.append(f"transform\t{_current['nodeid']}\t{col}\t{dtype}\t{int(wide)}\t{int(lossy)}")
    return _orig_transform(self, X)


def _export(path: Any, fit_result: Any, refit_result: Any, *args: Any, **kwargs: Any) -> Any:
    states = list(getattr(fit_result, "pipeline_state_per_fold", None) or [])
    states.append(getattr(fit_result, "pipeline_state", None))
    states.append(getattr(refit_result, "pipeline_state", None))
    refuse = sorted(
        {col for st in states for col, cats in _state_columns(st).items() if _unwritable(cats)}
    )
    raised = ""
    try:
        return _orig_export(path, fit_result, refit_result, *args, **kwargs)
    except Exception as exc:
        raised = type(exc).__name__
        raise
    finally:
        _rows.append(f"export\t{_current['nodeid']}\t{','.join(refuse)}\t{raised}")


ce.CategoricalEncoder.fit = _fit  # type: ignore[method-assign]
ce.CategoricalEncoder.transform_with_warnings = _transform  # type: ignore[method-assign]
exporter.export = _export


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    _current["nodeid"] = item.nodeid


def pytest_sessionfinish(session: pytest.Session) -> None:
    out = os.environ.get("FIRING_RATE_H0127_OUT")
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(_rows) + "\n")
