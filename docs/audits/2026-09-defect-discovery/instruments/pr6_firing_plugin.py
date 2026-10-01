"""pytest plugin: count where PR 6's two new refusals WOULD fire, without raising.

1. INCOMPATIBLE_COLUMNS: a column numeric (or bool) at fit arrives non-numeric at predict.
2. METRIC_REQUIRES_PROBA: a needs_proba metric receives values that are not
   probabilities -- non-finite, outside [0, 1], or 1-D when y_true has more than
   two classes.

Each event is recorded with the test id and whether the original call then
succeeded, so a refusal of a call that used to work is visible.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from lizyml.core import _model_predict
from lizyml.metrics.base import BaseMetric

LOG = os.environ.get("PR6_FIRING_LOG", "/tmp/claude-1000/pr6_firing.jsonl")
_calls = {"predict": 0, "proba_metric": 0}


def _write(kind: str, detail: dict, ok: bool) -> None:
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"kind": kind, "ok_before": ok, "detail": detail,
                            "test": os.environ.get("PYTEST_CURRENT_TEST", "")}) + "\n")


def _numeric_dtype(dt: object) -> bool:
    # The rule H-0106 adopts: the dtype's scalar type is a numpy integer,
    # floating or bool type, excluding timedelta64 and longdouble -- the set
    # LightGBM accepts (measured 0 disagreements over 33 dtypes).
    t = getattr(dt, "type", None)
    return bool(
        isinstance(t, type)
        and issubclass(t, (np.integer, np.floating, np.bool_))
        and not issubclass(t, (np.timedelta64, np.longdouble))
    )


def _is_numeric(dtype_name: str) -> bool:
    try:
        dt = pd.api.types.pandas_dtype(dtype_name)
    except TypeError:
        return False
    return _numeric_dtype(dt)


_real_run_predict = _model_predict.run_predict


def _counting_run_predict(*args, **kwargs):  # noqa: ANN002, ANN003
    _calls["predict"] += 1
    fit_result = kwargs.get("fit_result")
    X = kwargs.get("X")
    bad = []
    if fit_result is not None and isinstance(X, pd.DataFrame):
        for col, dtype_name in (fit_result.dtypes or {}).items():
            if col in X.columns and _is_numeric(dtype_name):
                got = X[col].dtype
                if not _numeric_dtype(got):
                    bad.append({"column": col, "fit": dtype_name, "predict": str(got)})
    try:
        out = _real_run_predict(*args, **kwargs)
    except Exception:
        if bad:
            _write("INCOMPATIBLE_COLUMNS", {"cols": bad}, ok=False)
        raise
    if bad:
        _write("INCOMPATIBLE_COLUMNS", {"cols": bad}, ok=True)
    return out


_model_predict.run_predict = _counting_run_predict
import lizyml.core.model as _model_mod  # noqa: E402

_model_mod.run_predict = _counting_run_predict


def _wrap_metric(cls: type) -> None:
    real_call = cls.__call__

    def call(self, y_true, y_pred):  # noqa: ANN001
        if not self.needs_proba:
            return real_call(self, y_true, y_pred)
        _calls["proba_metric"] += 1
        p = np.asarray(y_pred, dtype=float) if np.asarray(y_pred).dtype != object else None
        reason = None
        if p is None:
            reason = "object"
        elif not np.all(np.isfinite(p)):
            reason = "non-finite"
        elif p.size and (p.min() < 0.0 or p.max() > 1.0):
            reason = f"range {p.min():.3g}..{p.max():.3g}"
        elif p.ndim == 1 and len(np.unique(np.asarray(y_true))) > 2:
            reason = "1-D for >2 classes"
        try:
            out = real_call(self, y_true, y_pred)
        except Exception:
            if reason:
                _write("METRIC_REQUIRES_PROBA", {"metric": self.name, "reason": reason}, ok=False)
            raise
        if reason:
            _write("METRIC_REQUIRES_PROBA", {"metric": self.name, "reason": reason}, ok=True)
        return out

    cls.__call__ = call


def _all_subclasses(cls: type) -> list[type]:
    out = []
    for sub in cls.__subclasses__():
        out.append(sub)
        out.extend(_all_subclasses(sub))
    return out


import lizyml.metrics  # noqa: E402,F401  (registers every metric)

for _cls in _all_subclasses(BaseMetric):
    if "__call__" in vars(_cls):
        _wrap_metric(_cls)


# 3. config_version: a LizyMLConfig that reaches Model, or comes out of
#    model_validate, holding a version outside the supported set. Each such
#    success would be refused by H-0106's new check positions.
from lizyml.config.loader import SUPPORTED_CONFIG_VERSIONS  # noqa: E402
from lizyml.config.schema import LizyMLConfig  # noqa: E402

_calls["model_init"] = 0
_calls["model_validate"] = 0
_real_model_init = _model_mod.Model.__init__


def _counting_model_init(self, config, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
    _calls["model_init"] += 1
    _real_model_init(self, config, *args, **kwargs)
    version = getattr(self._cfg, "config_version", None)
    if version not in SUPPORTED_CONFIG_VERSIONS:
        _write("CONFIG_VERSION", {"entry": "Model.__init__",
                                  "instance": isinstance(config, LizyMLConfig),
                                  "version": repr(version)}, ok=True)


_model_mod.Model.__init__ = _counting_model_init
_real_model_validate = LizyMLConfig.model_validate.__func__  # type: ignore[attr-defined]


def _counting_model_validate(cls, obj, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
    _calls["model_validate"] += 1
    out = _real_model_validate(cls, obj, *args, **kwargs)
    if out.config_version not in SUPPORTED_CONFIG_VERSIONS:
        _write("CONFIG_VERSION", {"entry": "model_validate",
                                  "version": repr(out.config_version)}, ok=True)
    return out


LizyMLConfig.model_validate = classmethod(_counting_model_validate)  # type: ignore[method-assign,assignment]


def pytest_configure(config):  # noqa: ANN001, ARG001
    if os.path.exists(LOG):
        os.remove(LOG)


def pytest_unconfigure(config):  # noqa: ANN001, ARG001
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "TOTALS", **_calls}) + "\n")
