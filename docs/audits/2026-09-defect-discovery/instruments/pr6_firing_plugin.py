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


def _proba_reason(y_true, y_pred) -> str | None:  # noqa: ANN001
    """H-0106 decision 3's rule, exactly: None when *y_pred* is a probability."""
    try:
        p = np.asarray(y_pred, dtype=float)
    except (TypeError, ValueError):
        return "not numeric"
    if not np.all(np.isfinite(p)):
        return "non-finite"
    if p.size and (p.min() < 0.0 or p.max() > 1.0):
        return f"range {p.min():.3g}..{p.max():.3g}"
    if p.ndim == 1 and len(np.unique(np.asarray(y_true))) > 2:
        return "1-D for >2 classes"
    return None


def _wrap_metric(cls: type) -> None:
    real_call = cls.__call__

    def call(self, y_true, y_pred):  # noqa: ANN001
        if not self.needs_proba:
            return real_call(self, y_true, y_pred)
        _calls["proba_metric"] += 1
        # Decided before the real call, so an input the metric then fails on
        # is still recorded (ok=False).
        reason = _proba_reason(y_true, y_pred)
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


# 3. config_version: a config that reaches Model, or that pydantic validation
#    produces, holding a version H-0106's check rejects. Denominators count
#    COMPLETED calls only; a call the loader already refused is not a success
#    the new positions could refuse.
from lizyml.config.loader import SUPPORTED_CONFIG_VERSIONS  # noqa: E402
from lizyml.config.schema import LizyMLConfig  # noqa: E402


def _version_rejected(value: object) -> bool:
    """H-0106 decision 4: int() coercion (so False -> 0, True -> 1), then membership."""
    try:
        return int(value) not in SUPPORTED_CONFIG_VERSIONS  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return True


_calls["model_init_completed"] = 0
_calls["config_validated_completed"] = 0
_real_model_init = _model_mod.Model.__init__


def _counting_model_init(self, config, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
    _real_model_init(self, config, *args, **kwargs)
    _calls["model_init_completed"] += 1
    version = getattr(self._cfg, "config_version", None)
    if _version_rejected(version):
        _write("CONFIG_VERSION", {"entry": "Model.__init__",
                                  "instance": isinstance(config, LizyMLConfig),
                                  "version": repr(version)}, ok=True)


_model_mod.Model.__init__ = _counting_model_init


def _record_validated(entry: str, cfg: LizyMLConfig) -> None:
    _calls["config_validated_completed"] += 1
    if _version_rejected(cfg.config_version):
        _write("CONFIG_VERSION", {"entry": entry, "version": repr(cfg.config_version)}, ok=True)


# Every pydantic validation entry point of the schema: the constructor and the
# three model_validate* classmethods (model_validate does not call __init__).
_real_init = LizyMLConfig.__init__


def _counting_init(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
    _real_init(self, *args, **kwargs)
    _record_validated("LizyMLConfig.__init__", self)


LizyMLConfig.__init__ = _counting_init  # type: ignore[method-assign]

for _name in ("model_validate", "model_validate_json", "model_validate_strings"):
    _real = getattr(LizyMLConfig, _name).__func__  # type: ignore[attr-defined]

    def _make(real, name):  # noqa: ANN001, ANN202
        def counting(cls, obj, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            out = real(cls, obj, *args, **kwargs)
            _record_validated(name, out)
            return out

        return classmethod(counting)

    setattr(LizyMLConfig, _name, _make(_real, _name))


def _controls() -> None:
    """Positive and negative controls: the predicates fire where they must."""
    assert _proba_reason([0, 1], np.array([0.1, 0.9], dtype=object)) is None
    assert _proba_reason([0, 1], [0.0, 1.0]) is None
    assert _proba_reason([0, 1], [0.1, 1.5]) is not None
    assert _proba_reason([0, 1], ["a", "b"]) == "not numeric"
    assert _proba_reason([0, 1], [np.nan, 0.5]) == "non-finite"
    assert _proba_reason([0, 1, 2], [0.1, 0.2, 0.3]) == "1-D for >2 classes"
    assert _numeric_dtype(pd.Series([1.0]).dtype)
    assert _numeric_dtype(pd.Series([1]).astype("Int64").dtype)
    assert not _numeric_dtype(pd.Series(["a"]).dtype)
    assert not _numeric_dtype(pd.Series([1]).astype("int64[pyarrow]").dtype)
    assert not _version_rejected(1)
    assert _version_rejected(2)
    assert _version_rejected(False)
    assert not _version_rejected(True)


def pytest_configure(config):  # noqa: ANN001, ARG001
    _controls()
    if os.path.exists(LOG):
        os.remove(LOG)


def pytest_unconfigure(config):  # noqa: ANN001, ARG001
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "TOTALS", **_calls}) + "\n")
