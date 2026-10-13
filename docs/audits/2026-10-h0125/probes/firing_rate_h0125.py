"""Firing rate of the H-0125 rules (#307, #313, general tune value check), on the suite.

Loaded as a pytest plugin (``-p firing_rate_h0125``) with
``docs/audits/2026-10-h0125/probes`` on ``PYTHONPATH``. Without changing any
outcome it records one row per event:

- ``obj``: every objective value ``check_objective_compatible`` sees (the
  facade's ``check_param_values`` and ``LGBMAdapter._build_params``). Reason
  ``xentlambda`` when the task is binary and the value is
  ``cross_entropy_lambda`` (#307); ``refused_today`` when today's rule
  already raises.
- ``metric``: every metric entry ``resolve_metrics`` sees for model params.
  Reason ``native_params:<name>`` when the name is not a feval metric of the
  task and its params are non-empty (#313).
- ``dim``: every ``objective`` / ``metric`` dimension (any LightGBM spelling)
  in a round's effective space that ``validate_tuning_dimensions`` sees, with
  the choices the proposed pre-study check refuses: ``bad_objective:<v>``
  (not in the task's accepted set after #307) or ``bad_metric:<v>`` (refused by
  ``resolve_metrics`` today, or native params).
- ``tune``: one row per ``Model.tune`` call; reason ``trial_refusal`` when an
  objective / metric check raised inside it (today such a refusal fails the
  trial instead of the call).

Writes ``<kind>\t<nodeid>\t<detail>\t<reasons or "">`` to ``$FIRING_RATE_H0125_OUT``.
"""

from __future__ import annotations

import functools
import os
from typing import Any

import pytest

from lizyml.core import _model_tuning, _tuning_admission
from lizyml.core.exceptions import LizyMLError
from lizyml.estimators.lgbm import adapter as _adapter
from lizyml.estimators.lgbm import metric_bridge, param_validation
from lizyml.estimators.lgbm.defaults import TASK_COMPATIBLE_OBJECTIVES
from lizyml.estimators.lgbm.param_names import LGBM_CANONICAL_NAME
from lizyml.metrics.registry import parse_metric_entries

_current = {"nodeid": "<none>"}
_tune_depth = {"n": 0, "refused": False}
_rows: list[str] = []

_check_obj = param_validation.check_objective_compatible
_resolve = param_validation.resolve_metrics
_validate = _tuning_admission.validate_tuning_dimensions
_tune = _model_tuning.ModelTuningMixin.tune


def _accepted_after(task: str) -> frozenset[str]:
    accepted = TASK_COMPATIBLE_OBJECTIVES.get(task, frozenset())
    return accepted - {"cross_entropy_lambda"} if task == "binary" else accepted


def _native_param_reasons(metrics: Any, task: str) -> list[str]:
    try:
        entries = metrics if isinstance(metrics, list) else [metrics]
        parsed = parse_metric_entries([e for e in entries if e])
    except Exception:  # noqa: BLE001 -- recording only; the real call decides
        return []
    feval = metric_bridge._FEVAL_METRICS.get(task, frozenset())
    return [f"native_params:{n}" for n, kw in parsed if n not in feval and kw]


def _wrapped_check_obj(task: Any, objective: Any) -> Any:
    reasons = []
    # Never run the value's own __eq__ / __repr__ (tests pass hostile values).
    text = str.__str__(objective) if isinstance(objective, str) else None
    shown = repr(text) if text is not None else f"<{type(objective).__name__}>"
    if task == "binary" and text == "cross_entropy_lambda":
        reasons.append("xentlambda")
    try:
        return _check_obj(task, objective)
    except LizyMLError:
        reasons.append("refused_today")
        if _tune_depth["n"]:
            _tune_depth["refused"] = True
        raise
    finally:
        tag = "tune" if _tune_depth["n"] else "fit"
        _rows.append(f"obj\t{_current['nodeid']}\t{tag}:{task}:{shown}\t{';'.join(reasons)}")


def _wrapped_resolve(metrics: Any, task: Any, num_class: Any = None) -> Any:
    reasons = _native_param_reasons(metrics, task)
    try:
        return _resolve(metrics, task, num_class)
    except LizyMLError:
        reasons.append("refused_today")
        if _tune_depth["n"]:
            _tune_depth["refused"] = True
        raise
    finally:
        tag = "tune" if _tune_depth["n"] else "fit"
        _rows.append(f"metric\t{_current['nodeid']}\t{tag}:{task}:{metrics!r}\t{';'.join(reasons)}")


def _dim_reasons(dim: Any, task: str) -> list[str] | None:
    canonical = LGBM_CANONICAL_NAME.get(dim.name)
    if canonical not in ("objective", "metric"):
        return None
    values = list(getattr(dim, "choices", None) or [])
    out = []
    for v in values:
        if canonical == "objective" and v not in _accepted_after(task):
            out.append(f"bad_objective:{v!r}")
        if canonical == "metric":
            try:
                _resolve([v] if not isinstance(v, list) else v, task, None)
            except LizyMLError:
                out.append(f"bad_metric:{v!r}")
            except Exception as exc:  # noqa: BLE001 -- recording only
                out.append(f"bad_metric_raw:{type(exc).__name__}:{v!r}")
            out.extend(_native_param_reasons(v, task))
    return out


def _wrapped_validate(provider: Any, space: Any, smart: Any, task: Any, *args: Any, **kwargs: Any) -> Any:
    for dim in space:
        reasons = _dim_reasons(dim, task)
        if reasons is not None:
            _rows.append(f"dim\t{_current['nodeid']}\t{dim.name}\t{';'.join(reasons)}")
    return _validate(provider, space, smart, task, *args, **kwargs)


@functools.wraps(_tune)
def _wrapped_tune(self: Any, *args: Any, **kwargs: Any) -> Any:
    outer = _tune_depth["n"] == 0
    if outer:
        _tune_depth["refused"] = False
    _tune_depth["n"] += 1
    try:
        return _tune(self, *args, **kwargs)
    finally:
        _tune_depth["n"] -= 1
        if outer:
            reason = "trial_refusal" if _tune_depth["refused"] else ""
            _rows.append(f"tune\t{_current['nodeid']}\t-\t{reason}")


param_validation.check_objective_compatible = _wrapped_check_obj  # type: ignore[assignment]
_adapter._check_objective_compatible = _wrapped_check_obj  # type: ignore[attr-defined]
param_validation.resolve_metrics = _wrapped_resolve  # type: ignore[assignment]
_tuning_admission.validate_tuning_dimensions = _wrapped_validate  # type: ignore[assignment]
_model_tuning.ModelTuningMixin.tune = _wrapped_tune  # type: ignore[method-assign]


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    _current["nodeid"] = item.nodeid


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    out = os.environ.get("FIRING_RATE_H0125_OUT")
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(_rows) + "\n")
