"""H-0136 population probe: what the proposed rule would refuse at every export exit.

A pytest plugin (``-p h0136_export_census`` with this directory on
``PYTHONPATH``), active only when ``H0136_CENSUS_LOG`` is set. It wraps the
three exits and the fit warning, evaluates the prototype rule
(``h0136_rule.py``) on what each would save, then runs the real call
unchanged and records its outcome. One JSON record per call:

- ``surface``: ``Model.export`` / ``persistence.export`` (both through
  ``check_category_values``), ``export_code`` (the exit itself), or
  ``fit_warning``. After H-0136, ``export_code`` also calls
  ``check_category_values``; that inner call is recorded apart as
  ``export_code (check)``, so count exits on ``export_code``.
  ``fit_warning`` wraps the warning fit gives before CV
  (``warn_unsaveable_category_values``); H-0136's own warning comes after the
  fit (``warn_saved_state_changes``) and is not recorded here;
- ``today``: ``ok`` or the error code the real call raised (for
  ``fit_warning``: whether today's warning fires);
- ``breaks``: the prototype's breaks, each with the state, column,
  serializer and kind; ``original_fails``: columns whose original state
  already fails to transform (not a serializer difference);
- for ``export_code``: ``today_check``, what ``check_category_values`` (which
  export_code does not call today) answers for the same model.

A state the prototype cannot read is recorded with ``rule_error``, never
skipped.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

_LOG = os.environ.get("H0136_CENSUS_LOG")


def _write(record: dict[str, Any]) -> None:
    record.update(pid=os.getpid(), test=os.environ.get("PYTEST_CURRENT_TEST", ""))
    with open(_LOG, "a", encoding="utf-8") as fh:  # type: ignore[arg-type]
        fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def _outcome(exc: BaseException) -> str:
    return getattr(getattr(exc, "code", None), "name", type(exc).__name__)


def _model_rule(fit_result: Any, refit_result: Any, declared: Any) -> dict[str, Any]:
    from h0136_rule import declared_breaks, state_breaks

    from lizyml.persistence.exporter import _saved_pipeline_states

    try:
        saved = _saved_pipeline_states(fit_result, refit_result)
    except Exception as exc:  # noqa: BLE001 -- recorded
        return {"rule_error": _outcome(exc), "breaks": [], "original_fails": []}
    breaks: list[dict[str, Any]] = []
    fails: list[dict[str, Any]] = []
    for name, state, _ in saved:
        b, f = state_breaks(state)
        breaks += [{"state": name, **x} for x in b]
        fails += [{"state": name, **x} for x in f]
    breaks += [{"state": "declared categories", **x} for x in declared_breaks(declared)]
    return {"breaks": breaks, "original_fails": fails}


def pytest_configure(config: Any) -> None:
    if not _LOG:
        return
    sys.path.insert(0, os.path.dirname(__file__))
    import lizyml.core.model as model_mod
    import lizyml.persistence.exporter as exporter
    from h0136_rule import state_breaks

    from lizyml.core._model_persistence import ModelPersistenceMixin

    original_check = exporter.check_category_values

    def check_category_values(
        fit_result: Any, refit_result: Any, declared: Any = None, **kwargs: Any
    ) -> None:
        if "surface" in kwargs:  # after H-0136: every exit names itself
            # export_code's own call keeps its "export_code" record below.
            surface = {
                "lizyml.persistence.export": "persistence.export",
                "export_code": "export_code (check)",
            }.get(kwargs["surface"], kwargs["surface"])
        else:  # before H-0136: Model.export and persistence.export only
            caller = sys._getframe(1).f_globals.get("__name__", "")
            surface = (
                "Model.export" if caller.endswith("_model_persistence") else "persistence.export"
            )
        try:
            rule = _model_rule(fit_result, refit_result, declared)
        except Exception as exc:  # noqa: BLE001 -- recorded, never raised into the test
            rule = {"rule_error": _outcome(exc), "breaks": [], "original_fails": []}
        record = {"surface": surface, **rule}
        try:
            original_check(fit_result, refit_result, declared, **kwargs)
        except Exception as exc:
            record["today"] = _outcome(exc)
            _write(record)
            raise
        record["today"] = "ok"
        _write(record)

    exporter.check_category_values = check_category_values

    original_export_code = ModelPersistenceMixin.export_code

    def export_code(self: Any, path: Any) -> Any:
        record: dict[str, Any] = {"surface": "export_code"}
        try:
            state = self._get_fit_state()
            refit = self._require_refit()
            record.update(_model_rule(state.fit_result, refit, state.declared_categories))
            try:
                original_check(state.fit_result, refit, state.declared_categories)
                record["today_check"] = "ok"
            except Exception as exc:  # noqa: BLE001 -- recorded
                record["today_check"] = _outcome(exc)
        except Exception as exc:  # noqa: BLE001 -- the real call reports it
            record["rule_error"] = _outcome(exc)
        try:
            out = original_export_code(self, path)
        except Exception as exc:
            record["today"] = _outcome(exc)
            _write(record)
            raise
        record["today"] = "ok"
        _write(record)
        return out

    ModelPersistenceMixin.export_code = export_code  # type: ignore[method-assign]

    original_warn = model_mod.warn_unsaveable_category_values

    def warn_unsaveable_category_values(state: Any) -> None:
        record: dict[str, Any] = {"surface": "fit_warning"}
        try:
            b, f = state_breaks(state)
            record.update(breaks=[{"state": "fit", **x} for x in b], original_fails=f)
        except Exception as exc:  # noqa: BLE001 -- recorded
            record["rule_error"] = _outcome(exc)
        record["today"] = "warns" if exporter.unsaveable_category_values(state) else "silent"
        _write(record)
        original_warn(state)

    model_mod.warn_unsaveable_category_values = warn_unsaveable_category_values
