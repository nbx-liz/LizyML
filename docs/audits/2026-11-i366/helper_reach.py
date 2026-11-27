"""H-0131 census: every registered key constructs through its public helper today.

For each metric key, calls ``get_metric(key)`` with no keyword arguments, and
``get_metrics_for_task([key], task)`` for each task that lists the key. For each
calibrator key, calls ``get_calibrator(key)`` with no params. Also records what
the Facade's calibration preflight (``_model_factories.py`` around line 1265)
does today for an unregistered method with non-empty params, and the order of
the ``Available`` list in the unknown-name messages.

Run from the repository root: ``.venv/bin/python docs/audits/2026-11-i366/helper_reach.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import lizyml.metrics.registry as metric_helpers  # noqa: E402
from lizyml.calibration.registry import get_calibrator  # noqa: E402
from lizyml.core.exceptions import LizyMLError  # noqa: E402
from lizyml.core.registries import CalibratorRegistry, MetricRegistry  # noqa: E402


def outcome(fn: object, *args: object, **kwargs: object) -> str:
    try:
        result = fn(*args, **kwargs)  # type: ignore[operator]
    except LizyMLError as exc:
        return f"LizyMLError {exc.code.value}: {exc.user_message}"
    except Exception as exc:  # noqa: BLE001 -- a census records every outcome
        return f"{type(exc).__name__}: {exc}"
    if isinstance(result, list):
        return "ok " + ",".join(type(r).__name__ for r in result)
    return f"ok {type(result).__name__}"


def main() -> int:
    print("## metrics: get_metric(key)")
    for key in MetricRegistry.keys():  # noqa: SIM118 -- a registry, not a dict
        print(f"  {key:16s} {outcome(metric_helpers.get_metric, key)}")
    print("## metrics: get_metrics_for_task([key], task)")
    for task, keys in metric_helpers._TASK_METRICS.items():  # noqa: SLF001
        for key in sorted(keys):
            res = outcome(metric_helpers.get_metrics_for_task, [key], task)
            print(f"  {task:10s} {key:16s} {res}")
    print("## calibrators: get_calibrator(key)")
    for key in CalibratorRegistry.keys():  # noqa: SIM118 -- a registry, not a dict
        print(f"  {key:16s} {outcome(get_calibrator, key)}")
    print("## unknown names (message order)")
    print(f"  metric:     {outcome(metric_helpers.get_metric, 'nope')}")
    print(f"  calibrator: {outcome(get_calibrator, 'nope')}")
    print("## Facade preflight with an unregistered method and params")
    from lizyml.core._model_factories import check_calibration_param_names

    class _Cfg:
        method = "nope"
        params = {"a": 1}

    print(f"  {outcome(check_calibration_param_names, _Cfg())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
