"""Measure the population a calibration-params gate for platt and beta would touch.

PR 3c (H-0100) extends the calibration.params name check from isotonic to every
calibrator. For platt and beta that is an allow condition: a config whose
calibration.params names something outside the declared surface is refused
before training, where today it is accepted and ignored. The Change Gate asks
for a measured firing rate before implementation.

The check runs at the entrance of fit and tune through
check_calibration_param_names, so that is where this plugin listens: every
calibrated config the shipped suite builds passes through it, whatever test
constructed it.

Run from the repository root, with this directory on PYTHONPATH::

    PYTHONPATH=docs/audits/2026-09-defect-discovery/instruments \\
    CALIBRATION_PARAMS_OUT=/path/to/report.txt \\
    uv run pytest tests -q --no-cov -p calibration_params_firing_rate

It records, per method, how many calibrated configs arrived and how many carried
a non-empty calibration.params, with the test that built each of the latter.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Any

COUNTS: Counter[str] = Counter()
NON_EMPTY: list[tuple[str, str, list[str]]] = []


def _test_id() -> str:
    return os.environ.get("PYTEST_CURRENT_TEST", "?").split(" ")[0]


def _record(calibration_cfg: Any) -> None:
    if calibration_cfg is None:
        return
    method = str(getattr(calibration_cfg, "method", "?"))
    params = getattr(calibration_cfg, "params", None) or {}
    COUNTS[f"{method}: calibrated configs at the entrance"] += 1
    if params:
        COUNTS[f"{method}: with non-empty calibration.params"] += 1
        NON_EMPTY.append((_test_id(), method, sorted(params)))


def pytest_configure(config):  # noqa: ARG001
    from lizyml.core import _model_factories, _model_tuning, model

    original = _model_factories.check_calibration_param_names

    def wrapped(calibration_cfg):
        try:
            _record(calibration_cfg)
        except Exception:  # noqa: BLE001 -- measuring must not change behaviour
            COUNTS["recording raised"] += 1
        return original(calibration_cfg)

    # The facade imports the function by name, so patch every binding that calls it.
    _model_factories.check_calibration_param_names = wrapped
    for module in (model, _model_tuning):
        if hasattr(module, "check_calibration_param_names"):
            module.check_calibration_param_names = wrapped


def pytest_terminal_summary(terminalreporter, *args, **kwargs):  # noqa: ARG001
    lines = [f"{key}: {value}" for key, value in sorted(COUNTS.items())]
    by_method = Counter(method for _, method, _ in NON_EMPTY)
    lines.append("")
    lines.append(f"NON-EMPTY calibration.params by method: {dict(by_method)}")
    for test, method, names in NON_EMPTY:
        if method in ("platt", "beta"):
            lines.append(f"  PLATT/BETA  {method}  {names}  {test}")
    text = "\n".join(lines)
    terminalreporter.write_line("=== CALIBRATION PARAMS MEASUREMENT ===")
    terminalreporter.write_line(text)
    out = os.environ.get("CALIBRATION_PARAMS_OUT")
    if out:
        with open(out, "w") as handle:
            handle.write(text + "\n")
