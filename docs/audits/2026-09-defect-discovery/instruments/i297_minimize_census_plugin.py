"""pytest plugin for #297: census of scipy ``minimize`` outcomes in calibration.

Wraps ``scipy.optimize.minimize`` (the calibrators import it inside ``fit``,
so the wrapper is what they get) and records, per call, the calling file, the
test, whether the optimiser reported ``success`` and its message. Calls from
the Platt / Beta calibrators and from generated code are counted separately
from other callers (for example the calibration-params validation probe).

Run from the repo root:

    PYTHONPATH=docs/audits/2026-09-defect-discovery/instruments \
        uv run pytest -p i297_minimize_census_plugin -q

The summary separates calls from a calibrator that carries user
``calibration_params`` ("custom") from the rest ("default").
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from typing import Any

import scipy.optimize as _opt

_real = _opt.minimize
_counts: Counter[str] = Counter()
_failures: list[str] = []


def _caller() -> tuple[str, str]:
    """Return (who, setting).

    ``setting`` is "custom" when the calling calibrator carries user
    ``calibration_params`` (its ``_params``), else "default". Generated code
    cannot be told apart and reports "unknown".
    """
    frame = sys._getframe(2)
    path = frame.f_code.co_filename
    owner = frame.f_locals.get("self")
    if path.endswith(os.path.join("calibration", "platt.py")):
        # PlattCalibrator keeps the user params in ``_params``.
        return "platt", "custom" if getattr(owner, "_params", None) else "default"
    if path.endswith(os.path.join("calibration", "beta.py")):
        # BetaCalibrator keeps them in ``_settings`` (``_params`` is the fit).
        return "beta", "custom" if getattr(owner, "_settings", None) else "default"
    if "lizyml" not in path and "site-packages" not in path:
        return "generated-or-test", "unknown"
    return "other", "n/a"


def _counting(fun: Any, x0: Any, *args: Any, **kwargs: Any) -> Any:
    who, setting = _caller()
    result = _real(fun, x0, *args, **kwargs)
    ok = bool(getattr(result, "success", True))
    _counts[f"{who}:{setting}:{'ok' if ok else 'FAILED'}"] += 1
    if not ok and who != "other":
        test = os.environ.get("PYTEST_CURRENT_TEST", "?")
        _failures.append(f"{who}:{setting} {test} :: {getattr(result, 'message', '')}")
    return result


_opt.minimize = _counting


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: Any) -> None:  # noqa: ARG001
    terminalreporter.write_line("i297 minimize census:")
    for key, n in sorted(_counts.items()):
        terminalreporter.write_line(f"  {key}: {n}")
    for line in sorted(set(_failures)):
        terminalreporter.write_line(f"  failed: {line}")
