"""H-0132 firing-rate probe: log every Evaluator.evaluate call's fold overlap.

Loaded through ``PYTHONPATH`` (this directory first) only when
``H0132_FIRING_LOG`` is set, so it also runs inside notebook kernels started by
the notebook tests. When ``lizyml.evaluation.evaluator`` is imported, its
``Evaluator.evaluate`` is wrapped: before the real call, the wrapper records
how many covered rows sit in two or more validation folds, for a regression
fit how many folds have a zero or non-finite outer-train scale (where MASE's
``bind_train`` would refuse), whether a metric that needs training targets
(MASE) was requested, and which test or process made the fit. The real call then runs unchanged. Nothing is skipped silently:
a failure to record is written to the log as an error line.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import os
import sys

_LOG = os.environ.get("H0132_FIRING_LOG")
_TARGET = "lizyml.evaluation.evaluator"


def _write(record: dict) -> None:
    with open(_LOG, "a", encoding="utf-8") as fh:  # type: ignore[arg-type]
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _patch(module) -> None:  # type: ignore[no-untyped-def]
    import numpy as np

    original = module.Evaluator.evaluate

    def evaluate(self, fit_result, y, metric_entries):  # type: ignore[no-untyped-def]
        record = {
            "pid": os.getpid(),
            "test": os.environ.get("PYTEST_CURRENT_TEST", ""),
            "argv0": sys.argv[0] if sys.argv else "",
            "task": getattr(self, "task", None),
        }
        try:
            n = len(np.asarray(y))
            counts = np.zeros(n, dtype=np.intp)
            for _, valid_idx in fit_result.splits.outer:
                np.add.at(counts, valid_idx, 1)
            names = [e if isinstance(e, str) else next(iter(e)) for e in metric_entries]
            zero_scale = None
            if getattr(self, "task", None) == "regression":
                y_arr = np.asarray(y, dtype=float)
                zero_scale = 0
                for train_idx, _ in fit_result.splits.outer:
                    t = y_arr[train_idx]
                    s = float(np.mean(np.abs(t - t.mean()))) if len(t) else float("nan")
                    zero_scale += int(not np.isfinite(s) or s == 0.0)
            record.update(
                n_folds=len(fit_result.splits.outer),
                n_rows=n,
                overlapping_rows=int((counts > 1).sum()),
                needs_train_target="mase" in names,
                zero_scale_folds=zero_scale,
            )
        except Exception as exc:  # noqa: BLE001 -- recorded, not swallowed
            record["error"] = f"{type(exc).__name__}: {exc}"
        _write(record)
        return original(self, fit_result, y, metric_entries)

    module.Evaluator.evaluate = evaluate


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):  # type: ignore[no-untyped-def]
        if fullname != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader
        exec_module = loader.exec_module

        def exec_and_patch(module):  # type: ignore[no-untyped-def]
            exec_module(module)
            _patch(module)

        loader.exec_module = exec_and_patch  # type: ignore[method-assign]
        return spec


if _LOG:
    sys.meta_path.insert(0, _Finder())
