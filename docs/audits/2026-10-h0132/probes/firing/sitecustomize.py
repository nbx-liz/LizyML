"""H-0132 firing-rate probe: log what MASE's two refusals would see.

Loaded through ``PYTHONPATH`` (this directory first) only when
``H0132_FIRING_LOG`` is set, so it also runs inside the notebook kernels the
notebook tests start (they inherit the environment). It wraps two functions
when their modules are imported, and the real call always runs unchanged:

- ``lizyml.evaluation.evaluator.Evaluator.evaluate`` -- one ``evaluate`` record
  per call: how many covered rows sit in two or more validation folds (where
  the overlap refusal fires), for a regression fit how many folds have a zero
  or non-finite outer-train scale, and whether MASE was requested;
- ``lizyml.metrics.regression.MASE.bind_train`` -- one ``bind`` record per call,
  from every caller (the Evaluator, the LightGBM feval through
  ``LGBMAdapter.fit``, direct calls): the size of the ``y_train`` it got and
  whether it refused (the zero-scale refusal itself).

Each record names its process role (``kernel`` when the interpreter runs
``ipykernel``, else ``pytest``) and the current test. A failure to record is
written as an ``error`` record, never skipped.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import os
import sys

_LOG = os.environ.get("H0132_FIRING_LOG")


def _role() -> str:
    # Read when a record is written, not at import: under ``python -m
    # ipykernel_launcher`` sitecustomize runs while ``sys.argv[0]`` is still
    # ``-m``, so an import-time reading labelled every kernel ``pytest``.
    return "kernel" if any("ipykernel" in arg for arg in sys.argv) else "pytest"


def _write(record: dict) -> None:
    record.update(
        pid=os.getpid(),
        role=_role(),
        test=os.environ.get("PYTEST_CURRENT_TEST", ""),
        argv0=sys.argv[0] if sys.argv else "",
    )
    with open(_LOG, "a", encoding="utf-8") as fh:  # type: ignore[arg-type]
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _patch_evaluator(module) -> None:  # type: ignore[no-untyped-def]
    import numpy as np

    original = module.Evaluator.evaluate

    def evaluate(self, fit_result, y, metric_entries):  # type: ignore[no-untyped-def]
        record: dict = {"kind": "evaluate", "task": getattr(self, "task", None)}
        try:
            n = len(np.asarray(y))
            counts = np.zeros(n, dtype=np.intp)
            for _, valid_idx in fit_result.splits.outer:
                np.add.at(counts, valid_idx, 1)
            names = [e if isinstance(e, str) else next(iter(e)) for e in metric_entries]
            zero_scale = None
            if record["task"] == "regression":
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


def _patch_mase(module) -> None:  # type: ignore[no-untyped-def]
    original = module.MASE.bind_train

    def bind_train(self, y_train):  # type: ignore[no-untyped-def]
        record: dict = {"kind": "bind"}
        try:
            record["n_train"] = int(len(y_train))
        except Exception as exc:  # noqa: BLE001 -- recorded, not swallowed
            record["error"] = f"{type(exc).__name__}: {exc}"
        try:
            bound = original(self, y_train)
        except Exception as exc:
            record["refused"] = f"{type(exc).__name__}"
            _write(record)
            raise
        record["refused"] = None
        _write(record)
        return bound

    module.MASE.bind_train = bind_train


_TARGETS = {
    "lizyml.evaluation.evaluator": _patch_evaluator,
    "lizyml.metrics.regression": _patch_mase,
}


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):  # type: ignore[no-untyped-def]
        patch = _TARGETS.get(fullname)
        if patch is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):  # type: ignore[no-untyped-def]
            exec_module(module)
            patch(module)

        spec.loader.exec_module = exec_and_patch  # type: ignore[method-assign]
        return spec


if _LOG:
    sys.meta_path.insert(0, _Finder())
