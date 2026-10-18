"""H-0132 negative controls (criterion 7): each MASE test catches the defect it is for.

Every control patches one mechanism into a plausible wrong implementation, runs
the tests that guard it, and expects at least one of them to FAIL. Patches are
undone after each control, so one control cannot make another look caught. A
control whose tests all pass means the tests do not guard that mechanism; the
script then exits 1.

Usage (from the repository root):
    uv run python docs/audits/2026-10-h0132/probes/negative_controls_h0132.py
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import lizyml.codegen.templates as templates  # noqa: E402
import lizyml.estimators.lgbm.metric_bridge as metric_bridge  # noqa: E402
import lizyml.estimators.lgbm.param_validation as param_validation  # noqa: E402
import lizyml.evaluation.evaluator as evaluator  # noqa: E402
from lizyml.metrics.regression import MASE  # noqa: E402

EVAL = "tests/test_evaluation/test_mase_evaluator.py"
FEVAL = "tests/test_estimators/test_mase_feval.py"
MODEL = "tests/test_core/test_mase_model.py"
CODEGEN = "tests/test_codegen/test_mase_codegen.py"


def _scale_from_scored_rows(mp: pytest.MonkeyPatch) -> None:
    """(a) Bind the metric to the rows it scores instead of the training rows."""
    original = evaluator._compute_metrics

    def compute(metrics, y_true, y_pred, task, y_train=None):  # type: ignore[no-untyped-def]
        return original(metrics, y_true, y_pred, task, y_train=y_true)

    mp.setattr(evaluator, "_compute_metrics", compute)


def _unweighted_pool(mp: pytest.MonkeyPatch) -> None:
    """(b) Pool the per-fold values with an unweighted mean."""
    mp.setattr(
        evaluator, "_pool_by_fold_size", lambda values, sizes: float(np.mean(values))
    )


def _feval_scale_from_labels(mp: pytest.MonkeyPatch) -> None:
    """(c) The feval takes its scale from the eval set's labels: training targets
    are dropped, and an unbound MASE silently scales by the scored rows."""
    original = metric_bridge.resolve_metrics

    def resolve(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("y_train", None)
        return original(*args, **kwargs)

    def call(self: MASE, y_true: Any, y_pred: Any) -> float:
        scale = self._scale
        if scale is None:
            scale = float(np.mean(np.abs(y_true - np.mean(y_true))))
        return float(np.mean(np.abs(y_true - y_pred)) / scale)

    mp.setattr(param_validation, "resolve_metrics", resolve)
    mp.setattr(metric_bridge, "resolve_metrics", resolve)
    mp.setattr(MASE, "__call__", call)


def _overlap_passes(mp: pytest.MonkeyPatch) -> None:
    """(d) Overlapping validation folds pass silently."""
    mp.setattr(evaluator, "_refuse_overlapping_folds", lambda *args, **kwargs: None)


def _codegen_scale_from_all_rows(mp: pytest.MonkeyPatch) -> None:
    """(e) The generated train.py binds the feval to every row, inner-valid included."""
    line = "build_feval_from_config(y.iloc[tr] if iv is not None else y)"
    assert line in templates._TRAIN_PY
    mp.setattr(
        templates, "_TRAIN_PY", templates._TRAIN_PY.replace(line, "build_feval_from_config(y)")
    )


CONTROLS: dict[str, tuple[Callable[[pytest.MonkeyPatch], None], list[str]]] = {
    "(a) scale from the scored rows": (_scale_from_scored_rows, [EVAL]),
    "(b) unweighted OOF pool": (_unweighted_pool, [EVAL]),
    "(c) feval scale from eval labels": (_feval_scale_from_labels, [FEVAL]),
    "(d) overlapping folds pass": (_overlap_passes, [EVAL, MODEL]),
    "(e) generated scale from all rows": (_codegen_scale_from_all_rows, [CODEGEN]),
}


class _Patch:
    def __init__(self, apply: Callable[[pytest.MonkeyPatch], None]) -> None:
        self.apply = apply
        self.mp = pytest.MonkeyPatch()

    def pytest_configure(self, config: Any) -> None:
        self.apply(self.mp)

    def pytest_unconfigure(self, config: Any) -> None:
        self.mp.undo()


def _forget_test_modules() -> None:
    """A test module that imports a patched name binds the patch; pytest then
    reuses the cached module in the next run. Drop them so every run imports
    its tests afresh (the unpatched baseline below guards this)."""
    for module in [m for m in sys.modules if m == "tests" or m.startswith("tests.")]:
        del sys.modules[module]


def main() -> int:
    failures = 0
    for name, (apply, tests) in CONTROLS.items():
        _forget_test_modules()
        code = pytest.main(
            ["-q", "-p", "no:cacheprovider", *tests],
            plugins=[_Patch(apply)],
        )
        caught = code == pytest.ExitCode.TESTS_FAILED
        print(f"CONTROL {name}: {'caught (tests failed)' if caught else 'NOT CAUGHT'}")
        failures += not caught
    _forget_test_modules()
    baseline = pytest.main(["-q", "-p", "no:cacheprovider", EVAL, FEVAL, MODEL, CODEGEN])
    print(f"BASELINE without patches: exit {int(baseline)}")
    print(f"\n{len(CONTROLS) - failures}/{len(CONTROLS)} controls caught")
    return 1 if failures or baseline != pytest.ExitCode.OK else 0


if __name__ == "__main__":
    raise SystemExit(main())
