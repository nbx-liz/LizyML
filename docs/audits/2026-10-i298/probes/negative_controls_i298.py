"""#298 negative controls: each new test catches the false claim it is for.

Every control patches one mechanism into a plausible wrong implementation, runs
the test that guards it, and expects it to FAIL. Patches are undone after each
control (``MonkeyPatch.undo``), test modules are re-imported for every run, and
a final run without any patch must pass: a control is "caught" only against
that clean baseline. Exit status 1 when a control is not caught or the baseline
fails.

Usage (from the repository root):
    uv run python docs/audits/2026-10-i298/probes/negative_controls_i298.py
"""

from __future__ import annotations

import inspect
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import lizyml.calibration._optimizer as optimizer  # noqa: E402
import lizyml.calibration.beta as beta  # noqa: E402
import lizyml.calibration.platt as platt  # noqa: E402
import lizyml.calibration.registry as registry  # noqa: E402
import lizyml.core._model_factories as factories  # noqa: E402
import lizyml.core.model as model_module  # noqa: E402

TESTS = "tests/test_calibration/test_i298_pr3c_evidence.py"


def _c_final_without_params(mp: pytest.MonkeyPatch) -> None:
    """(a, b) C_final is built without the written params; the folds keep them."""
    real = registry.get_calibrator
    calls = {"n": 0}

    def get(name: str, params: dict[str, Any] | None = None) -> Any:
        calls["n"] += 1
        return real(name, params=None if calls["n"] % 4 == 0 else params)

    mp.setattr(registry, "get_calibrator", get)


def _resolve_variant(*, pop_tol: bool, tol_after: bool, defaults: bool) -> Callable[..., Any]:
    def resolve(params: Any, *, n_coef: int, default_x0: Any, default_options: Any) -> Any:
        method = params.get("method", optimizer.DEFAULT_METHOD)
        options = dict(default_options.get(method, {})) if defaults else {}
        kwargs: dict[str, Any] = {"method": method}
        if tol_after:
            options.update(params.get("options", {}))
        if "tol" in params:
            if pop_tol:
                for key in optimizer._TOL_KEYS[method]:
                    options.pop(key, None)
            kwargs["tol"] = params["tol"]
        if not tol_after:
            options.update(params.get("options", {}))
        kwargs["options"] = options
        kwargs["x0"] = [float(v) for v in params.get("x0", default_x0)]
        if "bounds" in params:
            kwargs["bounds"] = [tuple(pair) for pair in params["bounds"]]
        return kwargs

    return resolve


def _tol_keeps_default_options(mp: pytest.MonkeyPatch) -> None:
    """(c) A written tol leaves the platt default ftol / gtol in options."""
    mp.setattr(platt, "resolve_minimize_kwargs", _resolve_variant(pop_tol=False, tol_after=False, defaults=True))


def _tol_overrides_written_options(mp: pytest.MonkeyPatch) -> None:
    """(d) Written options are applied before tol, which then removes them."""
    mp.setattr(platt, "resolve_minimize_kwargs", _resolve_variant(pop_tol=True, tol_after=True, defaults=True))


def _no_default_options(mp: pytest.MonkeyPatch) -> None:
    """(e) The per-method default options are dropped."""
    variant = _resolve_variant(pop_tol=True, tol_after=False, defaults=False)
    mp.setattr(platt, "resolve_minimize_kwargs", variant)
    mp.setattr(beta, "resolve_minimize_kwargs", variant)


def _no_rescale_transform(mp: pytest.MonkeyPatch) -> None:
    """(f) The written x0 and slope bounds are not moved into rescaled coordinates."""
    source = textwrap.dedent(inspect.getsource(platt.PlattCalibrator.fit))
    start = source.index('    kwargs["x0"] = [kwargs["x0"][0] * scale')
    end = source.index("    gradient = METHODS")
    patched = source[:start] + source[end:]
    namespace: dict[str, Any] = {}
    exec(compile(patched, "<no-rescale>", "exec"), vars(platt), namespace)  # noqa: S102
    mp.setattr(platt.PlattCalibrator, "fit", namespace["fit"])


def _no_canonicalisation(mp: pytest.MonkeyPatch) -> None:
    """(g) An alias reaches the isotonic calibrator and config.json as written."""
    mp.setattr(factories, "canonicalise_calibration_params", lambda params: dict(params))


def _no_inherited_seed(mp: pytest.MonkeyPatch) -> None:
    """(h) The training seed is not inherited by the isotonic calibrator."""
    real = factories.prepare_calibration_params

    def prepare(*args: Any, **kwargs: Any) -> Any:
        out = real(*args, **kwargs)
        if isinstance(out, dict):
            out.pop("seed", None)
        return out

    mp.setattr(factories, "prepare_calibration_params", prepare)
    mp.setattr(model_module, "prepare_calibration_params", prepare)


def _legacy_state_swapped(mp: pytest.MonkeyPatch) -> None:
    """(i) The legacy LogisticRegression state is read with slope and intercept swapped."""
    real = platt.PlattCalibrator.__setstate__

    def setstate(self: Any, state: dict[str, Any]) -> None:
        real(self, state)
        if self._coef is not None and "_model" in state:
            a, b = self._coef
            self._coef = (b, a)

    mp.setattr(platt.PlattCalibrator, "__setstate__", setstate)


T = TESTS + "::"
CONTROLS: dict[str, tuple[Callable[[pytest.MonkeyPatch], None], list[str]]] = {
    "(a) C_final without params, minimize -> 3c": (
        _c_final_without_params,
        [T + "test_3c_every_fold_and_c_final_solve_with_the_written_options"],
    ),
    "(b) C_final without params, isotonic -> 3c": (
        _c_final_without_params,
        [T + "test_3c_every_fold_and_c_final_train_isotonic_with_the_prepared_params"],
    ),
    "(c) tol keeps the default options -> 4a": (
        _tol_keeps_default_options,
        [T + "test_4a_written_tol_replaces_the_tolerance_options_it_sets"],
    ),
    "(d) tol overrides written options -> 4b": (
        _tol_overrides_written_options,
        [T + "test_4b_written_options_win_over_tol_and_over_the_defaults"],
    ),
    "(e) no per-method defaults -> 4c": (
        _no_default_options,
        [T + "test_4c_platt_default_options_apply_to_l_bfgs_b_only"],
    ),
    "(f) x0 and bounds not rescaled -> 5b": (
        _no_rescale_transform,
        [T + "test_5b_rescaled_solve_equals_the_unscaled_one_with_x0_and_slope_bounds"],
    ),
    "(g) alias not canonicalised -> 8a": (
        _no_canonicalisation,
        [T + "test_8a_config_json_carries_the_prepared_isotonic_params"],
    ),
    "(h) training seed not inherited -> 8a": (
        _no_inherited_seed,
        [T + "test_8a_config_json_carries_the_prepared_isotonic_params"],
    ),
    "(i) legacy state swapped -> 9a": (
        _legacy_state_swapped,
        [T + "test_9a_a_v0_17_1_platt_model_predicts_as_it_did"],
    ),
    "(i) legacy state swapped -> 9b": (
        _legacy_state_swapped,
        [T + "test_9b_the_legacy_platt_calibrator_maps_scores_as_it_did"],
    ),
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
    for module in [m for m in sys.modules if m == "tests" or m.startswith("tests.")]:
        del sys.modules[module]


def main() -> int:
    failures = 0
    for name, (apply, tests) in CONTROLS.items():
        _forget_test_modules()
        code = pytest.main(["-q", "-p", "no:cacheprovider", "-x", *tests], plugins=[_Patch(apply)])
        caught = code == pytest.ExitCode.TESTS_FAILED
        # A patch that fails to apply ends pytest with an error, not a test
        # failure: report the exit code, so it is not read as "passed".
        verdict = "caught (tests failed)" if caught else f"NOT CAUGHT (pytest exit {int(code)})"
        print(f"CONTROL {name}: {verdict}")
        failures += not caught
    _forget_test_modules()
    baseline = pytest.main(["-q", "-p", "no:cacheprovider", TESTS])
    print(f"BASELINE without patches: exit {int(baseline)}")
    print(f"\n{len(CONTROLS) - failures}/{len(CONTROLS)} controls caught")
    return 1 if failures or baseline != pytest.ExitCode.OK else 0


if __name__ == "__main__":
    raise SystemExit(main())
