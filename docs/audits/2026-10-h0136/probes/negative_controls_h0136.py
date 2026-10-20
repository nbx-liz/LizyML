"""H-0136 negative controls (acceptance criterion 9): each check is guarded.

Every control patches one part of the H-0136 check back to a plausible wrong
implementation, runs the tests that guard it (selected by ``-k``) and expects
them to FAIL. Patches are undone after each control and the test modules are
re-imported for every run. A final run without any patch must pass. The exit
status is 1 when a control is not caught or the baseline fails.

Run it on pandas 2 with pytz: the joblib controls are guarded by the pytz
cases, which pandas 3 skips.

Usage (from the repository root):
    python docs/audits/2026-10-h0136/probes/negative_controls_h0136.py
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import lizyml.core.model as model_mod  # noqa: E402
import lizyml.features.encoders.state_changes as changes  # noqa: E402
import lizyml.persistence._category_preflight as preflight  # noqa: E402
import lizyml.persistence.exporter as exporter  # noqa: E402
from lizyml.core._model_persistence import ModelPersistenceMixin  # noqa: E402

TESTS = [
    "tests/test_persistence/test_h0136_acceptance.py",
    "tests/test_persistence/test_h0136_state_changes.py",
]


def _p1_off(mp: pytest.MonkeyPatch) -> None:
    """No behaviour check at all (the H-0133 exporter)."""
    mp.setattr(exporter, "state_changes", lambda state: [])
    mp.setattr(exporter, "declared_change", lambda column, cats: None)


def _joblib_off(mp: pytest.MonkeyPatch) -> None:
    mp.setattr(preflight, "SERIALIZERS", ("json",))


def _json_off(mp: pytest.MonkeyPatch) -> None:
    mp.setattr(preflight, "SERIALIZERS", ("joblib",))


def _kinds_collapsed(mp: pytest.MonkeyPatch) -> None:
    """Every change reported as ``raises``."""
    real = changes.column_change

    def column_change(state: Any, projected: Any, column: Any) -> Any:
        found = real(state, projected, column)
        if found is None:
            return None
        return changes.CategoryStateChange(found.column, "raises", exception="Exception")

    mp.setattr(changes, "column_change", column_change)


def _mode_by_original_value(mp: pytest.MonkeyPatch) -> None:
    """The first implementation: the original mode looked up under both states."""
    real = changes.column_change

    def column_change(state: Any, projected: Any, column: Any) -> Any:
        mode = (state.get("modes") or {}).get(column)
        same_mode = {**projected, "modes": {**(projected.get("modes") or {}), column: mode}}
        return real(state, same_mode, column)

    mp.setattr(changes, "column_change", column_change)


def _declared_off(mp: pytest.MonkeyPatch) -> None:
    mp.setattr(exporter, "declared_change", lambda column, cats: None)
    mp.setattr(preflight, "declared_change", lambda column, cats: None)


def _export_code_unchecked(mp: pytest.MonkeyPatch) -> None:
    """export_code without the shared check (before H-0136)."""
    real = exporter.check_category_values

    def check(*args: Any, surface: str = "Model.export", **kwargs: Any) -> None:
        if surface != "export_code":
            real(*args, surface=surface, **kwargs)

    mp.setattr(exporter, "check_category_values", check)


def _export_code_checks_after_writing(mp: pytest.MonkeyPatch) -> None:
    real_check = exporter.check_category_values
    real_export_code = ModelPersistenceMixin.export_code

    def export_code(self: Any, path: Any) -> Any:
        state = self._get_fit_state()
        mp.setattr(exporter, "check_category_values", lambda *a, **k: None)
        try:
            out = real_export_code(self, path)
        finally:
            mp.setattr(exporter, "check_category_values", real_check)
        real_check(state.fit_result, self._require_refit(), state.declared_categories)
        return out

    mp.setattr(ModelPersistenceMixin, "export_code", export_code)


def _fit_warning_off(mp: pytest.MonkeyPatch) -> None:
    mp.setattr(model_mod, "warn_saved_state_changes", lambda *args: None)


CONTROLS: dict[str, tuple[Callable[[pytest.MonkeyPatch], None], str]] = {
    "P1 off": (_p1_off, "refused"),
    "joblib leg off": (_joblib_off, "pytz or 372"),
    "JSON leg off": (_json_off, "uint64"),
    "kinds collapsed to raises": (_kinds_collapsed, "unseen or reordered or mode"),
    "mode looked up by the original value": (_mode_by_original_value, "mode"),
    "declared categories not checked": (_declared_off, "declared"),
    "export_code not checked": (_export_code_unchecked, "refused"),
    "export_code checks after writing": (_export_code_checks_after_writing, "refused"),
    "fit warning off": (_fit_warning_off, "refused"),
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
    import pandas as pd

    print(f"pandas={pd.__version__}")
    failures = 0
    for name, (apply, selection) in CONTROLS.items():
        _forget_test_modules()
        args = ["-q", "-p", "no:cacheprovider", "-x", "-k", selection, *TESTS]
        code = pytest.main(args, plugins=[_Patch(apply)])
        caught = code == pytest.ExitCode.TESTS_FAILED
        verdict = "caught (tests failed)" if caught else f"NOT CAUGHT (pytest exit {int(code)})"
        print(f"CONTROL {name}: {verdict}")
        failures += not caught
    _forget_test_modules()
    baseline = pytest.main(["-q", "-p", "no:cacheprovider", *TESTS])
    print(f"BASELINE without patches: exit {int(baseline)}")
    print(f"\n{len(CONTROLS) - failures}/{len(CONTROLS)} controls caught")
    return 1 if failures or baseline != pytest.ExitCode.OK else 0


if __name__ == "__main__":
    raise SystemExit(main())
