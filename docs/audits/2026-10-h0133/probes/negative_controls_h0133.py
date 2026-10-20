"""H-0133 negative controls (criterion 8): each test catches the defect it is for.

Every control patches one mechanism into a plausible wrong implementation, runs
the tests that guard it, and expects at least one of them to FAIL. Patches are
undone after each control (``MonkeyPatch.undo``), test modules are re-imported
for every run, and a final run without any patch must pass: a control is
"caught" only against that clean baseline. Exit status 1 when a control is not
caught or the baseline fails.

Usage (from the repository root):
    uv run python docs/audits/2026-10-h0133/probes/negative_controls_h0133.py
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import lizyml.codegen.config_writer as config_writer  # noqa: E402
import lizyml.codegen.values as codegen_values  # noqa: E402
import lizyml.core.category_values as category_values  # noqa: E402
import lizyml.persistence.loader as loader  # noqa: E402
from lizyml.features.encoders.categorical_encoder import CategoricalEncoder  # noqa: E402

VALUES = "tests/test_core/test_category_values.py"
STAND_IN = "tests/test_features/test_category_stand_in.py"
ACCEPT = "tests/test_persistence/test_h0133_acceptance.py"
RECORDS = "tests/test_persistence/test_h0120_records.py"


def _no_rename(mp: pytest.MonkeyPatch) -> None:
    """(a) transform hands LightGBM the raw categories of a stand-in column."""
    original = CategoricalEncoder.transform_with_warnings

    def transform(self: CategoricalEncoder, X: Any) -> Any:
        saved, self._stand_in_columns = self._stand_in_columns, []
        try:
            return original(self, X)
        finally:
            self._stand_in_columns = saved

    mp.setattr(CategoricalEncoder, "transform_with_warnings", transform)


def _decided_in_transform(mp: pytest.MonkeyPatch) -> None:
    """(b) The form is decided at transform, so a state saved after fit (CV
    snapshots it before any transform) records no stand-in column."""
    fit, transform = CategoricalEncoder.fit, CategoricalEncoder.transform_with_warnings

    def late_fit(self: CategoricalEncoder, *args: Any, **kwargs: Any) -> Any:
        out = fit(self, *args, **kwargs)
        self._late = self._stand_in_columns  # type: ignore[attr-defined]
        self._stand_in_columns = []
        return out

    def late_transform(self: CategoricalEncoder, X: Any) -> Any:
        late = getattr(self, "_late", None)
        if late is not None:
            self._stand_in_columns = late
        return transform(self, X)

    mp.setattr(CategoricalEncoder, "fit", late_fit)
    mp.setattr(CategoricalEncoder, "transform_with_warnings", late_transform)


def _loader_ignores_the_form(mp: pytest.MonkeyPatch) -> None:
    """(c) A loaded state is read as raw categories."""
    original = CategoricalEncoder.load_state

    def load_state(self: CategoricalEncoder, state: dict[str, Any]) -> Any:
        return original(self, {k: v for k, v in state.items() if k != "stand_in_columns"})

    mp.setattr(CategoricalEncoder, "load_state", load_state)


def _tuple_as_list(mp: pytest.MonkeyPatch) -> None:
    """(d) A decoded tuple comes back as a list (what plain JSON returns)."""
    original = category_values._decode_tagged

    def decode(obj: dict[str, Any], depth: int) -> Any:
        value = original(obj, depth)
        return list(value) if type(value) is tuple else value

    mp.setattr(category_values, "_decode_tagged", decode)


def _no_nan_check(mp: pytest.MonkeyPatch) -> None:
    """(e) The value check no longer refuses NaN-like values."""
    mp.setattr(category_values, "math", SimpleNamespace(isnan=lambda _x: False))


def _hash_order(mp: pytest.MonkeyPatch) -> None:
    """(f) A frozenset is written in its iteration (hash) order."""
    original = category_values._encode

    def encode(value: Any) -> Any:
        if type(value) is frozenset:
            return {"$lizyml": "frozenset", "items": [category_values._encode(v) for v in value]}
        return original(value)

    mp.setattr(category_values, "_encode", encode)


def _labels_take_the_category_codec(mp: pytest.MonkeyPatch) -> None:
    """(g) Target labels are written with the feature-category codec."""
    mp.setattr(config_writer, "plain_values", codegen_values.category_values)


def _load_skips_decoding(mp: pytest.MonkeyPatch) -> None:
    """(h) ``lizyml.persistence.load`` returns the tagged record undecoded."""
    mp.setattr(loader, "_decoded_declared_categories", lambda record, path: record)


CONTROLS: dict[str, tuple[Callable[[pytest.MonkeyPatch], None], list[str]]] = {
    "(a) no rename in transform": (_no_rename, [ACCEPT]),
    # Criterion 8 (b) names the fold-SHAP guard, so it runs alone here.
    "(b) form decided in transform": (
        _decided_in_transform,
        [f"{ACCEPT}::test_fold_shap_of_a_stand_in_column_equals_that_of_the_same_codes_as_str"],
    ),
    "(c) loader ignores stand_in_columns": (_loader_ignores_the_form, [ACCEPT]),
    "(d) tuple decoded as a list": (_tuple_as_list, [VALUES]),
    "(e) nested NaN admitted": (_no_nan_check, [VALUES, ACCEPT]),
    "(f) frozenset in hash order": (_hash_order, [VALUES]),
    "(g) labels take the category codec": (_labels_take_the_category_codec, [ACCEPT]),
    "(h) persistence.load skips decoding": (_load_skips_decoding, [RECORDS]),
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
    """A test module that imports a patched name binds the patch; drop the
    cached test modules so every run imports them afresh."""
    for module in [m for m in sys.modules if m == "tests" or m.startswith("tests.")]:
        del sys.modules[module]


def main() -> int:
    failures = 0
    for name, (apply, tests) in CONTROLS.items():
        _forget_test_modules()
        code = pytest.main(
            ["-q", "-p", "no:cacheprovider", "-x", *tests], plugins=[_Patch(apply)]
        )
        caught = code == pytest.ExitCode.TESTS_FAILED
        print(f"CONTROL {name}: {'caught (tests failed)' if caught else 'NOT CAUGHT'}")
        failures += not caught
    _forget_test_modules()
    baseline = pytest.main(["-q", "-p", "no:cacheprovider", VALUES, STAND_IN, ACCEPT, RECORDS])
    print(f"BASELINE without patches: exit {int(baseline)}")
    print(f"\n{len(CONTROLS) - failures}/{len(CONTROLS)} controls caught")
    return 1 if failures or baseline != pytest.ExitCode.OK else 0


if __name__ == "__main__":
    raise SystemExit(main())
