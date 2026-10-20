"""H-0134 negative controls (criterion 10): each test catches the defect it is for.

Every control patches one mechanism into a plausible wrong implementation, runs
the tests that guard it, and expects at least one of them to FAIL. Patches are
undone after each control (``MonkeyPatch.undo``), test modules are re-imported
for every run, and a final run without any patch must pass: a control is
"caught" only against that clean baseline. Exit status 1 when a control is not
caught or the baseline fails.

Usage (from the repository root):
    uv run python docs/audits/2026-10-h0134/probes/negative_controls_h0134.py
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import lizyml.core.tuning_codec as codec  # noqa: E402
import lizyml.persistence.exporter as exporter  # noqa: E402
import lizyml.persistence.loader as loader  # noqa: E402
from lizyml import Model  # noqa: E402
from lizyml.core.types.tuning_result import TrialResult  # noqa: E402

CENSUS = "tests/test_persistence/test_reporting_surfaces_survive_load.py"
ACCEPT = "tests/test_persistence/test_h0134_acceptance.py"
CODEC = "tests/test_core/test_tuning_history_codec.py"


def _no_history_written(mp: pytest.MonkeyPatch) -> None:
    """(a) Export declares no history (the H-0086 behaviour)."""
    original = exporter.write_checked

    def write_checked(*args: Any, **kwargs: Any) -> None:
        kwargs["tuning_history_available"] = False
        original(*args, **kwargs)

    mp.setattr(exporter, "write_checked", write_checked)


def _loose_decode(mp: pytest.MonkeyPatch) -> None:
    """(b) The reader ignores unknown keys and the schema, and skips the
    canonical check."""
    mp.setattr(codec, "_exact_keys", lambda obj, keys, what: None)

    def parse(raw: bytes) -> codec.TuningHistory:
        obj = json.loads(raw)
        report = obj.get("boundary_report")
        return codec.TuningHistory(
            trials=[codec._decode_trial(t) for t in obj["trials"]],
            rounds=tuple(codec._decode_round(r) for r in obj["rounds"]),
            boundary_report=None if report is None else codec.decode_report(report),
        )

    mp.setattr(loader, "parse_history", parse)


def _missing_file_is_empty(mp: pytest.MonkeyPatch) -> None:
    """(c) A declared history that cannot be read loads as an empty one."""
    original = loader._declared_tuning_history

    def declared(src: Path, metadata: dict[str, Any], path: Any) -> Any:
        if not (src / codec.HISTORY_FILE).exists():
            return codec.TuningHistory(trials=[], rounds=(), boundary_report=None)
        return original(src, metadata, path)

    mp.setattr(loader, "_declared_tuning_history", declared)


def _empty_table_without_history(mp: pytest.MonkeyPatch) -> None:
    """(d) The surfaces answer for a history-less load (empty table, no traces)."""
    original = Model._get_tuning_state

    def get(self: Model, method: str) -> Any:
        saved, self._tuning_history_available = self._tuning_history_available, True
        try:
            return original(self, method)
        finally:
            self._tuning_history_available = saved

    mp.setattr(Model, "_get_tuning_state", get)


def _load_marks_available(mp: pytest.MonkeyPatch) -> None:
    """(e) A history-less load is marked as holding its history."""
    original = Model.load.__func__  # type: ignore[attr-defined]

    def load(cls: type[Model], path: Any) -> Model:
        model = original(cls, path)
        model._tuning_history_available = True
        return model

    mp.setattr(Model, "load", classmethod(load))


def _params_sorted(mp: pytest.MonkeyPatch) -> None:
    """(f) The writer sorts each trial's parameters by name."""
    original = codec._encode_trial

    def encode(trial: TrialResult) -> dict[str, Any]:
        return original(
            TrialResult(
                trial.number,
                dict(sorted(trial.params.items())),
                trial.score,
                trial.state,
                trial.round,
            )
        )

    mp.setattr(codec, "_encode_trial", encode)


def _nan_as_none(mp: pytest.MonkeyPatch) -> None:
    """(g) A NaN score is written as None (and read back as None)."""
    encode, decode = codec._encode_score, codec._decode_score

    def encode_score(value: Any, what: str) -> list[Any]:
        if type(value) is float and math.isnan(value):
            return codec.encode_scalar(None)
        return encode(value, what)

    def decode_score(payload: Any, what: str) -> Any:
        if payload == ["none", None]:
            return None
        return decode(payload, what)

    mp.setattr(codec, "_encode_score", encode_score)
    mp.setattr(codec, "_decode_score", decode_score)


def _bad_declaration_is_no_history(mp: pytest.MonkeyPatch) -> None:
    """(h) A declaration of the wrong shape is read as "no history"."""
    original = loader._declared_tuning_history

    def declared(src: Path, metadata: dict[str, Any], path: Any) -> Any:
        decl = metadata["tuning"].get("history")
        if decl is not None and decl != loader._HISTORY_DECLARATION:
            return None
        return original(src, metadata, path)

    mp.setattr(loader, "_declared_tuning_history", declared)


def _unfinished_trials_dropped(mp: pytest.MonkeyPatch) -> None:
    """(i) Export drops running and waiting trials."""
    original = exporter.history_bytes

    def history_bytes(history: codec.TuningHistory) -> bytes:
        kept = [t for t in history.trials if t.state not in ("running", "waiting")]
        return original(history._replace(trials=kept))

    mp.setattr(exporter, "history_bytes", history_bytes)


EVERY_SHAPE = f"{ACCEPT}::test_every_shape_answers_the_same_after_load"
BROKEN = f"{ACCEPT}::test_a_broken_history_or_declaration_is_refused"
NO_HISTORY = f"{ACCEPT}::test_an_artifact_without_history_says_so_on_every_surface"
REEXPORT = f"{ACCEPT}::test_reexporting_a_history_less_model_keeps_it_unknown"

CONTROLS: dict[str, tuple[Callable[[pytest.MonkeyPatch], None], list[str]]] = {
    "(a) history not written -> 1": (_no_history_written, [CENSUS]),
    "(b) loose decode -> 4": (_loose_decode, [BROKEN]),
    "(c) missing file is empty -> 4": (_missing_file_is_empty, [BROKEN]),
    "(d) empty table without history -> 2": (_empty_table_without_history, [NO_HISTORY]),
    "(e) history-less load marked available -> 2 re-export": (
        _load_marks_available,
        [REEXPORT],
    ),
    "(f) params sorted -> 3": (_params_sorted, [EVERY_SHAPE]),
    "(g) NaN written as None -> 3": (_nan_as_none, [EVERY_SHAPE]),
    "(h) bad declaration is no history -> 4": (_bad_declaration_is_no_history, [BROKEN]),
    "(i) unfinished trials dropped -> 3": (_unfinished_trials_dropped, [EVERY_SHAPE]),
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
    baseline = pytest.main(["-q", "-p", "no:cacheprovider", CENSUS, ACCEPT, CODEC])
    print(f"BASELINE without patches: exit {int(baseline)}")
    print(f"\n{len(CONTROLS) - failures}/{len(CONTROLS)} controls caught")
    return 1 if failures or baseline != pytest.ExitCode.OK else 0


if __name__ == "__main__":
    raise SystemExit(main())
