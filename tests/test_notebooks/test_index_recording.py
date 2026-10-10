"""The call recorder and the retry policy of the index execution (H-0119 6.(b)).

Acceptance 7 (the recorder, on ``Model`` directly, no notebook) and 8 (the
retry policy, with the executor replaced, no kernel), plus the runtime replays
of acceptance 3 and 6: an instrumented notebook is executed cell by cell with
``exec`` (fast) or in a real kernel (``slow``), and each counterexample must
fail the recording.
"""

from __future__ import annotations

import copy
import inspect
import pathlib
from typing import Any

import pytest
from nbclient.exceptions import CellExecutionError

from lizyml import Model
from lizyml.core.exceptions import LizyMLError
from tests._helpers import make_config, make_regression_df
from tests.test_notebooks import index_runner
from tests.test_notebooks.index_recorder import CallRecorder
from tests.test_notebooks.network_markers import NETWORK_ERROR_MARKERS


def _public(cls: type) -> dict[str, Any]:
    return {
        name: inspect.getattr_static(cls, name)
        for name, value in inspect.getmembers(cls)
        if not name.startswith("_") and callable(value)
    }


@pytest.fixture(scope="module")
def fitted() -> Model:
    model = Model(make_config("regression", n_estimators=5))
    model.fit(data=make_regression_df(n=120))
    return model


@pytest.fixture
def recorder() -> Any:
    rec = CallRecorder(Model)
    rec.install()
    try:
        yield rec
    finally:
        rec.uninstall()


# --- 7. The recorder ------------------------------------------------------------


def test_load_stays_a_classmethod_bound_alike_from_class_and_instance(
    recorder: CallRecorder, fitted: Model, tmp_path: pathlib.Path
) -> None:
    assert isinstance(inspect.getattr_static(Model, "load"), classmethod)
    assert Model.load.__self__ is Model
    assert fitted.load.__self__ is Model
    fitted.export(tmp_path / "artifact")
    recorder.calls.clear()
    assert isinstance(Model.load(tmp_path / "artifact"), Model)
    assert isinstance(fitted.load(tmp_path / "artifact"), Model)
    assert recorder.calls == [("load", id(Model)), ("load", id(Model))]


def test_properties_are_left_as_they_are(recorder: CallRecorder) -> None:
    del recorder
    assert isinstance(inspect.getattr_static(Model, "fit_result"), property)
    assert "fit_result" not in _public(Model)


def test_static_and_class_methods_and_properties_keep_their_binding() -> None:
    class Sample:
        @staticmethod
        def make(x: int) -> int:
            """Static."""
            return x + 1

        @classmethod
        def build(cls, x: int) -> tuple[type, int]:
            return cls, x

        @property
        def value(self) -> int:
            return 7

        def run(self) -> int:
            return self.make(1) + self.build(2)[1]

    original = dict(vars(Sample))
    rec = CallRecorder(Sample)
    rec.install()
    try:
        assert isinstance(inspect.getattr_static(Sample, "make"), staticmethod)
        assert isinstance(inspect.getattr_static(Sample, "build"), classmethod)
        assert inspect.getattr_static(Sample, "value") is original["value"]
        sample = Sample()
        assert Sample.make(1) == sample.make(1) == 2
        assert Sample.build(3) == sample.build(3) == (Sample, 3)
        assert sample.value == 7
        assert inspect.getdoc(Sample.make) == "Static."
        rec.calls.clear()
        assert sample.run() == 4
        assert rec.calls == [("run", id(sample))], "inner calls are not recorded"
        Sample.make(0)
        assert rec.calls[-1] == ("make", None)
    finally:
        rec.uninstall()
    assert dict(vars(Sample)) == original


def test_names_signatures_and_docstrings_are_unchanged() -> None:
    before = {
        name: (
            inspect.signature(getattr(Model, name)),
            inspect.getdoc(getattr(Model, name)),
        )
        for name in _public(Model)
    }
    rec = CallRecorder(Model)
    rec.install()
    try:
        after = {
            name: (
                inspect.signature(getattr(Model, name)),
                inspect.getdoc(getattr(Model, name)),
            )
            for name in _public(Model)
        }
        assert {n: getattr(Model, n).__name__ for n in after} == {n: n for n in after}
    finally:
        rec.uninstall()
    assert after == before
    assert len(before) >= 20


def test_a_public_method_called_from_another_is_not_recorded(
    recorder: CallRecorder, fitted: Model
) -> None:
    recorder.calls.clear()
    fitted.importance_plot(kind="shap")
    assert recorder.calls == [("importance_plot", id(fitted))]


def test_depth_survives_a_call_that_raises(recorder: CallRecorder) -> None:
    unfitted = Model(make_config("regression"))
    recorder.calls.clear()
    with pytest.raises(LizyMLError):
        unfitted.importance_plot(kind="shap")  # raises inside importance()
    with pytest.raises(LizyMLError):
        unfitted.importance()
    assert recorder.calls == [
        ("importance_plot", id(unfitted)),
        ("importance", id(unfitted)),
    ]


def test_uninstall_restores_the_original_methods() -> None:
    before = _public(Model)
    own = dict(vars(Model))
    rec = CallRecorder(Model)
    rec.install()
    assert _public(Model) != before
    with pytest.raises(RuntimeError, match="installed"):
        rec.install()
    rec.uninstall()
    assert _public(Model) == before
    assert dict(vars(Model)) == own
    with pytest.raises(RuntimeError, match="not installed"):
        rec.uninstall()


def test_expect_confirm_finish(recorder: CallRecorder, fitted: Model) -> None:
    recorder.expect(fitted, "params_table", "1:0")
    fitted.params_table()
    recorder.confirm("1:0")
    assert recorder.finish(["1:0"]) == f"{index_runner.SENTINEL} 1"


@pytest.mark.parametrize(
    "case", ["not a Model", "not called", "another method", "nested expect", "missing"]
)
def test_the_recording_fails(recorder: CallRecorder, fitted: Model, case: str) -> None:
    with pytest.raises(AssertionError):
        if case == "not a Model":
            recorder.expect(object(), "params_table", "1:0")
        elif case == "not called":
            recorder.expect(fitted, "params_table", "1:0")
            recorder.confirm("1:0")
        elif case == "another method":
            recorder.expect(fitted, "params_table", "1:0")
            fitted.evaluate_table()
            recorder.confirm("1:0")
        elif case == "nested expect":
            recorder.expect(fitted, "params_table", "1:0")
            recorder.expect(fitted, "params_table", "1:1")
        else:
            recorder.finish(["1:0"])


# --- 8. The retry policy ------------------------------------------------------------


def _marker_error() -> CellExecutionError:
    return CellExecutionError(
        f"... {NETWORK_ERROR_MARKERS[0]}: 503 ...", "HTTPError", "503"
    )


def _run(outcomes: list[BaseException | None], tmp_path: pathlib.Path) -> list[Any]:
    log: list[Any] = []
    pending = list(outcomes)

    def execute(kernel: object, workdir: pathlib.Path) -> None:
        assert workdir.is_dir()
        outcome = pending.pop(0)
        if outcome is not None:
            raise outcome

    counter = iter(range(100))
    index_runner.run_with_retries(
        execute,
        new_kernel=object,
        new_workdir=lambda: index_runner.fresh_workdir(tmp_path / f"w{next(counter)}"),
        log=log,
    )
    return log


def test_success_on_the_first_attempt(tmp_path: pathlib.Path) -> None:
    assert len(_run([None], tmp_path)) == 1


def test_a_marker_failure_then_success(tmp_path: pathlib.Path) -> None:
    log = _run([_marker_error(), None], tmp_path)
    assert len(log) == 2
    assert log[0].error is not None and log[1].error is None


def test_three_marker_failures_fail_without_skipping(tmp_path: pathlib.Path) -> None:
    log: list[Any] = []

    def execute(kernel: object, workdir: pathlib.Path) -> None:
        raise _marker_error()

    with pytest.raises(CellExecutionError):
        index_runner.run_with_retries(
            execute,
            new_kernel=object,
            new_workdir=lambda: index_runner.fresh_workdir(tmp_path / str(len(log))),
            log=log,
        )
    assert len(log) == 3


@pytest.mark.parametrize(
    "error",
    [
        CellExecutionError("NameError: x", "NameError", "x"),
        RuntimeError("Kernel died before replying"),
        AssertionError(f"recording failed ({NETWORK_ERROR_MARKERS[0]} in text)"),
    ],
)
def test_any_other_failure_ends_after_one_attempt(
    error: BaseException, tmp_path: pathlib.Path
) -> None:
    log: list[Any] = []

    def execute(kernel: object, workdir: pathlib.Path) -> None:
        raise error

    with pytest.raises(type(error)):
        index_runner.run_with_retries(
            execute,
            new_kernel=object,
            new_workdir=lambda: index_runner.fresh_workdir(tmp_path / "w"),
            log=log,
        )
    assert len(log) == 1


def test_each_attempt_gets_its_own_kernel_and_workdir(tmp_path: pathlib.Path) -> None:
    log = _run([_marker_error(), _marker_error(), None], tmp_path)
    assert len(log) == 3
    assert len({id(a.kernel) for a in log}) == 3
    assert len({a.workdir for a in log}) == 3
    for attempt in log:
        assert (attempt.workdir / "tutorial_codegen_export.ipynb").is_file()


# --- Runtime replays (acceptance 3 and 6) ---------------------------------------------


def _cell(
    source: str, *, tagged: bool = False, tags: list[str] | None = None
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if tagged or tags:
        metadata["tags"] = (tags or []) + (["index-example"] if tagged else [])
    return {
        "cell_type": "code",
        "metadata": metadata,
        "source": source,
        "outputs": [],
        "execution_count": None,
    }


SETUP = (
    "from lizyml import Model\n"
    "from tests._helpers import make_config, make_regression_df\n"
    "model = Model(make_config('regression', n_estimators=5))\n"
    "model.fit(data=make_regression_df(n=120))"
)
DECLARED = {"models": ["model"], "methods": ["params_table"], "extras": []}


def _notebook(*cells: dict[str, Any]) -> dict[str, Any]:
    return {
        "cells": list(cells),
        "metadata": {"lizyml": {"index": copy.deepcopy(DECLARED)}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _exec(nb: dict[str, Any]) -> str:
    instrumented = index_runner.instrument_notebook(nb)
    namespace: dict[str, Any] = {}
    printed: list[str] = []
    namespace["print"] = lambda *a: printed.append(" ".join(map(str, a)))
    try:
        for cell in instrumented["cells"]:
            exec(compile(cell["source"], "<cell>", "exec"), namespace)
    finally:
        recorder = namespace.get(index_runner.RECORDER_NAME)
        if recorder is not None:
            recorder.uninstall()
    return "\n".join(printed)


def test_the_instrumented_notebook_records_its_examples() -> None:
    nb = _notebook(_cell(SETUP), _cell("model.params_table()", tagged=True))
    assert _exec(nb) == f"{index_runner.SENTINEL} 1"


LOOK_ALIKE = "class Fake:\n    def params_table(self):\n        return 1\n"
RUNTIME_REPLAYS = {
    "a look-alike receiver": LOOK_ALIKE + "model = Fake()",
    "a reassignment": "model = 1",
    "a `def` rebinding the model": "def model():\n    pass",
    "a `class` rebinding the model": (
        "class model:\n    params_table = staticmethod(lambda: 1)"
    ),
    "a subclass that never runs the method": (
        "class Sub(Model):\n    def params_table(self):\n        return 1\n"
        "model = Sub(make_config('regression'))"
    ),
}


@pytest.mark.parametrize("case", sorted(RUNTIME_REPLAYS))
def test_runtime_replays_fail(case: str) -> None:
    nb = _notebook(
        _cell(SETUP),
        _cell(RUNTIME_REPLAYS[case]),
        _cell("model.params_table()", tagged=True),
    )
    with pytest.raises(AssertionError):
        _exec(nb)


def test_instrumentation_does_not_change_the_file(tmp_path: pathlib.Path) -> None:
    nb = _notebook(_cell(SETUP), _cell("model.params_table()", tagged=True))
    snapshot = copy.deepcopy(nb)
    index_runner.instrument_notebook(nb)
    assert nb == snapshot


@pytest.mark.slow
@pytest.mark.parametrize(
    ("case", "cells", "reason"),
    [
        ("positive control", [_cell("model.params_table()", tagged=True)], None),
        (
            "a tagged cell the kernel skips",
            [_cell("model.params_table()", tagged=True, tags=["skip-execution"])],
            "never ran",
        ),
        (
            "a failed guard in a cell allowed to raise",
            [
                _cell(LOOK_ALIKE + "model = Fake()"),
                _cell("model.params_table()", tagged=True, tags=["raises-exception"]),
            ],
            "never ran",
        ),
        (
            "a receiver that is not a Model",
            [
                _cell(LOOK_ALIKE + "model = Fake()"),
                _cell("model.params_table()", tagged=True),
            ],
            "not a Model",
        ),
    ],
)
def test_kernel_probes(
    case: str, cells: list[dict[str, Any]], reason: str | None, tmp_path: pathlib.Path
) -> None:
    del case
    root = pathlib.Path(__file__).resolve().parents[2]
    setup = _cell(f"import sys\nsys.path.insert(0, {str(root)!r})\n" + SETUP)
    nb = _notebook(setup, *cells)
    if reason is None:
        index_runner.run_instrumented(nb, workdir_base=tmp_path, timeout=120)
    else:
        with pytest.raises(CellExecutionError, match=reason):
            index_runner.run_instrumented(nb, workdir_base=tmp_path, timeout=120)
