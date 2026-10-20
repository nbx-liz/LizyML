"""H-0130 (#362): what closing the layer exceptions must not change.

Criterion 4: the moved helpers are the same objects at their old names.
Criterion 5: the SHAP importance needs the provider's pipeline factory.
Criterion 6: the result types stay where pickles name them, the plots accept
exactly the calibration-result marker, and ``RefitResultView`` mirrors
``RefitResult`` field for field, name and type.
"""

from __future__ import annotations

import dataclasses
import pickle
import pickletools
import typing
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import pytest

from lizyml.calibration.base import BaseCalibrationResult
from lizyml.calibration.cross_fit import CalibrationResult
from lizyml.calibration.platt import PlattCalibrator
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.estimators.base import BaseEstimatorAdapter
from lizyml.plots.calibration import _require_calibrator
from lizyml.training.refit_trainer import RefitResult
from lizyml.training.results import RefitResultView

FIXTURE = Path(__file__).parent / "data" / "calibration_result_e8b4c4e.pkl"


# --- Criterion 4: the moves keep identity -------------------------------------


def test_the_version_is_the_generated_one() -> None:
    import lizyml
    import lizyml._version
    import lizyml.core.model

    assert lizyml.__version__ == lizyml._version.__version__
    assert lizyml.core.model.__version__ is lizyml._version.__version__


def test_the_ratio_ranges_are_one_declaration() -> None:
    import lizyml.config.schema as schema
    import lizyml.core.types.smart_ratio as smart_ratio

    assert schema.SMART_RATIO_RANGES is smart_ratio.SMART_RATIO_RANGES
    assert schema.RatioRange is smart_ratio.RatioRange


def test_the_oof_mask_is_one_function() -> None:
    import lizyml.core.types.oof_coverage as oof_coverage
    import lizyml.training.oof_assembly as oof_assembly

    assert oof_assembly.compute_oof_valid_mask is oof_coverage.compute_oof_valid_mask


# --- Criterion 5: SHAP needs the provider's factory ---------------------------


def test_shap_importance_without_a_pipeline_factory_is_refused() -> None:
    from lizyml.explain.shap_explainer import compute_shap_importance

    with pytest.raises(TypeError, match="pipeline_factory"):
        compute_shap_importance(  # type: ignore[call-arg]
            models=[],
            X=None,  # type: ignore[arg-type]
            splits_outer=[],
            task="regression",
            feature_names=["a"],
            pipeline_state=None,
        )


# --- Criterion 6(a): the plots accept exactly the marker ----------------------

#: The refusal of e8b4c4e, word for word (``plots/calibration.py:34-41``).
REFUSAL_MESSAGE = (
    "Calibration plots require calibration to be enabled. "
    "Set calibration.method in the config."
)


@pytest.mark.parametrize(
    ("calibrator", "type_name"),
    [(None, "NoneType"), (object(), "object")],
    ids=["none", "plain object"],
)
def test_a_calibrator_without_the_marker_is_refused_as_before(
    calibrator: object, type_name: str
) -> None:
    with pytest.raises(LizyMLError) as caught:
        _require_calibrator(SimpleNamespace(calibrator=calibrator))  # type: ignore[arg-type]
    assert caught.value.code is ErrorCode.CALIBRATION_NOT_SUPPORTED
    assert caught.value.user_message == REFUSAL_MESSAGE
    assert caught.value.context == {"calibrator_type": type_name}


def test_a_marker_subclass_is_accepted() -> None:
    class OtherResult(BaseCalibrationResult):
        def __init__(self) -> None:
            self.calibrated_oof = np.array([0.5])

    result = OtherResult()
    assert _require_calibrator(SimpleNamespace(calibrator=result)) is result  # type: ignore[arg-type]


def test_a_calibration_result_is_a_marker_instance() -> None:
    result = CalibrationResult(
        c_final=PlattCalibrator(),
        calibrated_oof=np.array([0.5]),
        method="platt",
        split_indices=[],
    )
    assert isinstance(result, BaseCalibrationResult)
    assert _require_calibrator(SimpleNamespace(calibrator=result)) is result  # type: ignore[arg-type]


def test_a_calibration_result_pickled_before_h0130_is_accepted() -> None:
    """The fixture was written by the e8b4c4e code (criterion 6(a2))."""
    loaded = pickle.loads(FIXTURE.read_bytes())  # noqa: S301 - committed fixture
    assert type(loaded) is CalibrationResult
    assert isinstance(loaded, BaseCalibrationResult)
    np.testing.assert_array_equal(loaded.calibrated_oof, [0.25, 0.75])
    assert _require_calibrator(SimpleNamespace(calibrator=loaded)) is loaded  # type: ignore[arg-type]


# --- Criterion 6(b): pickles name the classes where they were -----------------


def _globals(data: bytes) -> set[tuple[str, str]]:
    """Every (module, name) a pickle loads by reference."""
    found: set[tuple[str, str]] = set()
    strings: list[str] = []
    for opcode, arg, _ in pickletools.genops(data):
        if opcode.name == "GLOBAL":
            module, name = str(arg).split(" ", 1)
            found.add((module, name))
        elif opcode.name == "STACK_GLOBAL":
            found.add((strings[-2], strings[-1]))
        elif isinstance(arg, str):
            strings.append(arg)
    return found


def test_the_fixture_names_the_class_where_it_still_is() -> None:
    names = _globals(FIXTURE.read_bytes())
    assert ("lizyml.calibration.cross_fit", "CalibrationResult") in names


def test_the_fixture_uses_numpy_1_compatible_names() -> None:
    """numpy 1.24.0, the declared floor, cannot load this pickle when it names
    ``numpy._core`` (as numpy 2 writes it): PR #367's first fixture failed so in
    the lowest-deps lane. The fixture must name ``numpy.core`` instead, which
    numpy 2 still loads. This pins numpy-1-compatible names; it does not prove
    which numpy version wrote the file."""
    modules = {module for module, _ in _globals(FIXTURE.read_bytes())}
    assert "numpy.core.multiarray" in modules
    assert not any(module.startswith("numpy._core") for module in modules)


@pytest.mark.parametrize("protocol", [2, 4, pickle.HIGHEST_PROTOCOL])
def test_new_pickles_name_the_classes_where_they_were(protocol: int) -> None:
    calibration = CalibrationResult(
        c_final=PlattCalibrator(),
        calibrated_oof=np.array([0.5]),
        method="platt",
        split_indices=[],
    )
    refit = RefitResult(
        model=None,  # type: ignore[arg-type]
        pipeline_state={},
        feature_names=["a"],
        categorical_features=[],
        best_iteration=None,
        train_pred=np.array([0.0]),
        history={},
    )
    calibration_names = _globals(pickle.dumps(calibration, protocol=protocol))
    refit_names = _globals(pickle.dumps(refit, protocol=protocol))
    assert ("lizyml.calibration.cross_fit", "CalibrationResult") in calibration_names
    assert ("lizyml.training.refit_trainer", "RefitResult") in refit_names
    assert not {m for m, _ in calibration_names} & {"lizyml.calibration.base"}
    assert not {m for m, _ in refit_names} & {"lizyml.training.results"}


# --- Criterion 6(c): RefitResultView mirrors RefitResult ----------------------


def view_mismatches(view: type, result: type) -> list[str]:
    """How *view*'s annotations differ from *result*'s fields (name and type)."""
    view_hints = typing.get_type_hints(view)
    result_hints = typing.get_type_hints(result)
    fields = {f.name for f in dataclasses.fields(result)}
    problems = []
    if set(result_hints) != fields:
        problems.append(f"{result.__name__} hints {sorted(result_hints)} != fields")
    for name in sorted(set(view_hints) | fields):
        if name not in view_hints:
            problems.append(f"{name}: missing from {view.__name__}")
        elif name not in fields:
            problems.append(f"{name}: not a field of {result.__name__}")
        elif view_hints[name] != result_hints[name]:
            problems.append(f"{name}: {view_hints[name]!r} != {result_hints[name]!r}")
    return problems


def test_refit_result_view_mirrors_refit_result() -> None:
    assert view_mismatches(RefitResultView, RefitResult) == []


class _Widened(Protocol):
    model: BaseEstimatorAdapter
    pipeline_state: Any
    feature_names: Any
    categorical_features: list[str]
    best_iteration: int | None
    train_pred: npt.NDArray[np.float64]
    history: dict[str, Any]


class _Retyped(Protocol):
    model: BaseEstimatorAdapter
    pipeline_state: Any
    feature_names: list[str]
    categorical_features: list[str]
    best_iteration: float | None
    train_pred: npt.NDArray[np.float64]
    history: dict[str, Any]


class _Missing(Protocol):
    model: BaseEstimatorAdapter
    pipeline_state: Any
    feature_names: list[str]
    categorical_features: list[str]
    best_iteration: int | None
    train_pred: npt.NDArray[np.float64]


@pytest.mark.parametrize(
    ("view", "expected"),
    [
        (_Widened, "feature_names: typing.Any != list[str]"),
        (_Retyped, "best_iteration: float | None != int | None"),
        (_Missing, "history: missing from _Missing"),
    ],
    ids=["widened to Any", "retyped", "field removed"],
)
def test_the_comparison_catches_drift(view: type, expected: str) -> None:
    assert view_mismatches(view, RefitResult) == [expected]
