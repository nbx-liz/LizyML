"""Public-return isolation tests for ``Model.evaluate`` / ``Model.fit_result`` (H-0082).

``FitResult`` is a non-frozen dataclass with mutable nested dicts. Before H-0082
the unfiltered ``evaluate(None)`` and the ``fit_result`` property handed out live
internal references, so a caller mutating the returned object could corrupt
internal state — and because ``export()`` reads the live metrics, the worst case
was contaminated exported metadata (a reproducibility risk).

These tests pin the contract: public returns are independent deep copies; the
exported metadata is never affected by post-call mutation of a returned object.
"""

from __future__ import annotations

import json

import pytest

from lizyml import Model
from tests._helpers import make_config, make_regression_df

_SENTINEL = -999.0


@pytest.fixture(scope="module")
def fitted_model() -> Model:
    model = Model(make_config("regression"))
    model.fit(data=make_regression_df(n=120))
    return model


def _oof_key(metrics: dict) -> str:
    return next(iter(metrics["raw"]["oof"]))


def test_evaluate_none_returns_independent_copy(fitted_model: Model) -> None:
    snapshot = fitted_model.evaluate()
    key = _oof_key(snapshot)
    snapshot["raw"]["oof"][key] = _SENTINEL

    # Internal state must be untouched by mutating the returned object.
    assert fitted_model.evaluate()["raw"]["oof"][key] != _SENTINEL


def test_evaluate_none_distinct_objects_equal_values(fitted_model: Model) -> None:
    first = fitted_model.evaluate()
    second = fitted_model.evaluate()
    assert first is not second
    assert first["raw"]["oof"] is not second["raw"]["oof"]
    assert first == second  # structure / values are bit-identical


def test_fit_result_property_returns_independent_copy(fitted_model: Model) -> None:
    fr = fitted_model.fit_result
    key = _oof_key(fr.metrics)
    fr.metrics["raw"]["oof"][key] = _SENTINEL

    assert fitted_model.fit_result.metrics["raw"]["oof"][key] != _SENTINEL


def test_fit_result_property_distinct_objects(fitted_model: Model) -> None:
    assert fitted_model.fit_result is not fitted_model.fit_result


def test_fit_result_shares_trained_estimators_by_reference(
    fitted_model: Model,
) -> None:
    # Selective copy: trained estimators are shared (deep-copying a Booster
    # drops params fidelity), so identity is preserved against internal state.
    internal = fitted_model._fit_result
    returned = fitted_model.fit_result
    assert returned is not internal
    assert returned.models[0] is internal.models[0]
    assert returned.calibrator is internal.calibrator
    assert returned.pipeline_state is internal.pipeline_state
    # Mutable data is copied, not shared.
    assert returned.metrics is not internal.metrics


def test_mutating_evaluate_does_not_contaminate_export(
    fitted_model: Model, tmp_path
) -> None:
    snapshot = fitted_model.evaluate()
    key = _oof_key(snapshot)
    snapshot["raw"]["oof"][key] = _SENTINEL

    out = fitted_model.export(path=tmp_path / "artifact")
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["metrics"]["raw"]["oof"][key] != _SENTINEL


# --- #204: fit() return value and load()'s metrics must also be isolated ------


def test_fit_return_value_is_independent_copy() -> None:
    """``fit()`` — the primary access path — must not hand out the internal
    object (H-0086 / #204). Mutating its return must not corrupt internal state.
    """
    model = Model(make_config("regression"))
    returned = model.fit(data=make_regression_df(n=120))

    assert returned is not model._fit_result
    key = _oof_key(returned.metrics)
    returned.metrics["raw"]["oof"][key] = _SENTINEL

    assert model.evaluate()["raw"]["oof"][key] != _SENTINEL
    # Selective copy: trained estimators stay shared by reference.
    assert returned.models[0] is model._fit_result.models[0]


def test_mutating_fit_return_does_not_contaminate_export(tmp_path) -> None:
    model = Model(make_config("regression"))
    returned = model.fit(data=make_regression_df(n=120))
    key = _oof_key(returned.metrics)
    returned.metrics["raw"]["oof"][key] = _SENTINEL

    out = model.export(path=tmp_path / "artifact")
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["metrics"]["raw"]["oof"][key] != _SENTINEL


def test_load_metrics_not_shared_with_returned_fit_result(tmp_path) -> None:
    """After ``load()``, the internal ``_metrics`` dict must not be the same
    object the ``fit_result`` copy exposes, so mutating a public return cannot
    corrupt the reloaded internal state (#204).
    """
    model = Model(make_config("regression"))
    model.fit(data=make_regression_df(n=120))
    out = model.export(path=tmp_path / "artifact")

    loaded = Model.load(out)
    returned = loaded.fit_result
    key = _oof_key(returned.metrics)
    returned.metrics["raw"]["oof"][key] = _SENTINEL

    assert loaded.evaluate()["raw"]["oof"][key] != _SENTINEL
    assert loaded._metrics is not loaded._fit_result.metrics
