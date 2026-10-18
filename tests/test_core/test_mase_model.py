"""MASE through the Model (H-0132, criteria 3, 4 and 8)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import lizyml.estimators.lgbm.param_validation as param_validation
from lizyml import Model
from lizyml.core._tuning_validation import check_value_dimensions
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.search_dim import CategoricalDim
from lizyml.splitters.base import BaseSplitter
from tests._helpers import make_config, make_regression_df
from tests.test_tuning._resume_helpers import fast_config, frame


def _config(**overrides: Any) -> dict[str, Any]:
    cfg = make_config("regression", n_estimators=5, n_splits=3, **overrides)
    cfg["evaluation"] = {"metrics": ["mase", "rmse"]}
    return cfg


def test_the_feval_is_bound_to_each_folds_inner_train_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With inner valid on, each fold's booster trains on the inner-train part
    of its outer-train rows; the feval's scale must come from exactly those."""
    bound: list[npt.NDArray[Any]] = []
    original = param_validation.resolve_metrics

    def spy(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("y_train") is not None:
            bound.append(np.asarray(kwargs["y_train"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(param_validation, "resolve_metrics", spy)
    cfg = _config(metric=["mase"])
    cfg["training"]["early_stopping"] = {
        "enabled": True,
        "rounds": 3,
        "inner_valid": {"method": "holdout", "ratio": 0.25},
    }
    df = make_regression_df(n=120)
    model = Model(cfg, data=df)
    model.fit()
    y = df["target"].to_numpy()
    splits = model.fit_result.splits
    assert splits.inner is not None
    for k, (outer_train, _) in enumerate(splits.outer):
        inner_train, inner_valid = splits.inner[k]
        expected = y[outer_train][inner_train]
        assert np.array_equal(bound[k], expected)
        assert len(set(inner_valid.tolist()) & set(inner_train.tolist())) == 0
    assert "mase" in model.fit_result.history[0]["eval_history"]["valid_0"]


def test_metadata_routes_accept_an_unbound_mase(tmp_path: Path) -> None:
    model = Model(_config(metric=["mase", "l1"]), data=make_regression_df(n=90))
    model.fit()
    assert model.params_table() is not None
    model.export_code(tmp_path / "code")
    check_value_dimensions(
        "lgbm", [CategoricalDim("metric", ("mase", "l1"))], "regression"
    )


def test_mase_is_a_tune_objective_minimised(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each trial's score is the Evaluator's pooled OOF MASE (criterion 4)."""
    from lizyml.evaluation.evaluator import Evaluator

    pooled: list[float] = []
    original = Evaluator.evaluate

    def spy(self: Evaluator, *args: Any, **kwargs: Any) -> Any:
        out = original(self, *args, **kwargs)
        pooled.append(out["raw"]["oof"]["mase"])
        return out

    monkeypatch.setattr(Evaluator, "evaluate", spy)
    cfg = fast_config(n_trials=2)
    cfg["evaluation"] = {"metrics": ["mase"]}
    model = Model(cfg, data=frame())
    result = model.tune()
    assert result.direction == "minimize"
    scores = [t.score for t in result.trials if t.state == "complete"]
    assert len(scores) == 2
    # The tune evaluates once per trial, in trial order.
    assert scores == pytest.approx(pooled[: len(scores)], rel=1e-12)
    assert result.best_score == pytest.approx(min(scores), rel=1e-12)


def test_maximising_mase_is_refused() -> None:
    cfg = fast_config(n_trials=1)
    cfg["evaluation"] = {"metrics": ["mase"]}
    cfg["tuning"]["optuna"]["params"]["direction"] = "maximize"
    with pytest.raises(LizyMLError) as caught:
        Model(cfg, data=frame()).tune()
    assert caught.value.code == ErrorCode.CONFIG_INVALID


class _Overlapping(BaseSplitter):
    """Row 0 is in both validation folds."""

    def split(
        self,
        n_samples: int,
        y: npt.NDArray[Any] | None = None,
        groups: npt.NDArray[Any] | None = None,
    ) -> Iterator[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]]:
        half = n_samples // 2
        rows = np.arange(n_samples, dtype=np.intp)
        yield rows[half:], rows[:half]
        yield rows[1:half], np.concatenate([rows[:1], rows[half:]])


def _overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    import lizyml.core._model_tuning as model_tuning
    import lizyml.core.model as model_module

    def build(*args: Any, **kwargs: Any) -> BaseSplitter:
        return _Overlapping()

    monkeypatch.setattr(model_module, "build_splitter", build)
    monkeypatch.setattr(model_tuning, "build_splitter", build)


def test_fit_refuses_overlapping_validation_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _overlap(monkeypatch)
    with pytest.raises(LizyMLError) as caught:
        Model(_config(), data=make_regression_df(n=60)).fit()
    assert caught.value.code == ErrorCode.UNSUPPORTED_METRIC


def test_tune_fails_every_trial_on_overlapping_validation_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _overlap(monkeypatch)
    cfg = fast_config(n_trials=2)
    cfg["evaluation"] = {"metrics": ["mase"]}
    with pytest.raises(LizyMLError) as caught:
        Model(cfg, data=frame()).tune()
    assert caught.value.code == ErrorCode.TUNING_FAILED
    assert "mase" in caught.value.context["first_error"].lower()


def test_a_loaded_model_returns_the_stored_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = Model(_config(), data=make_regression_df(n=90))
    model.fit()
    stored = model.evaluate(metrics=["mase"])
    path = model.export(tmp_path / "artifact")

    def no_recompute(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("MASE was recomputed after load")

    from lizyml.metrics.regression import MASE

    monkeypatch.setattr(MASE, "bind_train", no_recompute)
    loaded = Model.load(path)
    assert loaded.evaluate(metrics=["mase"]) == stored
