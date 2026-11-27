"""H-0150 part A: ``Model.estimate_tune_memory`` (acceptance 2, 3, 4, 5, 7).

No test trains a large model: ``lightgbm.train`` is replaced by a stub that
records what LightGBM would receive and trains a tiny booster instead, so the
trials of a real ``tune`` can be compared with the envelope.
"""

from __future__ import annotations

import copy
import sqlite3
from pathlib import Path
from typing import Any

import lightgbm
import optuna
import pandas as pd
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types import MemoryEstimate, TuneMemoryEstimate
from lizyml.estimators.lgbm.memory_check import assess, feature_bins
from lizyml.estimators.lgbm.param_names import LGBM_CANONICAL_NAME
from lizyml.estimators.lgbm.provider import LGBMProvider
from tests._helpers import (
    make_binary_df,
    make_config,
    make_multiclass_df,
    make_regression_df,
)

_REAL_TRAIN = lightgbm.train
N_FOLDS = 2


class _Recorder:
    """Records each ``lgb.train`` call, then trains a tiny booster instead."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], int]] = []

    def __call__(
        self, params: dict[str, Any], train_set: Any, num_boost_round: int, **kw: Any
    ) -> Any:
        self.calls.append((dict(params), int(num_boost_round)))
        small = {
            k: v
            for k, v in params.items()
            if LGBM_CANONICAL_NAME.get(k, k)
            not in ("num_leaves", "max_depth", "device_type", "histogram_pool_size")
        }
        small.update(num_leaves=4, max_depth=2)
        return _REAL_TRAIN(small, train_set, num_boost_round=1, **kw)


@pytest.fixture
def train_stub(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    recorder = _Recorder()
    monkeypatch.setattr(lightgbm, "train", recorder)
    return recorder


def _df(task: str) -> pd.DataFrame:
    if task == "binary":
        return make_binary_df(n=90)
    if task == "multiclass":
        return make_multiclass_df(n=90)
    return make_regression_df(n=90)


def _cfg(
    task: str = "regression",
    *,
    space: dict[str, Any] | None = None,
    mode: str = "replace",
    n_trials: int = 4,
    auto: bool = True,
    **model: Any,
) -> dict[str, Any]:
    n_est = model.pop("n_estimators", 3)
    cfg = make_config(task, n_estimators=n_est, n_splits=N_FOLDS, **model)
    cfg["model"]["auto_num_leaves"] = auto
    tuning: dict[str, Any] = {"optuna": {"params": {"n_trials": n_trials}}}
    if space is not None:
        tuning["optuna"]["space"] = space
        tuning["optuna"]["space_mode"] = mode
    cfg["tuning"] = tuning
    return cfg


LR = {"learning_rate": {"type": "float", "low": 0.05, "high": 0.1}}


def _space(**dims: Any) -> dict[str, Any]:
    return {**copy.deepcopy(LR), **dims}


def _trial_estimates(rec: _Recorder, X: pd.DataFrame, auto: bool) -> list[int]:
    """The estimate of each trained trial, recomputed from what LightGBM got."""
    out: list[int] = []
    for i in range(0, len(rec.calls), N_FOLDS):
        params, rounds = rec.calls[i]
        out.append(
            assess(
                params,
                rounds,
                bins=feature_bins(X, params),
                auto_num_leaves=auto,
                n_boosters=N_FOLDS,
                n_features=X.shape[1],
            ).estimate_bytes
        )
    return out


def _estimate(cfg: dict[str, Any], df: pd.DataFrame, **kw: Any) -> TuneMemoryEstimate:
    est = Model(cfg).estimate_tune_memory(data=df, **kw)
    assert isinstance(est, TuneMemoryEstimate)
    return est


# --- acceptance 2: the envelope bounds the ordinary trials -------------------------


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_the_default_space_is_complete_and_bounds_every_trial(
    task: str, train_stub: _Recorder
) -> None:
    cfg = _cfg(task, n_trials=6)
    df = _df(task)
    est = _estimate(cfg, df)
    assert est.status == "complete"
    assert est.unresolved == () and est.waiting_trials == 0 and est.failures == ()
    assert est.round == 1 and est.n_candidates > 1 and est.envelope is not None
    Model(cfg).tune(data=df)
    X = df.drop(columns="target")
    sizes = _trial_estimates(train_stub, X, auto=True)
    assert len(sizes) == 6
    assert max(sizes) <= est.envelope.estimate_bytes
    assert est.envelope.n_boosters == N_FOLDS


def test_a_trial_at_the_corner_equals_the_envelope(train_stub: _Recorder) -> None:
    space = _space(
        n_estimators={"type": "categorical", "choices": [3, 6]},
        max_depth={"type": "categorical", "choices": [3, 5]},
    )
    cfg = _cfg(space=space, n_trials=12)
    df = _df("regression")
    est = _estimate(cfg, df)
    assert est.status == "complete" and est.envelope is not None
    Model(cfg).tune(data=df)
    sizes = _trial_estimates(train_stub, df.drop(columns="target"), auto=True)
    assert max(sizes) == est.envelope.estimate_bytes


def test_a_depth_range_with_zero_derives_131072_leaves(train_stub: _Recorder) -> None:
    space = _space(max_depth={"type": "int", "low": 0, "high": 4})
    est = _estimate(_cfg(space=space), _df("regression"))
    assert est.envelope is not None and est.envelope.num_leaves == 131072
    assert est.envelope.cause == "derived"


def test_a_stepped_dimension_uses_the_grid_upper_end(train_stub: _Recorder) -> None:
    space = _space(num_leaves={"type": "int", "low": 16, "high": 64, "step": 16})
    cfg = _cfg(space=space, auto=False, n_trials=4)
    df = _df("regression")
    est = _estimate(cfg, df)
    assert est.status == "complete" and est.envelope is not None
    assert est.envelope.num_leaves == 64 and est.envelope.cause == "explicit"
    Model(cfg).tune(data=df)
    sizes = _trial_estimates(train_stub, df.drop(columns="target"), auto=False)
    assert max(sizes) <= est.envelope.estimate_bytes


# --- acceptance 3: open inputs, unevaluable points, the device ---------------------


def test_a_name_outside_the_closed_set_is_incomplete() -> None:
    space = _space(histogram_pool_size={"type": "float", "low": 64.0, "high": 256.0})
    est = _estimate(_cfg(space=space), _df("regression"))
    assert est.status == "incomplete"
    assert "histogram_pool_size" in est.unresolved
    assert est.envelope is not None  # the maximum over the rest


def test_an_unevaluable_choice_is_recorded_and_the_rest_evaluated() -> None:
    space = _space(
        early_stopping_rounds={
            "type": "categorical",
            "choices": [None, 40],
            "category": "training",
        }
    )
    est = _estimate(_cfg(space=space), _df("regression"))
    assert est.status == "incomplete"
    assert est.failures and est.failures[0].startswith("TypeError")
    assert "early_stopping_rounds" in est.unresolved
    assert est.n_candidates >= 1 and est.envelope is not None  # 40 was evaluated


def test_a_round_count_below_one_is_unevaluable() -> None:
    space = _space(max_depth={"type": "int", "low": 3, "high": 6})
    est = _estimate(_cfg(space=space, n_estimators=-1), _df("regression"))
    assert est.status == "incomplete"
    assert est.failures
    assert all("at least 1" in f for f in est.failures)
    assert est.envelope is None and est.n_candidates == 0


def _fixed(monkeypatch: pytest.MonkeyPatch, **fixed: Any) -> None:
    real = LGBMProvider.default_fixed_params

    def patched(self: LGBMProvider, task: Any) -> dict[str, Any]:
        return {**real(self, task), **fixed}

    monkeypatch.setattr(LGBMProvider, "default_fixed_params", patched)


DEVICE_OFF_CPU: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
    ("model.params device_type=gpu", {"device_type": "gpu"}, {}),
    ("model.params device=cuda", {"device": "cuda"}, {}),
    (
        "categorical device [cpu, gpu]",
        {},
        {"device": {"type": "categorical", "choices": ["cpu", "gpu"]}},
    ),
    ("int device dimension", {}, {"device": {"type": "int", "low": 1, "high": 2}}),
]


@pytest.mark.parametrize(
    ("label", "params", "dims"), DEVICE_OFF_CPU, ids=[c[0] for c in DEVICE_OFF_CPU]
)
def test_a_possible_non_cpu_device_is_unresolved(
    label: str, params: dict[str, Any], dims: dict[str, Any]
) -> None:
    est = _estimate(_cfg(space=_space(**dims), **params), _df("regression"))
    assert est.status == "incomplete"
    assert "device_type" in est.unresolved


def test_a_fixed_device_is_unresolved_and_a_later_layer_hides_the_lower(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _df("regression")
    _fixed(monkeypatch, device_type="gpu")
    est = _estimate(_cfg(space=_space(), mode="merge"), df)
    assert "device_type" in est.unresolved
    _fixed(monkeypatch, device="cuda")  # an alias spelling of the fixed layer
    est = _estimate(_cfg(space=_space(), mode="merge"), df)
    assert "device_type" in est.unresolved


def test_a_fixed_cpu_hides_a_gpu_in_model_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fixed(monkeypatch, device_type="cpu")
    cfg = _cfg(space=_space(), mode="merge", device_type="gpu")
    est = _estimate(cfg, _df("regression"))
    assert "device_type" not in est.unresolved


def test_a_cpu_only_device_dimension_hides_a_gpu_in_model_params() -> None:
    dims = {"device": {"type": "categorical", "choices": ["cpu"]}}
    est = _estimate(_cfg(space=_space(**dims), device_type="gpu"), _df("regression"))
    assert "device_type" not in est.unresolved and est.status == "complete"


def test_the_device_is_decided_before_candidates_are_assembled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fixed GPU and a ``validation_ratio`` range holding 0 (an unevaluable
    point): the assembly failure must not hide the device."""
    _fixed(monkeypatch, device_type="gpu")
    space = _space(
        validation_ratio={
            "type": "float",
            "low": 0.0,
            "high": 0.3,
            "category": "training",
        }
    )
    est = _estimate(_cfg(space=space, mode="merge"), _df("regression"))
    assert "device_type" in est.unresolved and est.status == "incomplete"


# --- acceptance 4/5/7: the next round, WAITING trials, no writes, no-trial rounds --


def _url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'study.db'}"


def _dump(path: Path) -> dict[str, list[Any]]:
    con = sqlite3.connect(path)
    try:
        tables = [
            r[0]
            for r in con.execute(
                "select name from sqlite_master where type='table' order by name"
            )
        ]
        return {t: con.execute(f"select * from {t}").fetchall() for t in tables}  # noqa: S608
    finally:
        con.close()


def _tuned(
    tmp_path: Path, cfg: dict[str, Any], df: pd.DataFrame, **kw: Any
) -> tuple[Model, str]:
    url = _url(tmp_path)
    model = Model(cfg)
    model.tune(data=df, storage=url, study_name="s", **kw)
    return model, url


def test_an_inherited_range_is_the_previous_rounds(
    tmp_path: Path, train_stub: _Recorder
) -> None:
    space = _space(num_leaves={"type": "int", "low": 16, "high": 64, "step": 16})
    cfg = _cfg(space=space, auto=False, n_trials=3)
    df = _df("regression")
    _, url = _tuned(tmp_path, cfg, df, expand_boundary=False)
    est = Model(cfg).estimate_tune_memory(
        df, resume=True, storage=url, study_name="s", expand_boundary=False
    )
    assert est.round == 2 and est.waiting_trials == 0
    assert est.envelope is not None and est.envelope.num_leaves == 64


def test_a_new_space_replaces_the_range(tmp_path: Path, train_stub: _Recorder) -> None:
    space = _space(num_leaves={"type": "int", "low": 16, "high": 64, "step": 16})
    cfg = _cfg(space=space, auto=False, n_trials=3)
    df = _df("regression")
    _, url = _tuned(tmp_path, cfg, df, expand_boundary=False)
    wide = _space(num_leaves={"type": "int", "low": 16, "high": 128, "step": 16})
    est = Model(cfg).estimate_tune_memory(
        df,
        resume=True,
        storage=url,
        study_name="s",
        expand_boundary=False,
        space=wide,
    )
    assert est.envelope is not None and est.envelope.num_leaves == 128


def test_an_expanded_range_is_estimated(tmp_path: Path, train_stub: _Recorder) -> None:
    space = _space(num_leaves={"type": "int", "low": 4, "high": 8})
    cfg = _cfg(space=space, auto=False, n_trials=6)
    df = _df("regression")
    first, url = _tuned(
        tmp_path, cfg, df, expand_boundary=True, boundary_threshold=0.49
    )
    est = Model(cfg).estimate_tune_memory(
        df,
        resume=True,
        storage=url,
        study_name="s",
        expand_boundary=True,
        boundary_threshold=0.49,
    )
    result = first.tune(
        data=df, resume=True, expand_boundary=True, boundary_threshold=0.49
    )
    dim = {d.name: d for d in result.rounds[-1].space_snapshot}["num_leaves"]
    assert "num_leaves" in result.rounds[-1].expanded_dims  # the case is real
    assert est.envelope is not None and est.envelope.num_leaves == dim.high  # type: ignore[union-attr]
    assert dim.high > 8


def _crashing_close(monkeypatch: pytest.MonkeyPatch) -> None:
    from lizyml.core._model_tuning import ModelTuningMixin

    def boom(self: Any, *a: Any, **k: Any) -> None:
        raise RuntimeError("process died before the round closed")

    monkeypatch.setattr(ModelTuningMixin, "_close_round", boom)


def test_an_open_round_with_trials_left_uses_its_recorded_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, train_stub: _Recorder
) -> None:
    space = _space(num_leaves={"type": "int", "low": 16, "high": 64, "step": 16})
    cfg = _cfg(space=space, auto=False, n_trials=3)
    df = _df("regression")
    url = _url(tmp_path)
    monkeypatch.setattr(
        "lizyml.core._model_tuning.ModelTuningMixin._run_tune_round",
        lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("died at once")),
    )
    with pytest.raises(RuntimeError):
        Model(cfg).tune(data=df, storage=url, study_name="s")
    est = Model(cfg).estimate_tune_memory(df, resume=True, storage=url, study_name="s")
    assert est.round == 1 and est.n_candidates > 0
    assert est.envelope is not None and est.envelope.num_leaves == 64


def test_a_waiting_trial_makes_a_running_round_incomplete(
    tmp_path: Path, train_stub: _Recorder
) -> None:
    cfg = _cfg(space=_space(), n_trials=2)
    df = _df("regression")
    _, url = _tuned(tmp_path, cfg, df, expand_boundary=False)
    optuna.load_study(study_name="s", storage=url).enqueue_trial(
        {"learning_rate": 0.07}
    )
    est = Model(cfg).estimate_tune_memory(
        df, resume=True, storage=url, study_name="s", expand_boundary=False
    )
    assert est.status == "incomplete" and est.waiting_trials == 1
    assert est.envelope is not None  # the candidates are still evaluated


def test_a_round_with_no_trial_is_complete_and_evaluates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, train_stub: _Recorder
) -> None:
    cfg = _cfg(space=_space(), n_trials=2, device_type="gpu")
    df = _df("regression")
    url = _url(tmp_path)
    _crashing_close(monkeypatch)
    with pytest.raises(RuntimeError):
        Model(cfg).tune(data=df, storage=url, study_name="s")
    optuna.load_study(study_name="s", storage=url).enqueue_trial(
        {"learning_rate": 0.07}
    )
    # An open round whose trials are all done: nothing is left to run, and
    # neither the WAITING trial nor the GPU in model.params is looked at.
    est = Model(cfg).estimate_tune_memory(df, resume=True, storage=url, study_name="s")
    assert est.status == "complete"
    assert est.envelope is None and est.n_candidates == 0 and est.unresolved == ()


def test_the_study_is_not_written(tmp_path: Path, train_stub: _Recorder) -> None:
    cfg = _cfg(space=_space(), n_trials=2)
    df = _df("regression")
    _, url = _tuned(tmp_path, cfg, df, expand_boundary=False)
    optuna.load_study(study_name="s", storage=url).enqueue_trial(
        {"learning_rate": 0.07}
    )
    before = _dump(tmp_path / "study.db")
    Model(cfg).estimate_tune_memory(
        df, resume=True, storage=url, study_name="s", expand_boundary=False
    )
    assert _dump(tmp_path / "study.db") == before


def test_a_storage_without_the_study_is_planned_as_round_one(tmp_path: Path) -> None:
    url = _url(tmp_path)
    optuna.storages.RDBStorage(url)  # the file and its schema, no study
    cfg = _cfg(space=_space(), n_trials=2)
    est = Model(cfg).estimate_tune_memory(
        _df("regression"), resume=True, storage=url, study_name="new"
    )
    assert est.round == 1 and est.envelope is not None
    assert optuna.get_all_study_summaries(storage=url) == []


def _state(model: Model) -> dict[str, str]:
    return {k: repr(v) for k, v in vars(model).items()}


def test_the_model_is_unchanged(tmp_path: Path, train_stub: _Recorder) -> None:
    cfg = _cfg(space=_space(), n_trials=2)
    df = _df("regression")
    df["grp"] = pd.Categorical(["a", "b", "c"] * 30)
    model = Model(cfg)
    before = _state(model)
    model.estimate_tune_memory(df)
    assert _state(model) == before
    assert model._study is None and model._tuning_result is None
    model.tune(data=df)  # and a tune afterwards behaves as without the call
    assert model._tuning_result is not None


def test_a_resume_without_a_study_or_storage_is_the_tune_refusal() -> None:
    cfg = _cfg(space=_space())
    model = Model(cfg)
    with pytest.raises(LizyMLError) as est_err:
        model.estimate_tune_memory(_df("regression"), resume=True)
    with pytest.raises(LizyMLError) as tune_err:
        Model(cfg).tune(data=_df("regression"), resume=True)
    assert est_err.value.code is ErrorCode.TUNING_FAILED
    assert est_err.value.code is tune_err.value.code
    assert est_err.value.user_message == tune_err.value.user_message


def test_the_refusals_of_tune_are_the_estimates(tmp_path: Path) -> None:
    cfg = _cfg(space=_space())
    with pytest.raises(LizyMLError) as err:
        Model(cfg).estimate_tune_memory(_df("regression"), storage=_url(tmp_path))
    assert err.value.code is ErrorCode.CONFIG_INVALID  # storage needs study_name


# --- acceptance 7 (estimate part): the capability ------------------------------------


def test_a_provider_without_the_capability_is_refused_by_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Bare:
        """No capability at all."""

    for module in ("_model_memory", "_model_tune_memory"):
        monkeypatch.setattr(
            f"lizyml.core.{module}.get_provider", lambda _c: Bare(), raising=True
        )
    model = Model(_cfg(space=_space()))
    for call in (
        lambda: model.estimate_tune_memory(_df("regression")),
        lambda: model.estimate_memory(_df("regression")),
    ):
        with pytest.raises(LizyMLError) as err:
            call()
        assert err.value.code is ErrorCode.CONFIG_INVALID
        assert err.value.context["reason"] == "memory_estimate_unsupported"


def test_the_result_types_are_frozen() -> None:
    est = _estimate(_cfg(space=_space()), _df("regression"))
    assert isinstance(est.envelope, MemoryEstimate)
    with pytest.raises(AttributeError):
        est.status = "incomplete"  # type: ignore[misc]


def test_a_failing_point_alone_makes_it_incomplete() -> None:
    """No dimension is widened here, so only ``failures`` can say incomplete."""
    est = _estimate(_cfg(space=_space(), n_estimators=-1), _df("regression"))
    assert est.status == "incomplete" and est.unresolved == ()
    assert est.failures and est.envelope is None and est.n_candidates == 0
