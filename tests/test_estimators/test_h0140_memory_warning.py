"""H-0140: the pre-training memory warning, max_depth <= 0 and one-sided bounds.

No test trains a large model. ``lightgbm.train`` is replaced by a stub that
records what LightGBM would receive and trains a 4-leaf, 1-round booster
instead (or raises before training), so the warning's inputs are compared with
the real hand-off and fit / tune run end to end at no memory cost.
"""

from __future__ import annotations

import json
import logging
import math
import warnings
from pathlib import Path
from typing import Any

import lightgbm
import pandas as pd
import pytest

from lizyml import Model
from lizyml.core._resource_warning import WarningLatch, check_resources
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.train_components import TrainComponents
from lizyml.estimators.lgbm import smart_params
from lizyml.estimators.lgbm.adapter import LGBMAdapter
from lizyml.estimators.lgbm.memory_check import (
    BYTES_THRESHOLD,
    LEAVES_THRESHOLD,
    LIGHTGBM_DEFAULT_NUM_LEAVES,
    MemoryAssessment,
    estimate_bytes,
    finalized,
    message,
)
from lizyml.estimators.lgbm.param_names import LGBM_CANONICAL_NAME, accepted_spellings
from lizyml.estimators.lgbm.provider import LGBMProvider
from lizyml.estimators.provider import ResourceWarningProvider
from lizyml.tuning.search_space import IntDim, attach_bounds, detect_boundary
from tests._helpers import make_config, make_multiclass_df, make_regression_df

_REAL_TRAIN = lightgbm.train
_MEMORY = "allocates memory for every configured leaf"


def _canonical(params: dict[str, Any], name: str) -> Any:
    return next(
        (v for k, v in params.items() if LGBM_CANONICAL_NAME.get(k, k) == name), None
    )


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
            if LGBM_CANONICAL_NAME.get(k, k) not in ("num_leaves", "max_depth")
        }
        small.update(num_leaves=4, max_depth=2)
        return _REAL_TRAIN(small, train_set, num_boost_round=1, **kw)


@pytest.fixture
def train_stub(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    recorder = _Recorder()
    monkeypatch.setattr(lightgbm, "train", recorder)
    return recorder


def _config(**model: Any) -> dict[str, Any]:
    auto = model.pop("auto_num_leaves", True)
    n_splits = model.pop("n_splits", 2)
    cfg = make_config("regression", n_splits=n_splits, **model)
    cfg["model"]["auto_num_leaves"] = auto
    return cfg


def _with_tuning(cfg: dict[str, Any], n_trials: int) -> dict[str, Any]:
    """Tune only learning_rate, so the configured leaves stay as they are."""
    cfg["tuning"] = {
        "optuna": {
            "params": {"n_trials": n_trials},
            "space": {"learning_rate": {"type": "float", "low": 0.05, "high": 0.1}},
            "space_mode": "replace",
        }
    }
    return cfg


def _memory_warnings(record: list[warnings.WarningMessage]) -> list[str]:
    return [str(w.message) for w in record if _MEMORY in str(w.message)]


def _fit(cfg: dict[str, Any]) -> tuple[Model, list[str]]:
    model = Model(cfg)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model.fit(data=make_regression_df(n=80))
    return model, _memory_warnings(record)


# --- acceptance 1 and 5: auto-derived from an unlimited depth ----------------------


@pytest.mark.parametrize("depth", [-1, 0])
def test_an_unlimited_depth_warns_once_before_training(
    depth: int, train_stub: _Recorder
) -> None:
    _, found = _fit(_config(max_depth=depth))
    assert len(found) == 1  # all folds and the refit together
    assert "num_leaves=131072" in found[0]
    assert f"derived by auto_num_leaves from max_depth={depth}" in found[0]
    assert "GiB" in found[0] and "shallower max_depth" in found[0]
    assert all(_canonical(p, "num_leaves") == 131072 for p, _ in train_stub.calls)


def test_the_warning_points_at_the_fit_call(train_stub: _Recorder) -> None:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        Model(_config(max_depth=-1)).fit(data=make_regression_df(n=80))
    hits = [w for w in record if _MEMORY in str(w.message)]
    assert [Path(w.filename).name for w in hits] == [Path(__file__).name]
    assert all(w.category is UserWarning for w in hits)


# --- acceptance 2: explicit values, aliases and LightGBM's default -----------------


@pytest.mark.parametrize("spelling", sorted(accepted_spellings("num_leaves")))
def test_an_explicit_num_leaves_warns_under_every_spelling(
    spelling: str, train_stub: _Recorder
) -> None:
    _, found = _fit(_config(auto_num_leaves=False, **{spelling: LEAVES_THRESHOLD}))
    assert len(found) == 1
    assert f"num_leaves={LEAVES_THRESHOLD} (set in the parameters)" in found[0]
    assert "max_depth" not in found[0]  # the advice fits the cause
    assert {_canonical(p, "num_leaves") for p, _ in train_stub.calls} == {
        LEAVES_THRESHOLD
    }


def test_lightgbm_default_leaves_are_checked_as_31(train_stub: _Recorder) -> None:
    # 31 leaves x 200000 rounds x 3 boosters crosses 1 GiB.
    _, found = _fit(_config(auto_num_leaves=False, n_estimators=200_000))
    assert len(found) == 1
    assert f"num_leaves={LIGHTGBM_DEFAULT_NUM_LEAVES} (LightGBM's default)" in found[0]
    assert "max_depth" not in found[0]
    assert all(_canonical(p, "num_leaves") is None for p, _ in train_stub.calls)


# --- acceptance 3: the estimate alone, one booster and the folds kept alive --------


def test_the_folds_kept_alive_cross_the_byte_threshold(train_stub: _Recorder) -> None:
    base = {"auto_num_leaves": False, "num_leaves": 8192, "n_estimators": 500}
    per_booster = estimate_bytes(8192, 0, 500, 1, 1)
    assert per_booster < BYTES_THRESHOLD and LEAVES_THRESHOLD > 8192
    _, few = _fit(_config(n_splits=2, **base))  # 3 boosters
    _, many = _fit(_config(n_splits=20, **base))  # 21 boosters
    assert few == []
    assert len(many) == 1 and "21 booster(s)" in many[0]


def test_a_tune_trial_counts_its_folds(train_stub: _Recorder) -> None:
    cfg = _config(n_splits=20, auto_num_leaves=False, num_leaves=8192, n_estimators=500)
    cfg = _with_tuning(cfg, 1)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        Model(cfg).tune(data=make_regression_df(n=80))
    found = _memory_warnings(record)
    assert len(found) == 1 and "20 booster(s)" in found[0]


# --- acceptance 4: the defaults ----------------------------------------------------


def test_the_default_fit_does_not_warn(train_stub: _Recorder) -> None:
    _, found = _fit(make_config("regression", n_splits=2))
    assert found == []


def test_the_initial_default_space_never_reaches_the_leaf_threshold() -> None:
    """max_depth 3-12 and num_leaves_ratio <= 1 derive at most 4096 leaves."""
    leaves = {
        smart_params._compute_num_leaves(depth, ratio)  # noqa: SLF001
        for depth in range(3, 13)
        for ratio in (0.5, 0.75, 1.0)
    }
    assert max(leaves) == 4096 < LEAVES_THRESHOLD


# --- acceptance 6: one warning per tune call ---------------------------------------


def test_tune_warns_once_per_call_and_logs_the_rest(
    train_stub: _Recorder, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    cfg = _config(max_depth=-1)
    cfg = _with_tuning(cfg, 3)
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    data = make_regression_df(n=80)
    for resume in (False, True):
        caplog.clear()
        with (
            warnings.catch_warnings(record=True) as record,
            caplog.at_level(logging.INFO, logger="lizyml"),
        ):
            warnings.simplefilter("always")
            Model(cfg).tune(data=data, storage=storage, study_name="s", resume=resume)
        assert len(_memory_warnings(record)) == 1
        logged = [r for r in caplog.records if "tune.trial.memory_warning" in r.message]
        assert len(logged) == 2


# --- acceptance 7: warnings as errors stop before any training ---------------------


def test_warnings_as_errors_stop_fit_before_training(train_stub: _Recorder) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning, match=_MEMORY):
            Model(_config(max_depth=-1)).fit(data=make_regression_df(n=80))
    assert train_stub.calls == []


def test_warnings_as_errors_stop_tune_before_training(train_stub: _Recorder) -> None:
    """The trial's warning, raised as an error, ends the study: the tuner
    reports it as TUNING_FAILED carrying the warning text."""
    cfg = _config(max_depth=-1)
    cfg = _with_tuning(cfg, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(LizyMLError) as exc:
            Model(cfg).tune(data=make_regression_df(n=80))
    assert exc.value.code == ErrorCode.TUNING_FAILED
    assert _MEMORY in exc.value.user_message
    assert train_stub.calls == []


# --- acceptance 8: max_depth = 0 ----------------------------------------------------


def test_zero_depth_is_unlimited() -> None:
    assert smart_params._compute_num_leaves(0, 1.0) == 131072  # noqa: SLF001
    assert smart_params._compute_num_leaves(-1, 1.0) == 131072  # noqa: SLF001
    assert smart_params._compute_num_leaves(5, 1.0) == 32  # noqa: SLF001


def test_a_fresh_zero_depth_fit_exports_131072(
    train_stub: _Recorder, tmp_path: Path
) -> None:
    model, _ = _fit(_config(max_depth=0))
    model.export_code(tmp_path / "gen")
    config = json.loads((tmp_path / "gen" / "config.json").read_text())
    assert _canonical(config["lgbm_params"], "num_leaves") == 131072


def test_a_saved_zero_depth_model_keeps_its_8_leaves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fitted under the old rule (0 derived 8): load and export_code keep 8."""

    def old_rule(max_depth: int | None, ratio: float) -> int:
        base = 131072 if max_depth is None or max_depth < 0 else 2**max_depth
        return max(8, min(131072, math.ceil(base * ratio)))

    df = make_regression_df(n=80)
    X = df.drop(columns=["target"])
    with monkeypatch.context() as patch:
        patch.setattr(smart_params, "_compute_num_leaves", old_rule)
        model = Model(_config(max_depth=0))
        model.fit(data=df)
    expected = model.predict(X).pred
    model.export(tmp_path / "artifact")
    loaded = Model.load(tmp_path / "artifact")
    pd.testing.assert_series_equal(
        pd.Series(loaded.predict(X).pred), pd.Series(expected)
    )
    loaded.export_code(tmp_path / "gen")
    config = json.loads((tmp_path / "gen" / "config.json").read_text())
    assert _canonical(config["lgbm_params"], "num_leaves") == 8


# --- acceptance 9: the estimate and its inputs -------------------------------------


def test_the_estimate_formula() -> None:
    # The H-0140 measurement: 4096 leaves, 300 rounds, 3 features of 150
    # bins, 4 boosters kept alive -> 403 MiB predicted, 436 MiB measured.
    assert estimate_bytes(4096, 450, 300, 1, 4) == 16 * 450 * 4096 + 80 * 4096 * 300 * 4
    assert round(estimate_bytes(4096, 450, 300, 1, 4) / 2**20) == 403


@pytest.mark.parametrize("spelling", sorted(accepted_spellings("num_iterations")))
def test_the_rounds_are_the_ones_lightgbm_receives(
    spelling: str, train_stub: _Recorder
) -> None:
    adapter = LGBMAdapter(task="regression", params={spelling: 1234})
    _, rounds = finalized(adapter)
    df = make_regression_df(n=40)
    adapter.fit(df.drop(columns=["target"]), df["target"])
    assert rounds == 1234 == train_stub.calls[0][1]


@pytest.mark.parametrize("spelling", sorted(accepted_spellings("num_class")))
def test_a_num_class_override_is_the_one_lightgbm_receives(
    spelling: str, train_stub: _Recorder
) -> None:
    adapter = LGBMAdapter(task="multiclass", params={spelling: 7}, num_class=3)
    params, _ = finalized(adapter)
    df = make_multiclass_df(n=60)
    adapter.fit(df.drop(columns=["target"]), df["target"])
    assert _canonical(params, "num_class") == 7
    assert _canonical(train_stub.calls[0][0], "num_class") == 7


def test_a_num_class_override_reaches_the_estimate() -> None:
    def factory() -> LGBMAdapter:
        return LGBMAdapter(
            task="multiclass",
            params={"num_class": 100, "num_leaves": 1024, "n_estimators": 1500},
            num_class=3,
        )

    X = make_multiclass_df(n=60).drop(columns=["target"])
    text = LGBMProvider().resource_warning(
        estimator_factory=factory, X=X, smart_params={}, n_boosters=2
    )
    assert text is not None and "num_leaves=1024" in text


@pytest.mark.parametrize("spelling", sorted(accepted_spellings("max_bin")))
def test_max_bin_is_read_under_every_spelling(spelling: str) -> None:
    from lizyml.estimators.lgbm.memory_check import feature_bins

    X = pd.DataFrame({"a": range(500), "b": [1.5] * 500})
    assert feature_bins(X, {spelling: 63}) == 63 + 1


@pytest.mark.parametrize(
    "caps",
    [[8192, 8192], (8192, 8192), "8192,8192", " 8192 , 8192 "],
    ids=["list", "tuple", "string", "spaced-string"],
)
def test_max_bin_by_feature_is_read_in_every_form_lightgbm_accepts(caps: Any) -> None:
    """Implementation review round 1: a comma-separated string was ignored."""
    from lizyml.estimators.lgbm.memory_check import feature_bins

    X = pd.DataFrame({"a": range(10_000), "b": range(10_000)})
    assert feature_bins(X, {"max_bin": 63, "max_bin_by_feature": caps}) == 2 * 8192


def test_a_string_max_bin_by_feature_warns_end_to_end(train_stub: _Recorder) -> None:
    rows = 10_000
    df = pd.DataFrame({f"f{i}": range(rows) for i in range(10)})
    df["target"] = [float(i % 7) for i in range(rows)]
    cfg = _config(
        auto_num_leaves=False,
        num_leaves=8192,
        n_estimators=1,
        max_bin_by_feature=",".join(["8192"] * 10),
    )
    model = Model(cfg)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model.fit(data=df)
    assert len(_memory_warnings(record)) == 1
    assert all(
        _canonical(p, "max_bin_by_feature") == ",".join(["8192"] * 10)
        for p, _ in train_stub.calls
    )


# --- acceptance 12: max_depth has no upper bound ------------------------------------


def test_expansion_goes_past_30() -> None:
    bounds = LGBMProvider().parameter_bounds("regression")
    (dim,) = attach_bounds([IntDim("max_depth", 3, 30)], bounds)
    assert (dim.min_allowed, dim.max_allowed) == (-1, None)
    report = detect_boundary([dim], {"max_depth": 30})
    (status,) = report.dims
    assert status.expanded and status.new_high is not None and status.new_high > 30


def test_a_one_sided_bound_keeps_the_other_side_open() -> None:
    (low_only,) = attach_bounds([IntDim("x", 1, 5)], {"x": {"min": 0}})
    (high_only,) = attach_bounds([IntDim("x", 1, 5)], {"x": {"max": 9}})
    assert (low_only.min_allowed, low_only.max_allowed) == (0, None)
    assert (high_only.min_allowed, high_only.max_allowed) == (None, 9)


# --- acceptance 13: the optional capability ----------------------------------------


def test_the_lightgbm_provider_has_the_capability() -> None:
    assert isinstance(LGBMProvider(), ResourceWarningProvider)


def test_a_provider_without_the_capability_is_not_checked() -> None:
    class Plain:  # no resource_warning
        pass

    def unused() -> Any:
        raise AssertionError("the factory must not be called")

    tc = TrainComponents(
        estimator_factory=unused,
        sample_weight=None,
        ratio_resolver=None,
        inner_valid=None,  # type: ignore[arg-type]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_resources(
            Plain(), tc, pd.DataFrame({"a": [1]}), {}, n_boosters=1, stacklevel=1
        )


def test_the_latch_warns_once() -> None:
    latch = WarningLatch()

    class Risky:
        def resource_warning(self, **_: Any) -> str:
            return "LightGBM " + _MEMORY

    tc = TrainComponents(
        estimator_factory=lambda: None,  # type: ignore[arg-type, return-value]
        sample_weight=None,
        ratio_resolver=None,
        inner_valid=None,  # type: ignore[arg-type]
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        for _ in range(3):
            check_resources(
                Risky(), tc, pd.DataFrame(), {}, n_boosters=1, stacklevel=1, latch=latch
            )
    assert len(record) == 1 and latch.warned


def test_the_message_names_each_cause() -> None:
    common = {"max_depth": None, "estimate_bytes": 2 << 30, "n_boosters": 3}
    derived = message(
        MemoryAssessment(131072, "derived", **{**common, "max_depth": -1})
    )
    explicit = message(MemoryAssessment(20000, "explicit", **common))
    default = message(MemoryAssessment(31, "default", **common))
    assert "shallower max_depth" in derived
    assert "max_depth" not in explicit and "max_depth" not in default
    assert (
        "about 2.0 GiB" in default and "isotonic calibration is not counted" in default
    )
