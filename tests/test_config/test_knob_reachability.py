"""Every defaulted constructor knob is reachable or stated (H-0108, #268).

The population is computed now, not typed: ``knob_census`` sweeps ``lizyml/``.
``_knob_registry.REGISTRY`` must classify exactly that population; every
``config`` row has executed cells here (a non-default value set at the Config
path reaches the constructor on the production call), one per path whose source
differs; every row of another kind is stated in ``BLUEPRINT.md`` §5.5 with the
same kind.

#268 classified knobs by matching parameter names against Config field names.
That reported ``max_train_size`` (Config: ``train_size_max``), ``PrecisionAtK.k``
(dict-form metric entries, H-0065) and ``early_stopping_rounds`` (Config:
``training.early_stopping.rounds``) as unreachable when all of them were. These
cells execute the path instead of comparing names.
"""

from __future__ import annotations

import inspect
import re
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml.calibration.beta import BetaCalibrator
from lizyml.calibration.isotonic import IsotonicCalibrator
from lizyml.calibration.platt import PlattCalibrator
from lizyml.core.model import Model
from lizyml.estimators.lgbm.adapter import LGBMAdapter
from lizyml.evaluation.evaluator import Evaluator
from lizyml.features.encoders.categorical_encoder import CategoricalEncoder
from lizyml.features.pipelines_native import NativeFeaturePipeline
from lizyml.metrics.classification import ECE, PrecisionAtK
from lizyml.metrics.regression import HuberLoss
from lizyml.splitters.blocked_group_kfold import BlockedGroupKFoldSplitter
from lizyml.splitters.group_kfold import (
    GroupKFoldSplitter,
    StratifiedGroupKFoldSplitter,
)
from lizyml.splitters.group_time_series import GroupTimeSeriesSplitter
from lizyml.splitters.kfold import KFoldSplitter, StratifiedKFoldSplitter
from lizyml.splitters.purged_time_series import PurgedTimeSeriesSplitter
from lizyml.splitters.time_series import TimeSeriesSplitter
from lizyml.training.cv_trainer import CVTrainer
from lizyml.training.inner_valid import (
    BlockedGroupInnerValid,
    GroupHoldoutInnerValid,
    HoldoutInnerValid,
    StratifiedTimeHoldoutInnerValid,
    TimeHoldoutInnerValid,
)
from lizyml.training.refit_trainer import RefitTrainer
from lizyml.tuning.tuner import Tuner
from tests.test_config._knob_registry import KINDS, REGISTRY, knob_census

_ROOT = Path(__file__).resolve().parents[2]

_SPIED = [
    BetaCalibrator, IsotonicCalibrator, PlattCalibrator, LGBMAdapter, Evaluator,
    CategoricalEncoder, NativeFeaturePipeline, ECE, PrecisionAtK, HuberLoss,
    BlockedGroupKFoldSplitter, GroupKFoldSplitter, StratifiedGroupKFoldSplitter,
    GroupTimeSeriesSplitter, KFoldSplitter, StratifiedKFoldSplitter,
    PurgedTimeSeriesSplitter, TimeSeriesSplitter, CVTrainer, RefitTrainer,
    HoldoutInnerValid, GroupHoldoutInnerValid, TimeHoldoutInnerValid,
    BlockedGroupInnerValid, StratifiedTimeHoldoutInnerValid, Tuner,
]  # fmt: skip


@contextmanager
def _spying() -> Iterator[dict[str, list[dict[str, Any]]]]:
    """Record the bound ``__init__`` arguments of every spied class."""
    received: dict[str, list[dict[str, Any]]] = defaultdict(list)
    originals = {cls: cls.__init__ for cls in _SPIED}

    def wrap(cls: type, real: Callable[..., None]) -> Callable[..., None]:
        sig = inspect.signature(real)

        def init(self: Any, *args: Any, **kwargs: Any) -> None:
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            received[cls.__name__].append(dict(list(bound.arguments.items())[1:]))
            real(self, *args, **kwargs)

        return init

    try:
        for cls, real in originals.items():
            cls.__init__ = wrap(cls, real)  # type: ignore[method-assign]
        yield received
    finally:
        for cls, real in originals.items():
            cls.__init__ = real  # type: ignore[method-assign]


_N = 240


def _frame(task: str) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=_N), "b": rng.normal(size=_N)})
    df["g"] = np.repeat(np.arange(24), _N // 24)
    df["t"] = np.arange(_N)
    df["blk"] = np.repeat(np.arange(4), _N // 4)
    signal = df["a"] + rng.normal(scale=0.5, size=_N)
    df["y"] = signal if task == "regression" else (signal > 0).astype(int)
    return df


def _config(task: str, split: dict[str, Any], **extra: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "config_version": 1,
        "task": task,
        "data": {"target": "y"},
        "features": {"exclude": ["g", "t", "blk"]},
        "split": split,
        "model": {"name": "lgbm", "params": {"n_estimators": 5}},
    }
    if split["method"] in (
        "group_kfold",
        "stratified_group_kfold",
        "group_time_series",
    ):
        cfg["data"]["group_col"] = "g"
    if split["method"] in ("time_series", "purged_time_series", "group_time_series"):
        cfg["data"]["time_col"] = "t"
    cfg.update(extra)
    return cfg


def _fit(task: str, split: dict[str, Any], **extra: Any) -> Callable[[], object]:
    return lambda: Model(_config(task, split, **extra)).fit(data=_frame(task))


def _callback(info: Any) -> None:  # noqa: ARG001
    return None


_ES_HOLDOUT = {
    "enabled": True,
    "rounds": 7,
    "inner_valid": {
        "method": "holdout",
        "ratio": 0.2,
        "random_state": 7,
        "stratify": True,
    },
}
#: Early stopping left to automatic inner-valid resolution, with a ratio.
_ES_AUTO = {"enabled": True, "validation_ratio": 0.3}

#: scenario -> the call that builds the classes under the configured values.
_SCENARIOS: dict[str, Callable[[], object]] = {
    "kfold": _fit(
        "regression",
        {"method": "kfold", "n_splits": 3, "shuffle": False, "random_state": 7},
        model={"name": "lgbm", "params": {"n_estimators": 5, "learning_rate": 0.07}},
    ),
    "kfold_seed_fallback": _fit(
        "regression", {"method": "kfold", "n_splits": 3}, training={"seed": 17}
    ),
    "stratified_kfold": _fit(
        "binary", {"method": "stratified_kfold", "n_splits": 3, "random_state": 7}
    ),
    "stratified_kfold_auto": _fit(
        "binary",
        {"method": "stratified_kfold", "n_splits": 3},
        training={"seed": 17, "early_stopping": _ES_AUTO},
    ),
    "group_kfold": _fit("regression", {"method": "group_kfold", "n_splits": 3}),
    "group_kfold_auto": _fit(
        "regression",
        {"method": "group_kfold", "n_splits": 3},
        training={"seed": 17, "early_stopping": _ES_AUTO},
    ),
    "stratified_group_kfold": _fit(
        "binary",
        {
            "method": "stratified_group_kfold",
            "n_splits": 3,
            "shuffle": False,
            "random_state": 7,
        },
    ),
    "stratified_group_kfold_seed_fallback": _fit(
        "binary",
        {"method": "stratified_group_kfold", "n_splits": 3},
        training={"seed": 17},
    ),
    "time_series": _fit(
        "regression",
        {
            "method": "time_series",
            "n_splits": 3,
            "gap": 2,
            "train_size_max": 100,
            "test_size_max": 30,
        },
        training={"early_stopping": _ES_AUTO},
    ),  # fmt: skip
    "purged_time_series": _fit(
        "regression",
        {
            "method": "purged_time_series",
            "n_splits": 3,
            "purge_gap": 2,
            "train_size_max": 100,
            "test_size_max": 30,
        },
    ),  # fmt: skip
    "group_time_series": _fit(
        "regression",
        {
            "method": "group_time_series",
            "n_splits": 3,
            "gap": 1,
            "train_size_max": 12,
            "test_size_max": 4,
        },
    ),  # fmt: skip
    "blocked_group_kfold": _fit(
        "binary",
        {
            "method": "blocked_group_kfold",
            "blocks": {
                "col": "blk",
                "cutoffs": [1, 2],
                "mode": "sliding",
                "train_window": 2,
            },
            "groups": {"col": "g", "n_splits": 2, "stratify": True, "shuffle": False},
            "min_train_rows": 3,
            "min_valid_rows": 2,
        },
        training={
            "seed": 7,
            "early_stopping": {"enabled": True, "validation_ratio": 0.2},
        },
    ),  # fmt: skip
    "purged_time_series_auto": _fit(
        "regression",
        {"method": "purged_time_series", "n_splits": 3, "purge_gap": 3},
        training={"early_stopping": _ES_AUTO},
    ),
    # Three groups: every training fold has fewer than the four groups
    # BlockedGroupInnerValid needs, so the classification fallback runs.
    "blocked_group_fallback": lambda: Model(
        _config(
            "binary",
            {
                "method": "blocked_group_kfold",
                "blocks": {"col": "blk", "cutoffs": [1, 2]},
                "groups": {"col": "g", "n_splits": 2},
            },
            training={"early_stopping": {"enabled": True, "validation_ratio": 0.3}},
        )  # fmt: skip
    ).fit(data=_frame("binary").assign(g=lambda d: d.index % 3)),
    # The regression twin of the fallback above: BlockedGroupInnerValid falls
    # back to TimeHoldoutInnerValid(ratio), with no gap.
    "blocked_group_regression_fallback": lambda: Model(
        _config(
            "regression",
            {
                "method": "blocked_group_kfold",
                "blocks": {"col": "blk", "cutoffs": [1, 2]},
                "groups": {"col": "g", "n_splits": 2},
            },
            training={"early_stopping": {"enabled": True, "validation_ratio": 0.3}},
        )  # fmt: skip
    ).fit(data=_frame("regression").assign(g=lambda d: d.index % 3)),
    "tune_calibrated": lambda: Model(
        _config(
            "binary",
            {"method": "stratified_kfold", "n_splits": 3},
            tuning={"optuna": {"params": {"n_trials": 1}}},
            calibration={"method": "platt"},
            evaluation={"metrics": ["auc"]},
        )
    ).tune(data=_frame("binary")),
    "inner_holdout": _fit(
        "binary",
        {"method": "kfold", "n_splits": 3},
        training={"seed": 7, "early_stopping": _ES_HOLDOUT},
    ),
    "inner_auto_holdout": _fit(
        "binary",
        {"method": "kfold", "n_splits": 3},
        training={"seed": 17, "early_stopping": _ES_AUTO},
    ),
    "inner_group_holdout": _fit(
        "regression",
        {"method": "group_kfold", "n_splits": 3},
        training={
            "early_stopping": {
                "enabled": True,
                "inner_valid": {
                    "method": "group_holdout",
                    "ratio": 0.2,
                    "random_state": 7,
                },
            }
        },
    ),  # fmt: skip
    "inner_time_holdout": _fit(
        "regression",
        {"method": "time_series", "n_splits": 3},
        training={
            "early_stopping": {
                "enabled": True,
                "inner_valid": {"method": "time_holdout", "ratio": 0.2},
            }
        },
    ),  # fmt: skip
    "metrics_binary": _fit(
        "binary",
        {"method": "kfold", "n_splits": 3},
        evaluation={"metrics": [{"ece": {"n_bins": 5}}, {"precision_at_k": {"k": 20}}]},
    ),
    "metrics_feval": _fit(
        "binary",
        {"method": "kfold", "n_splits": 3},
        model={
            "name": "lgbm",
            "params": {
                "n_estimators": 5,
                "metric": [{"ece": {"n_bins": 7}}, {"precision_at_k": {"k": 15}}],
            },
        },
    ),  # fmt: skip
    "metrics_regression": _fit(
        "regression",
        {"method": "kfold", "n_splits": 3},
        evaluation={"metrics": [{"huber": {"delta": 2.0}}]},
    ),
    "calibration_platt": _fit(
        "binary",
        {"method": "kfold", "n_splits": 3},
        calibration={"method": "platt", "params": {"tol": 1e-7}},
    ),
    "calibration_isotonic": _fit(
        "binary",
        {"method": "kfold", "n_splits": 3},
        calibration={"method": "isotonic", "params": {"num_leaves": 7}},
    ),
    "calibration_beta": _fit(
        "binary",
        {"method": "kfold", "n_splits": 3},
        calibration={"method": "beta", "params": {"tol": 1e-7}},
    ),
    "unseen_policy": _fit(
        "regression",
        {"method": "kfold", "n_splits": 3},
        features={"exclude": ["g", "t", "blk"], "unseen_policy": "nan"},
    ),
    "tune": lambda: Model(
        _config(
            "regression",
            {"method": "kfold", "n_splits": 3},
            training={"seed": 7},
            tuning={
                "optuna": {
                    "params": {"n_trials": 2, "direction": "maximize", "timeout": 600.0}
                }
            },
            evaluation={"metrics": ["r2"]},
        )  # fmt: skip
    ).tune(data=_frame("regression")),
}

#: knob -> [(scenario, the value that path must deliver), ...], one cell per path
#: whose source differs (explicit / automatic resolution, fallback, route). Most
#: cells set a non-default value; a few check a branch whose value is the
#: default (regression task, automatic stratify=False).
#: ``test_every_config_row_has_a_non_default_witness`` keeps at least one
#: non-default cell per row.
_CELLS: dict[str, list[tuple[str, Any]]] = {
    "KFoldSplitter.n_splits": [("kfold", 3)],
    "KFoldSplitter.shuffle": [("kfold", False)],
    "KFoldSplitter.random_state": [("kfold", 7), ("kfold_seed_fallback", 17)],
    "LGBMAdapter.params": [("kfold", {"learning_rate": 0.07})],
    "LGBMAdapter.task": [("kfold", "regression"), ("stratified_kfold", "binary")],
    "Evaluator.task": [("kfold", "regression"), ("stratified_kfold", "binary")],
    "CVTrainer.task": [("kfold", "regression"), ("stratified_kfold", "binary")],
    "RefitTrainer.task": [("kfold", "regression"), ("stratified_kfold", "binary")],
    "StratifiedKFoldSplitter.n_splits": [("stratified_kfold", 3)],
    "StratifiedKFoldSplitter.random_state": [
        ("stratified_kfold", 7),
        ("stratified_kfold_auto", 17),
    ],
    "GroupKFoldSplitter.n_splits": [("group_kfold", 3)],
    "StratifiedGroupKFoldSplitter.n_splits": [("stratified_group_kfold", 3)],
    "StratifiedGroupKFoldSplitter.shuffle": [("stratified_group_kfold", False)],
    "StratifiedGroupKFoldSplitter.random_state": [
        ("stratified_group_kfold", 7),
        ("stratified_group_kfold_seed_fallback", 17),
    ],
    "TimeSeriesSplitter.n_splits": [("time_series", 3)],
    "TimeSeriesSplitter.gap": [("time_series", 2)],
    "TimeSeriesSplitter.max_train_size": [("time_series", 100)],
    "TimeSeriesSplitter.max_test_size": [("time_series", 30)],
    "PurgedTimeSeriesSplitter.n_splits": [("purged_time_series", 3)],
    "PurgedTimeSeriesSplitter.purge_gap": [("purged_time_series", 2)],
    "PurgedTimeSeriesSplitter.max_train_size": [("purged_time_series", 100)],
    "PurgedTimeSeriesSplitter.max_test_size": [("purged_time_series", 30)],
    "GroupTimeSeriesSplitter.n_splits": [("group_time_series", 3)],
    "GroupTimeSeriesSplitter.gap": [("group_time_series", 1)],
    "GroupTimeSeriesSplitter.max_train_size": [("group_time_series", 12)],
    "GroupTimeSeriesSplitter.max_test_size": [("group_time_series", 4)],
    "BlockedGroupKFoldSplitter.mode": [("blocked_group_kfold", "sliding")],
    "BlockedGroupKFoldSplitter.train_window": [("blocked_group_kfold", 2)],
    "BlockedGroupKFoldSplitter.n_splits": [("blocked_group_kfold", 2)],
    "BlockedGroupKFoldSplitter.stratify": [("blocked_group_kfold", True)],
    "BlockedGroupKFoldSplitter.shuffle": [("blocked_group_kfold", False)],
    "BlockedGroupKFoldSplitter.random_state": [("blocked_group_kfold", 7)],
    "BlockedGroupKFoldSplitter.min_train_rows": [("blocked_group_kfold", 3)],
    "BlockedGroupKFoldSplitter.min_valid_rows": [("blocked_group_kfold", 2)],
    "BlockedGroupInnerValid.ratio": [("blocked_group_kfold", 0.2)],
    "BlockedGroupInnerValid.task": [("blocked_group_kfold", "binary")],
    "HoldoutInnerValid.ratio": [("inner_holdout", 0.2), ("inner_auto_holdout", 0.3)],
    "HoldoutInnerValid.random_state": [
        ("inner_holdout", 7),
        ("inner_auto_holdout", 17),
    ],
    "HoldoutInnerValid.stratify": [
        ("inner_holdout", True),
        ("inner_auto_holdout", False),
        ("stratified_kfold_auto", True),
    ],
    "LGBMAdapter.early_stopping_rounds": [("inner_holdout", 7)],
    "LGBMAdapter.random_state": [("inner_holdout", 7)],
    "GroupHoldoutInnerValid.ratio": [
        ("inner_group_holdout", 0.2),
        ("group_kfold_auto", 0.3),
    ],
    "GroupHoldoutInnerValid.random_state": [
        ("inner_group_holdout", 7),
        ("group_kfold_auto", 17),
    ],
    "TimeHoldoutInnerValid.ratio": [
        ("inner_time_holdout", 0.2),
        ("time_series", 0.3),
        ("blocked_group_regression_fallback", 0.3),
    ],
    "TimeHoldoutInnerValid.gap": [("time_series", 2), ("purged_time_series_auto", 3)],
    "StratifiedTimeHoldoutInnerValid.ratio": [("blocked_group_fallback", 0.3)],
    "ECE.n_bins": [("metrics_binary", 5), ("metrics_feval", 7)],
    "PrecisionAtK.k": [("metrics_binary", 20), ("metrics_feval", 15)],
    "HuberLoss.delta": [("metrics_regression", 2.0)],
    "PlattCalibrator.params": [("calibration_platt", {"tol": 1e-7})],
    "IsotonicCalibrator.params": [("calibration_isotonic", {"num_leaves": 7})],
    "BetaCalibrator.params": [("calibration_beta", {"tol": 1e-7})],
    "CategoricalEncoder.unseen_policy": [("unseen_policy", "nan")],
    "NativeFeaturePipeline.unseen_policy": [("unseen_policy", "nan")],
    "Tuner.n_trials": [("tune", 2)],
    "Tuner.direction": [("tune", "maximize")],
    "Tuner.timeout": [("tune", 600.0)],
    "Tuner.seed": [("tune", 7)],
}

#: Config rows whose value is not a constructor argument the spy can read:
#: ``Model.output_dir`` resolves the Config key inside ``Model.__init__``.
_CHECKED_SEPARATELY = {"Model.output_dir"}

#: Arguments the library merges the configured mapping into (the adapter adds
#: its resolved parameters; the isotonic calibrator adds the seed), so the
#: configured items must be contained rather than equal.
_MERGED = {"LGBMAdapter.params", "IsotonicCalibrator.params"}

_CELL_IDS = [(knob, i) for knob in sorted(_CELLS) for i in range(len(_CELLS[knob]))]


@pytest.fixture(scope="module")
def constructed() -> dict[str, dict[str, list[dict[str, Any]]]]:
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for name, call in _SCENARIOS.items():
        with _spying() as received:
            call()
        out[name] = {cls: list(calls) for cls, calls in received.items()}
    return out


def test_registry_classifies_exactly_the_census() -> None:
    census = knob_census()
    assert len(census) >= 60, (
        "the sweep found too few knobs: a broken census, not a finding"
    )
    assert set(REGISTRY) == census
    assert {kind for kind, _ in REGISTRY.values()} <= set(KINDS)


def test_every_config_row_has_an_executed_cell() -> None:
    config_rows = {k for k, (kind, _) in REGISTRY.items() if kind == "config"}
    assert set(_CELLS) | _CHECKED_SEPARATELY == config_rows
    assert not set(_CELLS) & _CHECKED_SEPARATELY


def test_every_config_row_has_a_non_default_witness() -> None:
    """A cell whose value equals the constructor default proves nothing about
    reachability; every config row needs at least one that differs."""
    classes = {cls.__name__: cls for cls in _SPIED}
    for knob, cells in _CELLS.items():
        cls_name, param = knob.split(".")
        default = (
            inspect.signature(classes[cls_name].__init__).parameters[param].default
        )
        assert any(value != default for _, value in cells), knob


@pytest.mark.parametrize(
    ("knob", "index"), _CELL_IDS, ids=[f"{k}-{i}" for k, i in _CELL_IDS]
)
def test_config_value_reaches_the_constructor(
    knob: str, index: int, constructed: dict[str, dict[str, list[dict[str, Any]]]]
) -> None:
    cls, param = knob.split(".")
    scenario, want = _CELLS[knob][index]
    calls = constructed[scenario].get(cls, [])
    assert calls, f"{cls} was not constructed in scenario {scenario!r}"
    got = [call[param] for call in calls]
    if knob in _MERGED:
        assert all(want.items() <= value.items() for value in got), got
    else:
        assert all(value == want for value in got), got


def test_output_dir_comes_from_config_unless_given(tmp_path: Path) -> None:
    from_config = tmp_path / "from_config"
    cfg = _config(
        "regression", {"method": "kfold", "n_splits": 3}, output_dir=str(from_config)
    )
    assert str(Model(cfg)._output_dir) == str(from_config)
    assert Model(cfg, output_dir=tmp_path)._output_dir == tmp_path


def test_api_rows_reach_the_constructor(tmp_path: Path) -> None:
    df = _frame("regression")
    cfg = _config(
        "regression",
        {"method": "kfold", "n_splits": 3},
        tuning={"optuna": {"params": {"n_trials": 1}}},
        evaluation={"metrics": ["rmse"]},
    )
    model = Model(cfg, data=df)
    assert model._data is df
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    with _spying() as received:
        model.tune(progress_callback=_callback, storage=storage, study_name="knobs")
    (call,) = received["Tuner"]
    assert call["progress_callback"] is _callback
    assert call["storage"] == storage
    assert call["study_name"] == "knobs"


def test_deprecated_splitter_embargo_reaches_the_splitter() -> None:
    """The api row: only direct construction sets it, and it is added (H-0115)."""
    from lizyml.splitters import PurgedTimeSeriesSplitter

    with pytest.warns(DeprecationWarning, match="purge_gap"):
        splitter = PurgedTimeSeriesSplitter(n_splits=3, purge_gap=2, embargo=1)
    assert splitter.purge_gap == 3


def _blueprint_rows() -> dict[str, str]:
    """The §5.5 table, read only from inside top-level section 5."""
    text = (_ROOT / "BLUEPRINT.md").read_text(encoding="utf-8")
    start = text.index("\n## 5.5 ")
    parent = text.rindex("\n# ", 0, start)
    assert text.startswith("\n# 5. ", parent), "§5.5 must sit under section 5"
    end = text.index("\n# ", start)
    assert text.startswith("\n# 6.", end), "§5.5 must be the last part of section 5"
    rows = re.findall(
        r"^\| `([A-Za-z]+\.[a-z_]+)` \| ([a-z]+) \|", text[start:end], re.M
    )
    assert rows, "the §5.5 table is empty"
    return dict(rows)


def test_rows_outside_config_are_stated_in_blueprint() -> None:
    stated = _blueprint_rows()
    expected = {k: kind for k, (kind, _) in REGISTRY.items() if kind != "config"}
    assert stated == expected


def test_blueprint_counts_match_the_registry() -> None:
    """§5.5's prose counts follow the registry (H-0115 moved one row to api)."""
    text = (_ROOT / "BLUEPRINT.md").read_text(encoding="utf-8")
    start = text.index("\n## 5.5 ")
    section = text[start : text.index("\n# ", start)]
    total = len(REGISTRY)
    config = sum(1 for kind, _ in REGISTRY.values() if kind == "config")
    assert f"（{total} 個、AST で数えた母集団）" in section
    assert f"**Config のキーの値がそのまま渡る {config} 個**" in section
    assert f"残りの {total - config} 個" in section


def test_derived_class_counts_are_set_only_for_multiclass() -> None:
    for task, want in (("regression", None), ("binary", None), ("multiclass", 3)):
        df = _frame("regression" if task == "regression" else "binary")
        if task == "multiclass":
            df["y"] = np.digitize(df["a"], [-0.5, 0.5])
        with _spying() as received:
            Model(_config(task, {"method": "kfold", "n_splits": 3})).fit(data=df)
        assert {call["num_class"] for call in received["LGBMAdapter"]} == {want}
        assert {call["n_classes"] for call in received["CVTrainer"]} == {want}


def test_derived_collect_raw_scores_follows_calibration_on_fit_only(
    constructed: dict[str, dict[str, list[dict[str, Any]]]],
) -> None:
    assert {
        c["collect_raw_scores"] for c in constructed["calibration_platt"]["CVTrainer"]
    } == {True}
    assert {c["collect_raw_scores"] for c in constructed["kfold"]["CVTrainer"]} == {
        False
    }
    assert {c["collect_raw_scores"] for c in constructed["tune"]["CVTrainer"]} == {
        False
    }
    # Calibration configured: fit collects raw scores, tune trials still do not.
    calibrated = constructed["tune_calibrated"]["CVTrainer"]
    assert calibrated
    assert {c["collect_raw_scores"] for c in calibrated} == {False}


def test_explicit_time_holdout_gets_no_gap(
    constructed: dict[str, dict[str, list[dict[str, Any]]]],
) -> None:
    """The automatic paths pass the outer gap (config cells above); an explicit
    time_holdout passes none, so the constructor default 0 applies (H-0101)."""
    explicit = constructed["inner_time_holdout"]["TimeHoldoutInnerValid"]
    assert {c["gap"] for c in explicit} == {0}
    fallback = constructed["blocked_group_regression_fallback"]["TimeHoldoutInnerValid"]
    assert fallback
    assert {c["gap"] for c in fallback} == {0}


def test_policy_rows_hold_their_fixed_value(
    constructed: dict[str, dict[str, list[dict[str, Any]]]],
) -> None:
    """No Config key or argument moves the two library-fixed values."""
    for scenario in ("stratified_kfold", "stratified_kfold_auto"):
        calls = constructed[scenario]["StratifiedKFoldSplitter"]
        assert {c["shuffle"] for c in calls} == {True}
    for scenario in ("kfold", "inner_holdout", "tune"):
        assert {c["verbose_eval"] for c in constructed[scenario]["LGBMAdapter"]} == {-1}
