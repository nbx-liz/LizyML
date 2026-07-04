"""Codegen split reproduction (#228, H-0090).

The generated ``train.py`` must rebuild the model's outer CV folds from
``config.json["split"]`` so the calibration OOF respects the same time/group
boundary — a shuffled K-fold would leak across it on retrain. These tests load
the *actually generated* ``train.py`` and assert its ``_resolve_folds`` output
is fold-index-identical to LizyML's own splitter (``build_splitter``) for every
supported ``split.method``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from lizyml.config.schema import LizyMLConfig
from lizyml.core._model_factories import build_splitter
from lizyml.core.model import Model


def _load_generated_train(export_dir: Path) -> ModuleType:
    """Import the generated ``train.py`` as a module (reads its config.json)."""
    spec = importlib.util.spec_from_file_location(
        f"gen_train_{export_dir.name}", export_dir / "train.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _assert_folds_equal(
    generated: list[tuple[np.ndarray, np.ndarray]],
    ground_truth: list[tuple[np.ndarray, np.ndarray]],
    method: str,
) -> None:
    assert len(generated) == len(ground_truth), (
        f"{method}: fold count {len(generated)} != {len(ground_truth)}"
    )
    for i, ((gtr, gva), (etr, eva)) in enumerate(
        zip(generated, ground_truth, strict=True)
    ):
        np.testing.assert_array_equal(
            np.sort(np.asarray(gtr)),
            np.sort(np.asarray(etr)),
            err_msg=f"{method}: fold {i} train indices differ",
        )
        np.testing.assert_array_equal(
            np.sort(np.asarray(gva)),
            np.sort(np.asarray(eva)),
            err_msg=f"{method}: fold {i} valid indices differ",
        )


def _ground_truth_folds(
    cfg: LizyMLConfig,
    df: pd.DataFrame,
    y: np.ndarray,
    *,
    time_col: str | None,
    group_col: str | None,
    blocks_col: str | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """LizyML's own outer folds, mapped back to ``df`` row order.

    Reproduces the facade's sort-then-split: rows are ordered by ``time_col``
    (time-series family) or ``blocks_col`` (blocked), the splitter runs on the
    sorted arrays, and indices are mapped back through the sort order.
    """
    n = len(df)
    if blocks_col is not None:
        order = np.asarray(df[blocks_col].argsort())
        bv = df[blocks_col].to_numpy()[order]
        groups = df[group_col].to_numpy()[order] if group_col else None
        splitter = build_splitter(
            cfg, block_values=bv, task=cfg.task, seed=cfg.training.seed
        )
    elif time_col is not None:
        order = np.asarray(df[time_col].argsort())
        groups = df[group_col].to_numpy()[order] if group_col else None
        splitter = build_splitter(cfg, task=cfg.task, seed=cfg.training.seed)
    else:
        order = np.arange(n)
        groups = df[group_col].to_numpy() if group_col else None
        splitter = build_splitter(cfg, task=cfg.task, seed=cfg.training.seed)

    y_sorted = y[order]
    folds = list(splitter.split(n, y_sorted, groups))
    return [(order[np.asarray(tr)], order[np.asarray(va)]) for tr, va in folds]


def _make_df(
    n: int,
    *,
    time: bool = False,
    group: int | None = None,
    period: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "f0": rng.normal(size=n),
            "f1": rng.normal(size=n),
            "target": rng.integers(0, 2, n).astype(np.int64),
        }
    )
    if time:
        # Deliberately unsorted time to exercise the sort step.
        df["t"] = rng.permutation(n)
    if group is not None:
        df["g"] = rng.integers(0, group, n)
    if period is not None:
        df["p"] = rng.integers(0, period, n)
    return df


def _export_and_resolve(
    tmp_path: Path,
    cfg_dict: dict[str, Any],
    df: pd.DataFrame,
    name: str,
) -> tuple[ModuleType, LizyMLConfig]:
    cfg = LizyMLConfig(**cfg_dict)
    m = Model(cfg)
    m.fit(data=df)
    export_dir = tmp_path / name
    m.export_code(export_dir)
    return _load_generated_train(export_dir), cfg


# ---------------------------------------------------------------------------
# One test per supported split.method
# ---------------------------------------------------------------------------


class TestSplitReproduction:
    def test_time_series(self, tmp_path: Path) -> None:
        df = _make_df(90, time=True, seed=1)
        cfg_dict = {
            "config_version": 1,
            "task": "binary",
            "data": {"target": "target", "time_col": "t"},
            "split": {"method": "time_series", "n_splits": 3, "gap": 2},
            "model": {"name": "lgbm", "params": {"n_estimators": 5}},
            "training": {"seed": 7},
        }
        mod, cfg = _export_and_resolve(tmp_path, cfg_dict, df, "ts")
        y = df["target"].to_numpy()
        gen = mod._resolve_folds(df, y)
        gt = _ground_truth_folds(
            cfg, df, y, time_col="t", group_col=None, blocks_col=None
        )
        _assert_folds_equal(gen, gt, "time_series")

    def test_purged_time_series(self, tmp_path: Path) -> None:
        df = _make_df(100, time=True, seed=2)
        cfg_dict = {
            "config_version": 1,
            "task": "binary",
            "data": {"target": "target", "time_col": "t"},
            "split": {
                "method": "purged_time_series",
                "n_splits": 3,
                "purge_gap": 3,
                "embargo": 2,
            },
            "model": {"name": "lgbm", "params": {"n_estimators": 5}},
            "training": {"seed": 3},
        }
        mod, cfg = _export_and_resolve(tmp_path, cfg_dict, df, "purged")
        y = df["target"].to_numpy()
        gen = mod._resolve_folds(df, y)
        gt = _ground_truth_folds(
            cfg, df, y, time_col="t", group_col=None, blocks_col=None
        )
        _assert_folds_equal(gen, gt, "purged_time_series")

    def test_group_time_series(self, tmp_path: Path) -> None:
        df = _make_df(120, time=True, group=8, seed=4)
        cfg_dict = {
            "config_version": 1,
            "task": "binary",
            "data": {"target": "target", "time_col": "t", "group_col": "g"},
            "split": {"method": "group_time_series", "n_splits": 3, "gap": 1},
            "model": {"name": "lgbm", "params": {"n_estimators": 5}},
            "training": {"seed": 5},
        }
        mod, cfg = _export_and_resolve(tmp_path, cfg_dict, df, "gts")
        y = df["target"].to_numpy()
        gen = mod._resolve_folds(df, y)
        gt = _ground_truth_folds(
            cfg, df, y, time_col="t", group_col="g", blocks_col=None
        )
        _assert_folds_equal(gen, gt, "group_time_series")

    def test_group_kfold(self, tmp_path: Path) -> None:
        df = _make_df(120, group=10, seed=6)
        cfg_dict = {
            "config_version": 1,
            "task": "binary",
            "data": {"target": "target", "group_col": "g"},
            "split": {"method": "group_kfold", "n_splits": 3},
            "model": {"name": "lgbm", "params": {"n_estimators": 5}},
            "training": {"seed": 8},
        }
        mod, cfg = _export_and_resolve(tmp_path, cfg_dict, df, "gkf")
        y = df["target"].to_numpy()
        gen = mod._resolve_folds(df, y)
        gt = _ground_truth_folds(
            cfg, df, y, time_col=None, group_col="g", blocks_col=None
        )
        _assert_folds_equal(gen, gt, "group_kfold")

    def test_stratified_group_kfold(self, tmp_path: Path) -> None:
        df = _make_df(120, group=10, seed=9)
        cfg_dict = {
            "config_version": 1,
            "task": "binary",
            "data": {"target": "target", "group_col": "g"},
            "split": {
                "method": "stratified_group_kfold",
                "n_splits": 3,
                "shuffle": True,
                "random_state": 11,
            },
            "model": {"name": "lgbm", "params": {"n_estimators": 5}},
            "training": {"seed": 8},
        }
        mod, cfg = _export_and_resolve(tmp_path, cfg_dict, df, "sgkf")
        y = df["target"].to_numpy()
        gen = mod._resolve_folds(df, y)
        gt = _ground_truth_folds(
            cfg, df, y, time_col=None, group_col="g", blocks_col=None
        )
        _assert_folds_equal(gen, gt, "stratified_group_kfold")

    def test_blocked_group_kfold(self, tmp_path: Path) -> None:
        df = _make_df(200, group=12, period=6, seed=10)
        cfg_dict = {
            "config_version": 1,
            "task": "binary",
            "data": {"target": "target"},
            "split": {
                "method": "blocked_group_kfold",
                "blocks": {"col": "p", "cutoffs": [2, 4], "mode": "expanding"},
                "groups": {
                    "col": "g",
                    "n_splits": 2,
                    "stratify": True,
                    "shuffle": True,
                },
                "min_train_rows": 1,
                "min_valid_rows": 1,
            },
            "model": {"name": "lgbm", "params": {"n_estimators": 5}},
            "training": {"seed": 13},
        }
        mod, cfg = _export_and_resolve(tmp_path, cfg_dict, df, "blocked")
        y = df["target"].to_numpy()
        gen = mod._resolve_folds(df, y)
        gt = _ground_truth_folds(
            cfg, df, y, time_col=None, group_col="g", blocks_col="p"
        )
        _assert_folds_equal(gen, gt, "blocked_group_kfold")

    def test_retrain_calibration_runs_end_to_end(self, tmp_path: Path) -> None:
        """A binary + calibration time-series export retrains via the generated
        train() using the reproduced split, producing a calibrator.json."""
        df = _make_df(120, time=True, seed=15)
        cfg_dict = {
            "config_version": 1,
            "task": "binary",
            "data": {"target": "target", "time_col": "t"},
            "split": {"method": "time_series", "n_splits": 3},
            "model": {"name": "lgbm", "params": {"n_estimators": 5}},
            "training": {"seed": 1},
            "calibration": {"method": "platt"},
        }
        mod, _ = _export_and_resolve(tmp_path, cfg_dict, df, "retrain")
        export_dir = tmp_path / "retrain"
        mod.train(df, calibrate=True)
        assert (export_dir / "artifacts" / "calibrator.json").exists()

    def test_kfold_fallback_unchanged(self, tmp_path: Path) -> None:
        """Shuffle-safe kfold still reproduces (StratifiedKFold for binary)."""
        df = _make_df(90, seed=14)
        cfg_dict = {
            "config_version": 1,
            "task": "binary",
            "data": {"target": "target"},
            "split": {"method": "stratified_kfold", "n_splits": 3, "random_state": 42},
            "model": {"name": "lgbm", "params": {"n_estimators": 5}},
            "training": {"seed": 0},
        }
        mod, cfg = _export_and_resolve(tmp_path, cfg_dict, df, "skf")
        y = df["target"].to_numpy()
        gen = mod._resolve_folds(df, y)
        gt = _ground_truth_folds(
            cfg, df, y, time_col=None, group_col=None, blocks_col=None
        )
        _assert_folds_equal(gen, gt, "stratified_kfold")
