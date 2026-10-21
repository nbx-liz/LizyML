"""H-0139 probe: how today's develop treats category columns of temporal values.

For each case this probe fits a binary model (800 rows, the category decides
the target, as in ``docs/audits/2026-10-h0136/probes/inmemory_fold.py``) and
prints, per stage, what happened:

- ``fit``: the category warnings fit emitted (counted by first words);
- ``state``: the categories the refit encoder state holds (str, type, fold);
- ``predict``: unseen rows in-memory predict reports on the training rows;
- ``batch``: whether predicting each distinct value alone gives the same
  probability as predicting it inside the full training batch;
- ``export`` / ``load``: ``Model.export`` then ``Model.load`` and predict
  (max absolute difference against in-memory);
- ``export_code``: ``Model.export_code`` and the generated ``transform``.

Every stage's exception is printed as the outcome (the measurement), never
skipped. Nothing here decides a contract: the cases are the open questions of
the temporal identity contract (pytz/zoneinfo/fixed offset at one instant,
naive/aware, date/datetime/Timestamp, datetime64 units, timedelta forms).

Usage: python temporal_matrix.py   (pandas 2 with pytz runs every case)

``temporal_matrix_pd2.out`` / ``_pd3.out`` are its output on develop
``3f8b65f`` (pandas 2.3.3 with pytz; pandas 3.0.1 without pytz), with the
structlog ``event=`` lines and the heavy.sh banner lines removed.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import tempfile
import warnings
import zoneinfo
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import lizyml
from lizyml import Model

NY = zoneinfo.ZoneInfo("America/New_York")
UTC = dt.timezone.utc
WALL = dt.datetime(2024, 11, 3, 1, 30)
T0 = pd.Timestamp("2024-01-02 03:04:05")


def _cases() -> dict[str, list[Any]]:
    cases: dict[str, list[Any]] = {
        "zoneinfo fold 0 + fold 1": [
            pd.Timestamp(WALL.replace(fold=f, tzinfo=NY)) for f in (0, 1)
        ],
        "aware UTC vs NY, one instant": [
            pd.Timestamp("2024-01-02 08:00", tz="UTC"),
            pd.Timestamp("2024-01-02 03:00", tz="America/New_York"),
            "x",
        ],
        "fixed offset vs zoneinfo, one instant": [
            pd.Timestamp(dt.datetime(2024, 1, 2, 3, 0, tzinfo=dt.timezone(dt.timedelta(hours=-5)))),
            pd.Timestamp(dt.datetime(2024, 1, 2, 3, 0, tzinfo=NY)),
            "x",
        ],
        "naive vs aware, same wall time": [T0, T0.tz_localize("UTC"), "x"],
        "date + Timestamp midnight": [dt.date(2024, 1, 2), pd.Timestamp("2024-01-02"), "x"],
        "datetime + Timestamp, same wall": [T0.to_pydatetime(), T0, "x"],
        "Timestamp unit s vs ns, same value": [T0.as_unit("s"), T0.as_unit("ns"), "x"],
        "Timestamp only (control)": [T0, T0 + pd.Timedelta(days=1), T0 + pd.Timedelta(days=2)],
        "datetime64 D vs ns, same instant": [
            np.datetime64("2024-01-02", "D"),
            np.datetime64("2024-01-02T00:00:00", "ns"),
            "x",
        ],
        "datetime64 only, distinct days": [
            np.datetime64("2024-01-02", "D"),
            np.datetime64("2024-01-03", "D"),
            np.datetime64("2024-01-04", "D"),
        ],
        "timedelta64 mixed units": [
            np.timedelta64(1, "D"),
            np.timedelta64(36, "h"),
            np.timedelta64(90, "m"),
        ],
        "timedelta64 h vs m, same duration": [
            np.timedelta64(1, "h"),
            np.timedelta64(60, "m"),
            "x",
        ],
        "pd.Timedelta vs dt.timedelta, same duration": [
            pd.Timedelta(hours=1),
            dt.timedelta(hours=1),
            "x",
        ],
        "datetime64 + Timestamp, same instant": [
            np.datetime64("2024-01-02T03:04:05", "ns"),
            T0,
            "x",
        ],
        "tuple holding a Timestamp + str": [(T0, 1), (T0 + pd.Timedelta(days=1), 1), "x"],
    }
    try:
        import pytz  # type: ignore[import-untyped, unused-ignore]
    except ImportError:
        return cases
    ny = pytz.timezone("America/New_York")
    std = pd.Timestamp(ny.localize(WALL, is_dst=False))
    dst = pd.Timestamp(ny.localize(WALL, is_dst=True))
    cases["pytz DST + standard (#379)"] = [dst, std]
    cases["pytz standard + str (control)"] = [std, "x"]
    cases["pytz vs zoneinfo, one instant"] = [
        pd.Timestamp(ny.localize(dt.datetime(2024, 1, 2, 3, 0))),
        pd.Timestamp(dt.datetime(2024, 1, 2, 3, 0, tzinfo=NY)),
        "x",
    ]
    cases["pytz standard + zoneinfo fold 1 (same instant)"] = [
        std,
        pd.Timestamp(WALL.replace(fold=1, tzinfo=NY)),
        "y",
    ]
    return cases


def _frame(values: list[Any], n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), n)
    idx[: len(values)] = np.arange(len(values))
    target = (idx % 2 == 0) ^ (rng.random(n) < 0.1)
    column = pd.Series([values[i] for i in idx], dtype=object)
    return pd.DataFrame({"c": column, "x": rng.normal(size=n), "target": target.astype(int)})


def _stage(name: str, fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 -- the outcome is the measurement
        print(f"    {name}: {type(exc).__name__}: {str(exc).splitlines()[0][:140]}")
        return None


def _fit(df: pd.DataFrame) -> tuple[Model, list[str]]:
    cfg = {
        "config_version": 1,
        "task": "binary",
        "data": {"target": "target"},
        "features": {"categorical": ["c"]},
        "split": {"method": "kfold", "n_splits": 2, "random_state": 42},
        "model": {"name": "lgbm", "params": {"n_estimators": 8, "verbose": -1}},
        "training": {"seed": 0},
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = Model(cfg)
        model.fit(data=df)
    names = [" ".join(str(w.message).split()[:6]) for w in caught if "ateg" in str(w.message)]
    return model, names


def _batch_check(model: Model, X: pd.DataFrame, values: list[Any]) -> str:
    full = np.asarray(model.predict(X).proba)
    diffs = []
    for i in range(len(values)):
        row = X.iloc[[i]].reset_index(drop=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alone = np.asarray(model.predict(row).proba)
        diffs.append(float(abs(alone[0] - full[i])))
    return "same" if max(diffs) == 0.0 else f"DIFFERS {diffs}"


def _export(model: Model, X: pd.DataFrame) -> str:
    before = np.asarray(model.predict(X).proba)
    path = Path(tempfile.mkdtemp()) / "a"
    model.export(path)
    res = Model.load(path).predict(X)
    diff = float(np.max(np.abs(before - np.asarray(res.proba))))
    unseen = [w[:60] for w in res.warnings if "unseen" in w]
    return f"ok, max_abs_diff {diff}, unseen {unseen or 'none'}"


def _export_code(model: Model, X: pd.DataFrame) -> str:
    out = Path(tempfile.mkdtemp()) / "code"
    model.export_code(out)
    spec = importlib.util.spec_from_file_location("generated_predict", out / "predict.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.transform(X)
    return "ok (generated transform ran)"


def main() -> None:
    print(f"lizyml={lizyml.__version__} pandas={pd.__version__} numpy={np.__version__}")
    for name, values in _cases().items():
        print(f"{name}:")
        df = _frame(values)
        X = df.drop(columns=["target"])
        fitted = _stage("fit", lambda df=df: _fit(df))
        if fitted is None:
            continue
        model, fit_warnings = fitted
        print(f"    fit warnings: {dict(Counter(fit_warnings)) or 'none'}")
        state = model._refit_result.pipeline_state["encoder"]  # type: ignore[union-attr]
        held = [
            (str(v), type(v).__name__, getattr(v, "fold", None))
            for v in state["categories"]["c"]
        ]
        print(f"    state ({len(held)} categories for {len(values)} inputs): {held}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _stage("predict", lambda: model.predict(X))
        if res is not None:
            unseen = [w[:90] for w in res.warnings if "unseen" in w]
            print(f"    predict on training rows: {unseen or 'no unseen rows'}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = _stage("batch", lambda: _batch_check(model, X, values))
            if out is not None:
                print(f"    batch (each value alone vs in the full batch): {out}")
            out = _stage("export", lambda: _export(model, X))
            if out is not None:
                print(f"    export/load: {out}")
            out = _stage("export_code", lambda: _export_code(model, X))
            if out is not None:
                print(f"    export_code: {out}")


if __name__ == "__main__":
    main()
