"""Probe: the empty column name across positions, types and save surfaces (#406).

For each case: fit, then Model.export -> Model.load, then export_code, each
reported as ``ok`` or the error. The last line checks the mechanism outside
lizyml: a LightGBM booster trained with an empty feature name, written with
``model_to_string`` and read back with ``Booster(model_str=...)``.
"""

from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from lizyml import Model
from lizyml.core.exceptions import LizyMLError

N = 200
CONFIG = {
    "config_version": 1,
    "task": "binary",
    "data": {"target": "target"},
    "split": {"method": "kfold", "n_splits": 3},
    "model": {"name": "lgbm", "params": {"n_estimators": 5}},
}


def outcome(call: object) -> str:
    try:
        call()  # type: ignore[operator]
    except LizyMLError as exc:
        return f"{exc.code.name} {sorted(exc.context)}"
    except Exception as exc:  # noqa: BLE001 - the probe reports every failure
        return f"{type(exc).__name__}: {str(exc)[:60]!r}"
    return "ok"


def case(label: str, df: pd.DataFrame, config: dict[str, object]) -> str:
    with warnings.catch_warnings(), tempfile.TemporaryDirectory() as d:
        warnings.simplefilter("ignore")
        m = Model(config)
        fit = outcome(lambda: m.fit(data=df))
        if fit != "ok":
            return f"{label}\tfit={fit}"
        export = outcome(lambda: m.export(Path(d) / "m"))
        load = outcome(lambda: Model.load(Path(d) / "m")) if export == "ok" else "-"
        code = outcome(lambda: m.export_code(Path(d) / "code"))
        return f"{label}\texport={export}\tload={load}\texport_code={code}"


def base(names: list[object]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({name: rng.normal(size=N) for name in names})
    df["target"] = (rng.normal(size=N) > 0).astype(int)
    return df


def mechanism() -> str:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=N), "": rng.normal(size=N)})
    y = (X["x"] > 0).astype(int)
    params = {"objective": "binary", "verbose": -1}
    # lgb.train itself round-trips the model text unless the booster is kept.
    plain = outcome(lambda: lgb.train(params, lgb.Dataset(X, y), num_boost_round=3))
    booster = lgb.train(
        params, lgb.Dataset(X, y), num_boost_round=3, keep_training_booster=True
    )
    text = booster.model_to_string()
    line = next(row for row in text.splitlines() if row.startswith("feature_names="))
    read = outcome(lambda: lgb.Booster(model_str=text))
    return (
        f"lightgbm\ttrain={plain}\tfeature_name()={booster.feature_name()!r}"
        f"\t{line!r}\tread={read}"
    )


def main() -> int:
    print(case("empty_last", base(["x", ""]), CONFIG), flush=True)
    print(case("empty_middle", base(["x", "", "w"]), CONFIG), flush=True)
    print(case("np_str_empty", base(["x", np.str_("")]), CONFIG), flush=True)
    cat = base(["x"])
    cat[""] = pd.Series(["a", "b"] * (N // 2), dtype="category")
    print(case("empty_categorical", cat, CONFIG), flush=True)
    print(case("plain_control", base(["x", "z"]), CONFIG), flush=True)
    print(mechanism(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
