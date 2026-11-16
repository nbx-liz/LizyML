"""Probe: which column names survive fit -> export -> load -> predict (#406).

Each case fits a 5-tree binary model on a frame with one probed column beside
``x``, exports it, loads it back and compares predictions. One line per case:
``<case>\t<stage reached>\t<outcome>``. Run through heavy.sh.
"""

from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from lizyml import Model
from lizyml.core.exceptions import LizyMLError

CASES: list[tuple[str, object]] = [
    ("plain", "z"),
    ("empty", ""),
    ("space_only", " "),
    ("inner_space", "a b"),
    ("tab", "a\tb"),
    ("newline", "a\nb"),
    ("leading_space", " a"),
    ("unicode", "列"),
    ("equals", "a=b"),
    ("json_special", 'a"b'),
    ("int_name", 0),
]


def frame(name: object, n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=n)})
    df[name] = rng.normal(size=n)
    df["target"] = (df["x"] > 0).astype(int)
    return df


CONFIG = {
    "config_version": 1,
    "task": "binary",
    "data": {"target": "target"},
    "split": {"method": "kfold", "n_splits": 3},
    "model": {"name": "lgbm", "params": {"n_estimators": 5}},
}


def run(
    label: str,
    name: object,
    collide: bool = False,
    role: str = "feature",
) -> str:
    df = frame(name)
    config = {**CONFIG, "data": dict(CONFIG["data"])}  # type: ignore[call-overload]
    if collide:
        df.insert(1, "a_b", df[name] * 2.0)
    if role == "target":
        df = df.drop(columns=["target"]).rename(columns={name: "y"})
        df[name] = (df["x"] > 0).astype(int)
        df = df.drop(columns=["y"])
        config["data"] = {"target": name}
    if role == "excluded":
        config["features"] = {"exclude": [name]}
    target = config["data"]["target"]
    stage = "fit"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Model(config)
            m.fit(data=df)
            X = df.drop(columns=[target])
            before = np.asarray(m.predict(X).pred)
            with tempfile.TemporaryDirectory() as d:
                stage = "export"
                m.export(Path(d) / "m")
                stage = "load"
                loaded = Model.load(Path(d) / "m")
                stage = "predict"
                after = np.asarray(loaded.predict(X).pred)
        same = np.allclose(before, after)
        return f"{label}\tdone\t{'same predictions' if same else 'PREDICTIONS DIFFER'}"
    except LizyMLError as exc:
        return f"{label}\t{stage}\t{exc.code.name} {sorted(exc.context)}"
    except Exception as exc:  # noqa: BLE001 - the probe reports every failure
        return f"{label}\t{stage}\t{type(exc).__name__}: {str(exc)[:80]!r}"


def main() -> int:
    for label, name in CASES:
        print(run(label, name), flush=True)
    print(run("collide_a_b", "a b", collide=True), flush=True)
    print(run("empty_target", "", role="target"), flush=True)
    print(run("empty_excluded", "", role="excluded"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
