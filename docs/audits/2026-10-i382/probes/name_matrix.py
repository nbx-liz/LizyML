"""#382 / H-0141: what each column-name type does today, at this head.

25 label types x three column kinds (numeric, auto categorical, declared
categorical), each beside a string numeric column ``x`` and a string target.
120 rows, two folds, three boosting rounds. For each case, the outcome of:

- ``fit``: fit and in-memory predict;
- ``export_code``: writing the project into a real temporary directory;
- ``typed``: the generated ``predict.predict`` on the typed in-memory frame,
  compared with the in-memory prediction;
- ``csv``: the same frame written as CSV, read back with ``pd.read_csv`` and
  given to the generated ``predict.predict``.

Each stage's exception type is the outcome; nothing is skipped silently.

    PYTHONPATH=. python -B docs/audits/2026-10-i382/probes/name_matrix.py
"""

from __future__ import annotations

import importlib.util
import tempfile
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import lizyml
from lizyml import Model
from tests._helpers import make_config

warnings.simplefilter("ignore")

LABELS: dict[str, Any] = {
    "str": "c",
    "numpy.str_": np.str_("c"),
    "int": 7,
    "float": 7.5,
    "bool": True,
    "None": None,
    "numpy.float64": np.float64(7.5),
    "numpy.int8": np.int8(7),
    "numpy.int16": np.int16(7),
    "numpy.int32": np.int32(7),
    "numpy.int64": np.int64(7),
    "numpy.uint8": np.uint8(7),
    "numpy.uint16": np.uint16(7),
    "numpy.uint32": np.uint32(7),
    "numpy.uint64": np.uint64(7),
    "numpy.float16": np.float16(7.5),
    "numpy.float32": np.float32(7.5),
    "numpy.longdouble": np.longdouble(7.5),
    "numpy.bool_": np.bool_(True),
    "numpy.bytes_": np.bytes_(b"c"),
    "numpy.complex64": np.complex64(1 + 2j),
    "numpy.complex128": np.complex128(1 + 2j),
    "numpy.datetime64": np.datetime64("2024-01-01"),
    "numpy.timedelta64": np.timedelta64(1, "D"),
    "tuple": ("a", "b"),
}
KINDS = ("numeric", "categorical", "declared")


def _frame(label: Any, kind: str, n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cat = rng.choice(["a", "b", "c"], size=n)
    if kind == "numeric":
        column: Any = rng.normal(size=n)
    elif kind == "categorical":
        column = pd.Series(cat, dtype=object)
    else:
        column = pd.Series(cat).astype(pd.CategoricalDtype(["a", "b", "c"]))
    df = pd.DataFrame({"x": rng.normal(size=n)})
    df[label] = column if not isinstance(column, pd.Series) else column.to_numpy()
    if kind == "declared":
        df[label] = df[label].astype(pd.CategoricalDtype(["a", "b", "c"]))
    df["target"] = (pd.Series(cat) == "a").astype(float) + rng.normal(scale=0.1, size=n)
    return df


def _stage(fn: Any) -> tuple[str, Any]:
    try:
        return "ok", fn()
    except Exception as exc:  # noqa: BLE001 -- the outcome is the measurement
        return type(exc).__name__, None


def _predict_module(project: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"p{id(project)}", project / "predict.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _case(label: Any, kind: str) -> dict[str, str]:
    out: dict[str, str] = {}
    df = _frame(label, kind)
    X = df.drop(columns=["target"])
    model = Model(make_config("regression", n_estimators=3, n_splits=2))
    out["fit"], _ = _stage(lambda: model.fit(data=df))
    if out["fit"] != "ok":
        return out
    expected = np.asarray(model.predict(X).pred)
    project = Path(tempfile.mkdtemp()) / "code"
    out["export_code"], _ = _stage(lambda: model.export_code(project))
    if out["export_code"] != "ok":
        return out
    module_status, module = _stage(lambda: _predict_module(project))
    if module_status != "ok":
        out["typed"] = out["csv"] = f"load:{module_status}"
        return out

    def typed() -> str:
        got = np.asarray(module.predict(X)["pred"])
        return "same" if np.allclose(got, expected, rtol=1e-7, atol=0) else "DIFFERS"

    def csv() -> str:
        path = project / "x.csv"
        X.to_csv(path, index=False)
        got = np.asarray(module.predict(pd.read_csv(path))["pred"])
        return "same" if np.allclose(got, expected, rtol=1e-7, atol=0) else "DIFFERS"

    status, value = _stage(typed)
    out["typed"] = value if status == "ok" else status
    status, value = _stage(csv)
    out["csv"] = value if status == "ok" else status
    return out


def main() -> None:
    print(f"lizyml {lizyml.__version__} pandas {pd.__version__} numpy {np.__version__}")
    tally: Counter[str] = Counter()
    for name, label in LABELS.items():
        for kind in KINDS:
            result = _case(label, kind)
            print(f"{name:18} {kind:12} {result}")
            tally["cases"] += 1
            if result.get("fit") == "ok":
                tally["fit"] += 1
                tally["export " + result.get("export_code", "?")] += 1
                if result.get("typed") == "same":
                    tally["typed same"] += 1
                if result.get("csv") == "same":
                    tally["csv same"] += 1
    print(dict(tally))


if __name__ == "__main__":
    main()
