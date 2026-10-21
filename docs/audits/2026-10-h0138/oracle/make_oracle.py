"""H-0138 acceptance 2: freeze what today's code does for every faithful column.

Run once, BEFORE H-0138 is implemented (develop ``b297de4`` behavior), it
writes ``tests/test_features/data/h0138_oracle_pandas-<version>.json.gz``. The test of
acceptance 2 compares the implemented library against this file; nothing in
it is recomputed by the code under test.

Population:

- **encoder cases**: every column that the H-0138 gate classifies as
  ``faithful`` (``firing/h0138_rule.classify``), built as in
  ``identity_rows.py`` (the H-0133 SAMPLES singles and pairs and the issues'
  own shapes, each as training rows ``[*col, *col, "x"]``), plus ordinary
  inputs of every categorical dtype (object, pandas string, declared category
  with an unused category, explicit int64 / float64 / bool, missing values).
  For each: the categories with their types, the mode, ``stand_in_columns``,
  and the codes of two transform batches (forward and reversed, ``unseen``
  under ``mode``).
- **temporal cases** (acceptance 11): every probe column holding a temporal
  value, with exactly today's outcome: the categories and codes, or the
  step (cast / fit / transform) and exception type where it fails today.
- **model cases**: the ordinary inputs, the columns that hold two numerically
  equal values of different types, and every 50th encoder case (counted over the frozen encoder cases). For
  each: ``Model.fit`` (fixed seed, one thread), then the refit booster's
  ``pandas_categorical`` for the column, the sha256 of ``model_to_string()``,
  and the predictions (``float.hex``).

    PYTHONHASHSEED=0 PYTHONPATH=.:docs/audits/2026-10-h0138/firing \
        python docs/audits/2026-10-h0138/oracle/make_oracle.py
"""

from __future__ import annotations

import decimal
import gzip
import hashlib
import io
import itertools
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import pandas as pd
from h0138_rule import _number, classify

from lizyml import Model
from lizyml.core.category_values import encode
from lizyml.features.encoders.categorical_encoder import CategoricalEncoder
from tests.test_core.test_category_values import SAMPLES

warnings.simplefilter("ignore")
OUT_DIR = Path("tests/test_features/data")


def _typed(value: Any) -> Any:
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}", "value": encode(value)}


def _wrap(value: Any, depth: int) -> Any:
    for _ in range(depth):
        value = (value,)
    return value


def _distinct(values: list[Any]) -> list[Any]:
    out: list[Any] = []
    for v in values:
        if not any(v is d for d in out):
            out.append(v)
    return out


def _probe_columns() -> list[tuple[str, list[Any]]]:
    samples = [v for vs in SAMPLES.values() for v in vs]
    cols = [(f"s{i}", [v]) for i, v in enumerate(samples)]
    cols += [
        (f"p{i}-{j}", [samples[i], samples[j]])
        for i, j in itertools.combinations(range(len(samples)), 2)
    ]
    ones = [t(1) for t in (np.bool_, np.int8, np.int16, np.int32, np.int64,
                           np.uint8, np.uint16, np.uint32, np.uint64,
                           np.float32, np.float64, np.complex64, np.complex128)]  # fmt: skip
    for k, one in enumerate(ones):
        cols.append((f"d{k}-a", [decimal.Decimal("1"), one]))
        cols.append((f"d{k}-b", [one, decimal.Decimal("1")]))
        for depth in (1, 2, 3, 8):
            cols.append((f"t{k}-{depth}-a", [one, _wrap(1, depth)]))
            cols.append((f"t{k}-{depth}-b", [_wrap(1, depth), one]))
    return cols


def _ordinary() -> list[tuple[str, pd.Series, list[Any]]]:
    """(id, training series, transform values) of every categorical dtype."""
    n = 24
    letters = ["a", "b", "c", "b", "a", "a"] * (n // 6)
    return [
        ("o-object", pd.Series(letters, dtype=object), ["a", "b", "c", "z"]),
        ("o-string", pd.Series(letters, dtype="string"), ["a", "b", "c", "z"]),
        (
            "o-declared",
            pd.Series(pd.Categorical(letters, categories=["c", "a", "b", "unused"])),
            ["a", "b", "c", "unused"],
        ),
        ("o-int64", pd.Series([1, 2, 3, 2] * (n // 4), dtype="int64"), [1, 2, 3, 9]),
        ("o-float64", pd.Series([0.5, 1.5, 2.5, 1.5] * (n // 4)), [0.5, 1.5, 2.5, 9.5]),
        ("o-bool", pd.Series([True, False, True] * (n // 3)), [True, False]),
        (
            "o-missing",
            pd.Series(["a", None, "b", "a"] * (n // 4), dtype=object),
            ["a", "b", None, "z"],
        ),
        ("o-mixed", pd.Series([1, "a", 2, "a"] * (n // 4), dtype=object), [1, 2, "a", "z"]),
    ]


def _encoder_case(train: pd.Series, batch: list[Any]) -> dict[str, Any]:
    frame = pd.DataFrame({"c": train})
    if not isinstance(train.dtype, pd.CategoricalDtype):
        frame["c"] = frame["c"].astype("category")  # as dataframe_builder does
    enc = CategoricalEncoder(unseen_policy="mode").fit(frame, ["c"])
    state = enc.get_state()
    codes = []
    for order in (batch, list(reversed(batch))):
        out = enc.transform(pd.DataFrame({"c": pd.Series(order, dtype=object)}))
        codes.append([int(x) for x in out["c"].cat.codes])
    return {
        "categories": [_typed(v) for v in state["categories"]["c"]],
        "mode": None if state["modes"]["c"] is None else _typed(state["modes"]["c"]),
        "stand_in": state["stand_in_columns"],
        "batch": [_typed(v) if v is not None else None for v in batch],
        "codes_forward": codes[0],
        "codes_reversed": codes[1],
    }


def _model_case(train: pd.Series) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    n = len(train)
    df = pd.DataFrame(
        {"c": train.reset_index(drop=True), "num": rng.normal(size=n), "target": rng.normal(size=n)}
    )
    cfg = {
        "config_version": 1,
        "task": "regression",
        "data": {"target": "target"},
        "features": {"categorical": ["c"]},
        "split": {"method": "kfold", "n_splits": 2, "random_state": 42},
        "model": {
            "name": "lgbm",
            "params": {"n_estimators": 5, "num_threads": 1, "verbose": -1},
        },
        "training": {"seed": 0},
    }
    model = Model(cfg)
    model.fit(data=df)
    booster = model._refit_result.model.get_native_model()  # noqa: SLF001
    position = list(booster.feature_name()).index("c")
    pandas_categorical = booster.pandas_categorical or []
    cats = pandas_categorical[0] if pandas_categorical else None
    preds = model.predict(df.drop(columns=["target"])).pred
    return {
        "pandas_categorical": None if cats is None else [_typed(v) for v in cats],
        "feature_position": position,
        "model_sha256": hashlib.sha256(booster.model_to_string().encode()).hexdigest(),
        "pred_hex": [float(p).hex() for p in np.asarray(preds, dtype=float)],
    }


def _temporal_outcome(train: pd.Series, values: list[Any]) -> dict[str, Any]:
    """Today's result for a column holding a temporal value, or where it fails."""
    frame = pd.DataFrame({"c": train})
    try:
        frame["c"] = frame["c"].astype("category")
    except Exception as exc:  # noqa: BLE001 -- the outcome being frozen
        return {"fails_at": "cast", "exception": type(exc).__name__}
    try:
        enc = CategoricalEncoder(unseen_policy="mode").fit(frame, ["c"])
    except Exception as exc:  # noqa: BLE001 -- the outcome being frozen
        return {"fails_at": "fit", "exception": type(exc).__name__}
    batch = [*_distinct(values), "x"]
    codes = []
    for order in (batch, list(reversed(batch))):
        try:
            out = enc.transform(pd.DataFrame({"c": pd.Series(order, dtype=object)}))
        except Exception as exc:  # noqa: BLE001 -- the outcome being frozen
            return {"fails_at": "transform", "exception": type(exc).__name__}
        codes.append([int(x) for x in out["c"].cat.codes])
    state = enc.get_state()
    return {
        "categories": [_typed(v) for v in state["categories"]["c"]],
        "stand_in": state["stand_in_columns"],
        "codes_forward": codes[0],
        "codes_reversed": codes[1],
    }


def main() -> None:
    cases: list[dict[str, Any]] = []
    model_ids: set[str] = set()
    probe = _probe_columns()
    skipped: dict[str, int] = {}
    temporal: list[dict[str, Any]] = []
    for cid, values in probe:
        train = pd.Series([*values, *values, "x"], dtype=object)
        path = classify(train)
        if path == "temporal":
            # Acceptance 11: exactly today's outcome, a failure included.
            temporal.append({"id": cid, "outcome": _temporal_outcome(train, values)})
            continue
        if path != "faithful":
            skipped[path] = skipped.get(path, 0) + 1
            continue
        batch = [*_distinct(values), "x"]
        try:
            enc = _encoder_case(train, batch)
        except Exception as exc:  # noqa: BLE001 -- broken today: not frozen
            key = f"faithful but transform {type(exc).__name__}"
            skipped[key] = skipped.get(key, 0) + 1
            continue
        if enc["codes_forward"] != list(reversed(enc["codes_reversed"])):
            skipped["faithful but batch-order codes"] = skipped.get("faithful but batch-order codes", 0) + 1
            continue
        # The training rows are encoder.batch without "x", twice, then "x".
        record = {"id": cid, "encoder": enc}
        numbers = [_number(v) for v in values]
        cross_type = (
            len(values) == 2
            and None not in numbers
            and numbers[0] == numbers[1]
            and type(values[0]) is not type(values[1])
        )
        # Every 50th encoder case, counted over the frozen cases only.
        if cross_type or len(cases) % 50 == 0:
            model_ids.add(cid)
            record["model"] = _model_case(pd.Series([*values, "x"] * 8, dtype=object))
        cases.append(record)
    # Self-check (round 2): every 50th frozen probe case carries a model case.
    missing = [i for i in range(0, len(cases), 50) if "model" not in cases[i]]
    assert not missing, f"model sampling missed encoder cases {missing}"
    for cid, train, batch in _ordinary():
        record = {
            "id": cid,
            "dtype": str(train.dtype),
            "train": [_typed(v) if not pd.isna(v) else None for v in train.astype(object)],
            "encoder": _encoder_case(train, batch),
            "model": _model_case(train),
        }
        model_ids.add(cid)
        cases.append(record)
    meta = {
        "about": "H-0138 acceptance 2: today's behavior for every faithful column, frozen before implementation",
        "generator": "docs/audits/2026-10-h0138/oracle/make_oracle.py",
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "lightgbm": lightgbm.__version__,
        "encoder_cases": len(cases),
        "model_cases": len(model_ids),
        "temporal_cases": len(temporal),
        "probe_columns_not_frozen": skipped,
    }
    out = OUT_DIR / f"h0138_oracle_pandas-{pd.__version__}.json.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "cases": cases, "temporal_cases": temporal}
    raw = (json.dumps(payload, sort_keys=True, indent=0) + "\n").encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as fh:
        fh.write(raw)
    out.write_bytes(buffer.getvalue())
    print(out, json.dumps(meta))


if __name__ == "__main__":
    main()
