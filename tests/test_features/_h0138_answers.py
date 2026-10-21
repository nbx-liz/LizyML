"""Compute the answers the H-0138 oracle froze, for the running library.

Run as ``python -m tests.test_features._h0138_answers`` under
``PYTHONHASHSEED=0``, the seed the oracle was made with: today's order of
some mixed columns (a tuple beside a frozenset) follows string hashing, so
the comparison is made under the same seed. Prints one JSON object
``{"cases": {id: answer}, "temporal_cases": {id: outcome}}``.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from typing import Any

import pandas as pd

from lizyml.features.encoders.categorical_encoder import CategoricalEncoder
from tests.test_features import _h0138_oracle as h


def encoder_answer(train: pd.Series, batch: list[Any]) -> dict[str, Any]:
    frame = pd.DataFrame({"c": train})
    if not isinstance(train.dtype, pd.CategoricalDtype):
        frame["c"] = frame["c"].astype("category")
    enc = CategoricalEncoder(unseen_policy="mode").fit(frame, ["c"])
    state = enc.get_state()
    codes = []
    for order in (batch, list(reversed(batch))):
        out = enc.transform(pd.DataFrame({"c": pd.Series(order, dtype=object)}))
        codes.append([int(x) for x in out["c"].cat.codes])
    return {
        "categories": [h.typed(v) for v in state["categories"]["c"]],
        "mode": None if state["modes"]["c"] is None else h.typed(state["modes"]["c"]),
        "stand_in": state["stand_in_columns"],
        "batch": [h.typed(v) if v is not None else None for v in batch],
        "codes_forward": codes[0],
        "codes_reversed": codes[1],
    }


def temporal_answer(train: pd.Series, values: list[Any]) -> dict[str, Any]:
    frame = pd.DataFrame({"c": train})
    try:
        frame["c"] = frame["c"].astype("category")
    except Exception as exc:  # noqa: BLE001 -- the frozen outcome is the failure
        return {"fails_at": "cast", "exception": type(exc).__name__}
    try:
        enc = CategoricalEncoder(unseen_policy="mode").fit(frame, ["c"])
    except Exception as exc:  # noqa: BLE001 -- the frozen outcome is the failure
        return {"fails_at": "fit", "exception": type(exc).__name__}
    batch = [*h.distinct(values), "x"]
    codes = []
    for order in (batch, list(reversed(batch))):
        try:
            out = enc.transform(pd.DataFrame({"c": pd.Series(order, dtype=object)}))
        except Exception as exc:  # noqa: BLE001 -- the frozen outcome is the failure
            return {"fails_at": "transform", "exception": type(exc).__name__}
        codes.append([int(x) for x in out["c"].cat.codes])
    state = enc.get_state()
    return {
        "categories": [h.typed(v) for v in state["categories"]["c"]],
        "stand_in": state["stand_in_columns"],
        "codes_forward": codes[0],
        "codes_reversed": codes[1],
    }


def _inputs(case_id: str) -> tuple[pd.Series, list[Any]]:
    if case_id.startswith("o-"):
        return h.ordinary(case_id)
    values = h.probe_values(case_id)
    return pd.Series([*values, *values, "x"], dtype=object), [*h.distinct(values), "x"]


def main() -> None:
    assert os.environ.get("PYTHONHASHSEED") == "0", "run under PYTHONHASHSEED=0"
    warnings.simplefilter("ignore")
    oracle = h.oracle()
    answers: dict[str, Any] = {}
    for case in oracle["cases"]:
        try:
            answers[case["id"]] = encoder_answer(*_inputs(case["id"]))
        except Exception as exc:  # noqa: BLE001 -- a raised answer is a mismatch
            answers[case["id"]] = {"raised": f"{type(exc).__name__}: {exc}"[:300]}
    temporal: dict[str, Any] = {}
    for case in oracle["temporal_cases"]:
        values = h.probe_values(case["id"])
        train = pd.Series([*values, *values, "x"], dtype=object)
        temporal[case["id"]] = temporal_answer(train, values)
    json.dump({"cases": answers, "temporal_cases": temporal}, sys.stdout)


if __name__ == "__main__":
    main()
