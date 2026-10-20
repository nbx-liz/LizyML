"""H-0136 prototype rule: does a saved category state behave the same after its serializer?

This is the rule the Proposal measures, kept outside ``lizyml/`` (it is not
the implementation). For one encoder state it projects, per column, the
learned categories and the mode through each serializer a saved model uses:

- ``joblib``: the state through ``joblib.dump(..., compress=3)`` and back, as
  ``fit_result.pkl`` / ``refit_model.pkl`` hold it (``exporter.py:416-417``);
- ``json``: each value through ``encode`` -> ``json`` -> ``decode``, as the
  codegen ``pipeline_state.json`` holds it.

The library encoder (``unseen_policy="error"``) then transforms one row per
original category, with the original state and with the projected one. A
column breaks under a serializer when the projected transform raises (the
categories are not unique, or an original category is unseen), when a code
differs, or when the mode's code differs. Declared categories go through
``json`` only, and break when the projected ``CategoricalDtype`` is invalid or
gives the original values other codes.
"""

from __future__ import annotations

import copy
import io
import json
from typing import Any

import joblib
import pandas as pd

from lizyml.core.category_values import decode, encode
from lizyml.features.encoders.categorical_encoder import CategoricalEncoder


def _joblib(obj: Any) -> Any:
    buf = io.BytesIO()
    joblib.dump(obj, buf, compress=3)
    buf.seek(0)
    return joblib.load(buf)


def _json(value: Any) -> Any:
    return decode(json.loads(json.dumps(encode(value))))


def _codes(enc_state: dict[str, Any], column: str, values: list[Any]) -> list[int]:
    state = {
        "unseen_policy": "error",
        "categories": {column: enc_state["categories"][column]},
        "modes": {column: enc_state.get("modes", {}).get(column)},
        "stand_in_columns": [c for c in enc_state.get("stand_in_columns", []) if c == column],
    }
    enc = CategoricalEncoder().load_state(state)
    frame = pd.DataFrame({column: pd.Series(values, dtype=object)})
    return enc.transform(frame)[column].cat.codes.tolist()


def _project(enc_state: dict[str, Any], serializer: str) -> dict[str, Any]:
    if serializer == "joblib":
        return _joblib(copy.copy(enc_state))
    return {
        **enc_state,
        "categories": {c: [_json(v) for v in cats] for c, cats in enc_state["categories"].items()},
        "modes": {
            c: None if m is None else _json(m) for c, m in enc_state.get("modes", {}).items()
        },
    }


def state_breaks(state: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(breaks, original_fails) for one pipeline state."""
    enc_state = state.get("encoder") if isinstance(state, dict) else None
    if not isinstance(enc_state, dict) or not enc_state.get("categories"):
        return [], []
    breaks: list[dict[str, Any]] = []
    original_fails: list[dict[str, Any]] = []
    for column, cats in enc_state["categories"].items():
        cats = list(cats)
        mode = enc_state.get("modes", {}).get(column)
        try:
            want = _codes(enc_state, column, cats)
            want_mode = None if mode is None else _codes(enc_state, column, [mode])
        except Exception as exc:  # noqa: BLE001 -- recorded apart
            original_fails.append({"column": str(column), "error": type(exc).__name__})
            continue
        for serializer in ("joblib", "json"):
            try:
                projected = _project(enc_state, serializer)
                got = _codes(projected, column, cats)
                got_mode = None if mode is None else _codes(projected, column, [mode])
            except Exception as exc:  # noqa: BLE001 -- the break measured
                breaks.append(
                    {"column": str(column), "serializer": serializer, "kind": type(exc).__name__}
                )
                continue
            if got != want:
                kind = "codes"
            elif got_mode != want_mode:
                kind = "mode"
            else:
                continue
            breaks.append({"column": str(column), "serializer": serializer, "kind": kind})
    return breaks, original_fails


def declared_breaks(declared: Any) -> list[dict[str, Any]]:
    breaks: list[dict[str, Any]] = []
    for column, cats in (declared or {}).items():
        cats = list(cats)
        try:
            want = pd.Series(cats, dtype=object).astype(pd.CategoricalDtype(cats)).cat.codes
        except Exception:  # noqa: BLE001 -- the fit itself refuses such a dtype
            continue
        try:
            dtype = pd.CategoricalDtype([_json(v) for v in cats])
            got = pd.Series(cats, dtype=object).astype(dtype).cat.codes
        except Exception as exc:  # noqa: BLE001 -- the break measured
            breaks.append({"column": str(column), "serializer": "json", "kind": type(exc).__name__})
            continue
        if got.tolist() != want.tolist():
            breaks.append({"column": str(column), "serializer": "json", "kind": "codes"})
    return breaks
