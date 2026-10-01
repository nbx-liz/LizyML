"""The leakage check reports a column it could not compare (H-0107, #267).

``validate_no_target_leakage`` used to catch ``TypeError`` / ``ValueError``
from the comparison and move on, so a column it never checked looked exactly
like one it checked and found clean (``[]``). The reachable input is a
numeric-declared extension array whose ``__array__`` or ``isna`` raises
(``docs/audits/2026-09-defect-discovery/instruments/pr7_hostile_numeric_probe.py``).
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.data import validate_no_target_leakage, validators
from tests.test_data._hostile_arrays import HostileArray, HostileIsna, HostileOverflow

_VALUES = np.arange(10, dtype=float)
_TARGETS = {
    "int64": pd.Series(_VALUES.astype("int64")),
    "float64": pd.Series(_VALUES),
    "bool": pd.Series(_VALUES % 2 == 0),
    "Int64": pd.Series(_VALUES.astype("int64")).astype("Int64"),
    "complex128": pd.Series(_VALUES.astype("complex128")),
}
#: An OverflowError is not one the old handler caught: it pins that the
#: wrapper covers any comparison failure, not only TypeError / ValueError.
_SHAPES = {"array": HostileArray, "isna": HostileIsna, "overflow": HostileOverflow}


@pytest.mark.parametrize("raise_on_violation", [True, False])
@pytest.mark.parametrize("target", sorted(_TARGETS))
@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_unchecked_column_is_reported(
    shape: str, target: str, raise_on_violation: bool
) -> None:
    array_type = _SHAPES[shape]
    df = pd.DataFrame({"x": pd.Series(array_type(_VALUES)), "y": _TARGETS[target]})
    with pytest.raises(LizyMLError) as exc:
        validate_no_target_leakage(df, "y", raise_on_violation=raise_on_violation)
    assert exc.value.code == ErrorCode.DATA_SCHEMA_INVALID
    assert exc.value.context["column"] == "x"
    assert exc.value.context["target"] == "y"
    assert exc.value.cause is array_type.error


@pytest.mark.parametrize("order", ["unchecked_first", "leaking_first"])
@pytest.mark.parametrize("raise_on_violation", [True, False])
def test_unchecked_column_stops_the_check(order: str, raise_on_violation: bool) -> None:
    """Wherever the uncheckable column sits, the result is never the verdict
    on the other columns alone."""
    y = pd.Series(_VALUES)
    columns = {"bad": pd.Series(HostileArray(_VALUES)), "dup": y.copy()}
    if order == "leaking_first":
        columns = {"dup": columns["dup"], "bad": columns["bad"]}
    df = pd.DataFrame({**columns, "y": y})
    with pytest.raises(LizyMLError) as exc:
        validate_no_target_leakage(df, "y", raise_on_violation=raise_on_violation)
    if order == "unchecked_first" or not raise_on_violation:
        assert exc.value.code == ErrorCode.DATA_SCHEMA_INVALID
        assert exc.value.context["column"] == "bad"
    else:
        # The leaking column comes first and raises before "bad" is reached.
        assert exc.value.code == ErrorCode.LEAKAGE_SUSPECTED


@pytest.mark.parametrize("raise_on_violation", [True, False])
def test_ordinary_columns_are_unchanged(raise_on_violation: bool) -> None:
    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(size=20))
    clean = pd.DataFrame({"a": rng.normal(size=20), "b": list("ab") * 10, "y": y})
    assert (
        validate_no_target_leakage(clean, "y", raise_on_violation=raise_on_violation)
        == []
    )
    leaking = clean.assign(dup=y)
    if raise_on_violation:
        with pytest.raises(LizyMLError) as exc:
            validate_no_target_leakage(leaking, "y")
        assert exc.value.code == ErrorCode.LEAKAGE_SUSPECTED
        assert exc.value.context["leaking_column"] == "dup"
    else:
        warnings = validate_no_target_leakage(leaking, "y", raise_on_violation=False)
        assert len(warnings) == 1 and "'dup'" in warnings[0]


def test_no_silent_skip_remains() -> None:
    """No handler in the validators module ends in ``pass``."""
    tree = ast.parse(inspect.getsource(validators))
    silent = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler)
        and len(node.body) == 1
        and isinstance(node.body[0], ast.Pass)
    ]
    assert silent == []
    assert "Non-comparable" not in inspect.getsource(validators)
