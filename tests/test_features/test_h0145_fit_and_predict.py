"""H-0145 acceptance 3, 4, 6a, 6b (#391): models that meet a frozenset that
does not read back as written, on the NumPy where it does not.

Owner-accepted consequences (HISTORY.md H-0145, 管理者の判断 1-5): such a
column takes the pandas path at fit and cannot be exported; at predict such a
value is unseen, even when a V-safe value with the same identity key was
fitted.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml import Model
from lizyml.core.category_values import identity_eligible
from lizyml.core.exceptions import ErrorCode, LizyMLError
from tests._helpers import make_config
from tests.test_core.test_h0145_round_trip import CASES, REASON, collapsing


def _frame(values: list[Any]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = values * 20
    return pd.DataFrame(
        {
            "c": pd.Series(rows, dtype=object),
            "num": rng.normal(size=len(rows)),
            "target": rng.normal(size=len(rows)),
        }
    )


def _fit(
    values: list[Any], policy: str = "mode"
) -> tuple[Model, list[warnings.WarningMessage]]:
    config = make_config("regression", n_estimators=5, n_splits=2)
    config["features"] = {**config.get("features", {}), "unseen_policy": policy}
    model = Model(config)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(data=_frame(values))
    return model, caught


def _encoder(model: Model) -> dict[str, Any]:
    state: dict[str, Any] = model._refit_result.pipeline_state["encoder"]  # noqa: SLF001
    return state


def _unsaveable_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [
        str(w.message) for w in caught if "a saved model cannot hold" in str(w.message)
    ]


def _assert_exports_refused(model: Model, tmp_path: Path) -> None:
    for name, export in (
        ("export", model.export),
        ("export_code", model.export_code),
    ):
        target = tmp_path / name
        with pytest.raises(LizyMLError) as excinfo:
            export(target)
        assert excinfo.value.code is ErrorCode.SERIALIZATION_FAILED
        assert excinfo.value.context["column"] == "c"
        assert excinfo.value.context["type"] == "frozenset"
        assert not target.exists() or not any(target.iterdir())


# --- acceptance 4 and 3: fit takes the pandas path; the exits refuse ----------------


@pytest.mark.parametrize("name", list(CASES))
def test_fit_returns_with_one_warning_and_the_exits_refuse(
    name: str, tmp_path: Path
) -> None:
    value = collapsing(name)
    model, caught = _fit([value, "x"])
    found = _unsaveable_warnings(caught)
    assert len(found) == 1 and REASON in found[0]
    assert "c" not in _encoder(model)["identity_columns"]
    _assert_exports_refused(model, tmp_path)


# --- acceptance 6a: at predict the value is unseen --------------------------------


def _predict_with(model: Model, value: Any) -> Any:
    frame = _frame(["x"]).drop(columns=["target"]).head(2).copy()
    frame["c"] = pd.Series([value, "x"], dtype=object, index=frame.index)
    return model.predict(frame).pred


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("policy", ["mode", "nan", "error"])
def test_a_value_first_met_at_predict_is_unseen(name: str, policy: str) -> None:
    value = collapsing(name)
    model, _ = _fit(["a", "b", "x"], policy)
    assert "c" in _encoder(model)["identity_columns"]
    if policy == "error":
        with pytest.raises(LizyMLError) as excinfo:
            _predict_with(model, value)
        assert excinfo.value.code is ErrorCode.DATA_SCHEMA_INVALID
    else:
        np.testing.assert_array_equal(
            _predict_with(model, value), _predict_with(model, "never-seen")
        )


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("policy", ["nan", "error"])
def test_it_is_unseen_even_beside_a_fitted_key_equal_value(
    name: str, policy: str
) -> None:
    """Accepted consequence 4: before H-0145 the value matched the twin."""
    value = collapsing(name)
    twin = CASES[name][1]
    model, _ = _fit([twin, "x"], policy)
    assert "c" in _encoder(model)["identity_columns"]
    if policy == "error":
        with pytest.raises(LizyMLError) as excinfo:
            _predict_with(model, value)
        assert excinfo.value.code is ErrorCode.DATA_SCHEMA_INVALID
    else:
        np.testing.assert_array_equal(
            _predict_with(model, value), _predict_with(model, "never-seen")
        )
        assert not np.array_equal(
            _predict_with(model, value), _predict_with(model, twin)
        )


# --- acceptance 6b: a column holding both takes the pandas path -------------------


def test_a_column_holding_the_value_and_its_twin_takes_the_pandas_path(
    tmp_path: Path,
) -> None:
    """Accepted consequence 5 on NumPy 1.24: before H-0145 the twin absorbed
    the value by key; now the column is cast by pandas, which keeps the value
    as its own category (the two hash differently), so export refuses it."""
    value = collapsing("pair")
    twin = CASES["pair"][1]
    model, caught = _fit([twin, twin, value, "x"])
    assert "c" not in _encoder(model)["identity_columns"]
    assert len(_unsaveable_warnings(caught)) == 1
    _assert_exports_refused(model, tmp_path)


def test_on_numpy_2_pandas_merges_the_value_into_its_twin(tmp_path: Path) -> None:
    """Consequence 1 on NumPy 2.x: the column is cast by pandas, which merges
    the large-uint64 value into its twin (equal hashes), so the state holds
    only the twin, the encoder keys it, and the model exports."""
    value = collapsing("big")
    twin = CASES["big"][1]
    frame = _frame([twin, twin, value, "x"])
    assert not identity_eligible(frame["c"])  # the builder's path: pandas
    model, caught = _fit([twin, twin, value, "x"])
    assert _unsaveable_warnings(caught) == []
    assert _encoder(model)["categories"]["c"] == [twin, "x"]
    model.export(tmp_path / "export")  # nothing outside V is saved
