"""Feature names LightGBM's model text cannot hold intact are refused before
saving, and the fit warns (H-0147, #406).

LightGBM reads its model text line by line, and the ``feature_names=`` line
separates names by spaces. An empty name disappears from that line, and a
name holding ``\\n`` or ``\\r`` cuts it, so the saved booster either cannot be
read back (``Wrong size of feature_names``) or reads back with other names.
Before H-0147 ``Model.export`` wrote such a model and ``Model.load`` failed.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.model import Model
from lizyml.estimators.base import (
    BaseEstimatorAdapter,
    UnsaveableCategory,
    UnsaveableFeatureName,
)
from lizyml.estimators.lgbm import LGBMAdapter
from lizyml.persistence import exporter
from tests._helpers import make_config

N = 120

#: Feature frames whose names the save check refuses, with the refused name's
#: position and reason.
REFUSED: dict[str, tuple[list[Any], int, str]] = {
    "empty last": (["x", ""], 1, "empty_feature_name"),
    "empty middle": (["x", "", "w"], 1, "empty_feature_name"),
    "np.str_ empty": (["x", np.str_("")], 1, "empty_feature_name"),
    "line feed last": (["x", "a\nb"], 1, "line_break_in_feature_name"),
    "line feed first": (["\nb", "x"], 0, "line_break_in_feature_name"),
    "carriage return": (["a\rb", "x"], 0, "line_break_in_feature_name"),
}

#: Names that fit, save, load and predict the same (H-0147 acceptance 3).
ADMITTED: list[Any] = ["z", " ", "a b", "a\tb", " a", "列", "a=b", 0]

#: Names the LightGBM rule is checked against, first and last (acceptance 5).
RULE_NAMES: list[str] = [
    "z",
    "",
    " ",
    "a b",
    "a\tb",
    "\tb",
    "a\nb",
    "\nb",
    "a\n",
    "a\rb",
    "\rb",
    "a\r\nb",
    "a\x0bb",
    "a\x0cb",
    "a\x00b",
    " a",
    "列",
    "a=b",
    "=b",
    "a　b",
    "a b",
]


def _frame(names: list[Any], *, categorical: Any = None) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({name: rng.normal(size=N) for name in names})
    if categorical is not None:
        df[categorical] = pd.Categorical(["a", "b"] * (N // 2))
    df["target"] = (rng.normal(size=N) > 0).astype(int)
    return df


def _fit(df: pd.DataFrame, **extra: Any) -> tuple[Model, list[str]]:
    raw = make_config("binary", n_estimators=5)
    raw.update(extra)
    model = Model(raw)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(data=df)
    return model, [str(w.message) for w in caught if "feature name" in str(w.message)]


def _features(df: pd.DataFrame, target: Any = "target") -> pd.DataFrame:
    return df.drop(columns=[target])


# --- Acceptance 1: Model.export and persistence.export refuse ---------------


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_model_export_refuses_and_writes_nothing(case: str, tmp_path: Path) -> None:
    names, position, reason = REFUSED[case]
    model, _ = _fit(_frame(names))
    out = tmp_path / "out"
    with pytest.raises(LizyMLError) as exc:
        model.export(out)
    assert exc.value.code == ErrorCode.SERIALIZATION_FAILED
    assert exc.value.context == {
        "column": str(names[position]),
        "position": position,
        "state": "refit",
        "reason": reason,
        "surface": "Model.export",
    }
    assert not out.exists()


def test_an_empty_categorical_feature_name_is_refused(tmp_path: Path) -> None:
    model, _ = _fit(_frame(["x"], categorical=""))
    with pytest.raises(LizyMLError) as exc:
        model.export(tmp_path / "out")
    assert exc.value.context["reason"] == "empty_feature_name"
    assert exc.value.context["column"] == ""
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("case", ["empty last", "line feed first"])
def test_persistence_export_refuses_and_writes_nothing(
    case: str, tmp_path: Path
) -> None:
    names, position, reason = REFUSED[case]
    model, _ = _fit(_frame(names))
    out = tmp_path / "out"
    with pytest.raises(LizyMLError) as exc:
        exporter.export(out, model._fit_result, model._refit_result, {}, "binary")
    assert exc.value.code == ErrorCode.SERIALIZATION_FAILED
    assert exc.value.context["reason"] == reason
    assert exc.value.context["position"] == position
    assert exc.value.context["surface"] == "Model.export"
    assert not out.exists()


# --- Acceptance 2: export_code ----------------------------------------------


def test_export_code_keeps_its_empty_alias_refusal(tmp_path: Path) -> None:
    model, _ = _fit(_frame(["x", ""]))
    with pytest.raises(LizyMLError) as exc:
        model.export_code(tmp_path / "code")
    assert exc.value.code == ErrorCode.SERIALIZATION_FAILED
    assert exc.value.context["kind"] == "empty_alias"
    assert sorted(exc.value.context) == [
        "alias",
        "columns",
        "kind",
        "reason",
        "surface",
    ]
    assert not (tmp_path / "code").exists()


def test_export_code_refuses_a_line_break_name(tmp_path: Path) -> None:
    model, _ = _fit(_frame(["x", "a\nb"]))
    with pytest.raises(LizyMLError) as exc:
        model.export_code(tmp_path / "code")
    assert exc.value.code == ErrorCode.SERIALIZATION_FAILED
    assert exc.value.context == {
        "column": "a\nb",
        "position": 1,
        "state": "refit",
        "reason": "line_break_in_feature_name",
        "surface": "export_code",
    }
    assert not (tmp_path / "code").exists()


# --- Acceptance 3: admitted names still round-trip --------------------------


def _round_trip(model: Model, X: pd.DataFrame, out: Path) -> None:
    before = np.asarray(model.predict(X).pred)
    model.export(out)
    after = np.asarray(Model.load(out).predict(X).pred)
    np.testing.assert_allclose(before, after)


@pytest.mark.parametrize("name", ADMITTED, ids=repr)
def test_admitted_feature_names_round_trip(name: Any, tmp_path: Path) -> None:
    df = _frame(["x", name])
    model, messages = _fit(df)
    assert messages == []
    _round_trip(model, _features(df), tmp_path / "out")


def test_an_empty_target_or_excluded_name_still_round_trips(tmp_path: Path) -> None:
    df = _frame(["x", "w"]).rename(columns={"target": ""})
    raw = make_config("binary", n_estimators=5)
    raw["data"] = {"target": ""}
    model = Model(raw)
    model.fit(data=df)
    _round_trip(model, _features(df, ""), tmp_path / "target")

    df = _frame(["x", ""])
    model, messages = _fit(df, features={"exclude": [""]})
    assert messages == []
    _round_trip(model, _features(df), tmp_path / "excluded")


# --- Acceptance 4: the fit warning ------------------------------------------


@pytest.mark.parametrize("case", ["empty last", "line feed last"])
def test_fit_warns_once_and_predict_works(case: str) -> None:
    names, position, _ = REFUSED[case]
    df = _frame(names)
    model, messages = _fit(df)
    assert len(messages) == 1
    assert f"feature name at position {position}" in messages[0]
    assert "Model.export and export_code will refuse" in messages[0]
    assert len(model.predict(_features(df)).pred) == N


def test_fit_with_two_refused_names_warns_for_each() -> None:
    _, messages = _fit(_frame(["", "x", "a\nb"]))
    assert len(messages) == 2


# --- Acceptance 5: the rule against the installed LightGBM ------------------


def _booster(columns: list[str]) -> lgb.Booster:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({c: rng.normal(size=N) for c in columns})
    y = (X[columns[0]] + X[columns[1]] > 0).astype(int)
    return lgb.train(
        {"objective": "binary", "verbose": -1},
        lgb.Dataset(X, y),
        num_boost_round=2,
        keep_training_booster=True,
    )


def _reads_back_intact(booster: lgb.Booster) -> bool:
    try:
        back = lgb.Booster(model_str=booster.model_to_string())
    except lgb.basic.LightGBMError:
        return False
    return bool(back.feature_name() == booster.feature_name())


@pytest.mark.parametrize("position", ["first", "last"])
@pytest.mark.parametrize("name", RULE_NAMES, ids=repr)
def test_the_rule_matches_lightgbm(name: str, position: str) -> None:
    """Refused exactly when LightGBM cannot read the model text back with the
    same names. A LightGBM that changes its model text fails this test, which
    is the cue to revisit the rule (DC7)."""
    columns = [name, "x"] if position == "first" else ["x", name]
    booster = _booster(columns)
    adapter = LGBMAdapter()
    adapter._model = booster
    refused = bool(adapter.unsaveable_feature_names())
    assert refused is not _reads_back_intact(booster)


#: Name lists where ``=`` decides (LightGBM drops a ``=`` at the edges of the
#: joined names only when they split at ``=`` into one piece).
EQUALS_CASES: list[list[str]] = [
    ["=b", "x"],
    ["==b", "x"],
    ["x", "a="],
    ["=", "x"],
    ["x", "="],
    ["=a", "b="],
    ["=b", "c=d"],
    ["x", "a=", "c=d"],
    ["a", "=", "b"],
    ["=b", "x", "=c"],
    ["b=", "x"],
    ["x", "=b"],
]


@pytest.mark.parametrize("columns", EQUALS_CASES, ids=repr)
def test_the_rule_matches_lightgbm_for_equals_signs(columns: list[str]) -> None:
    booster = _booster(columns)
    adapter = LGBMAdapter()
    adapter._model = booster
    refused = bool(adapter.unsaveable_feature_names())
    assert refused is not _reads_back_intact(booster)


def test_an_equals_sign_at_the_edge_is_refused(tmp_path: Path) -> None:
    model, messages = _fit(_frame(["=b", "x"]))
    assert len(messages) == 1
    with pytest.raises(LizyMLError) as exc:
        model.export(tmp_path / "out")
    assert exc.value.context["reason"] == "equals_sign_at_feature_names_edge"
    assert exc.value.context["position"] == 0
    assert not (tmp_path / "out").exists()


def test_the_lightgbm_adapter_answers_in_position_order() -> None:
    adapter = LGBMAdapter()
    adapter._model = _booster(["\nb", "x", ""])
    assert adapter.unsaveable_feature_names() == [
        UnsaveableFeatureName(
            position=0, name="\nb", reason="line_break_in_feature_name"
        ),
        UnsaveableFeatureName(position=2, name="", reason="empty_feature_name"),
    ]


def test_an_unfitted_lightgbm_adapter_refuses_nothing() -> None:
    assert LGBMAdapter().unsaveable_feature_names() == []


# --- Acceptance 6: every saved adapter is asked -----------------------------


class _Adapter(BaseEstimatorAdapter):
    """An adapter that cannot train; only the save questions matter."""

    def fit(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> Any:
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> Any:
        raise NotImplementedError

    def predict_raw(self, X: pd.DataFrame) -> Any:
        raise NotImplementedError

    def importance(self, kind: Any = "split") -> dict[str, float]:
        raise NotImplementedError

    def get_native_model(self) -> Any:
        raise NotImplementedError

    def save_model_text(self, path: str | Path) -> Path:
        raise NotImplementedError


class _EmptyName(_Adapter):
    def unsaveable_feature_names(self) -> list[UnsaveableFeatureName]:
        return [UnsaveableFeatureName(position=3, name="", reason="empty_feature_name")]


_GOOD = {"encoder": {"categories": {"c": ["a", "b"]}}}


def _results(
    per_fold: bool, fold_models: list[Any], refit_model: Any
) -> tuple[Any, Any]:
    states = [_GOOD] * len(fold_models) if per_fold else None
    fit_result = SimpleNamespace(
        pipeline_state_per_fold=states, pipeline_state=_GOOD, models=fold_models
    )
    return fit_result, SimpleNamespace(pipeline_state=_GOOD, model=refit_model)


def test_the_default_answer_is_empty() -> None:
    assert _Adapter().unsaveable_feature_names() == []


@pytest.mark.parametrize("per_fold", [True, False], ids=["per-fold states", "none"])
def test_the_first_fold_adapter_is_asked(per_fold: bool, tmp_path: Path) -> None:
    """Without per-fold states the state check asks only the last fold, but
    every fold's booster is saved: the name check asks all of them."""
    results = _results(per_fold, [_EmptyName(), _Adapter(), _Adapter()], _Adapter())
    out = tmp_path / "out"
    with pytest.raises(LizyMLError) as exc:
        exporter.export(out, *results, {}, "binary")
    assert exc.value.context == {
        "column": "",
        "position": 3,
        "state": "fold 1",
        "reason": "empty_feature_name",
        "surface": "Model.export",
    }
    assert not out.exists()


def test_the_refit_adapter_is_asked_first() -> None:
    results = _results(True, [_EmptyName(), _Adapter()], _EmptyName())
    with pytest.raises(LizyMLError) as exc:
        exporter.check_category_values(*results)
    assert exc.value.context["state"] == "refit"


def test_saved_adapters_lists_refit_and_every_fold() -> None:
    folds = [_Adapter(), _Adapter(), _Adapter()]
    refit = _Adapter()
    fit_result, refit_result = _results(False, folds, refit)
    assert exporter._saved_adapters(fit_result, refit_result) == [
        ("refit", refit),
        ("fold 1", folds[0]),
        ("fold 2", folds[1]),
        ("fold 3", folds[2]),
    ]


# --- Acceptance 7: reason on the two category-save refusals -----------------


class _Unwritable(_Adapter):
    def unsaveable_categories(self, pipeline_state: Any) -> list[UnsaveableCategory]:
        return [UnsaveableCategory(column="c", shown="'z'", type_name="str")]


def test_the_adapter_category_refusal_carries_its_reason() -> None:
    results = _results(True, [_Adapter()], _Unwritable())
    with pytest.raises(LizyMLError) as exc:
        exporter.check_category_values(*results)
    assert exc.value.context == {
        "column": "c",
        "value": "'z'",
        "type": "str",
        "state": "refit",
        "reason": "lightgbm_unwritable_category",
    }


def test_a_declared_value_outside_v_carries_its_reason() -> None:
    results = _results(True, [_Adapter()], _Adapter())
    with pytest.raises(LizyMLError) as exc:
        exporter.check_category_values(*results, {"c": [(np.longdouble(1),)]})
    assert exc.value.context["state"] == "declared categories"
    assert exc.value.context["reason"] == "unsaveable_category_value"
    assert sorted(exc.value.context) == ["column", "reason", "state", "type", "value"]
