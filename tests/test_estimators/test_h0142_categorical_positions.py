"""H-0142 acceptance 1-5: LightGBM receives categorical columns by position (#393).

LightGBM reads an ``int`` entry of ``categorical_feature`` as a position, so a
categorical column named ``1`` used to make column 1 categorical, and one
named ``7`` none at all. The adapter now hands LightGBM the positions of the
declared columns in the training frame.
"""

from __future__ import annotations

import warnings
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.estimators.lgbm import adapter as adapter_module
from lizyml.estimators.lgbm.adapter import LGBMAdapter, categorical_positions
from tests._helpers import make_config

LABELS = ["c", np.str_("c"), 0, 1, 7]


def _frame(label: Any, n: int = 180, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cat = rng.choice(["a", "b", "c"], size=n)
    df = pd.DataFrame({label: pd.Series(cat, dtype=object), "x": rng.normal(size=n)})
    df["target"] = (pd.Series(cat) == "a").astype(float) + rng.normal(scale=0.1, size=n)
    return df


def _fit(df: pd.DataFrame, **overrides: Any) -> Model:
    model = Model(make_config("regression", n_estimators=3, n_splits=2, **overrides))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data=df)
    return model


def _boosters(model: Model) -> list[Any]:
    refit = model._refit_result.model.get_native_model()  # type: ignore[union-attr]  # noqa: SLF001
    folds = [m.get_native_model() for m in model._fit_result.models]  # type: ignore[union-attr]  # noqa: SLF001
    return [refit, *folds]


# --- acceptance 1 -------------------------------------------------------------------


@pytest.mark.parametrize("label", LABELS, ids=repr)
def test_the_named_column_is_the_categorical_one(label: Any) -> None:
    model = _fit(_frame(label))
    for booster in _boosters(model):
        assert booster.params["categorical_column"] == [0]


@pytest.mark.parametrize("label", [0, 1, 7], ids=repr)
def test_an_int_name_trains_the_model_a_str_name_trains(label: Any) -> None:
    df = _frame(label)
    renamed = df.rename(columns={label: "c"})
    got = _fit(df).predict(df.drop(columns=["target"])).pred
    want = _fit(renamed).predict(renamed.drop(columns=["target"])).pred
    np.testing.assert_array_equal(got, want)


# --- acceptance 2 -------------------------------------------------------------------


def test_several_columns_with_mixed_names_are_declared_by_position() -> None:
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame(
        {
            5: rng.normal(size=n),
            "b": pd.Series(rng.choice(["p", "q"], size=n), dtype=object),
            2: pd.Series(rng.choice(["u", "v", "w"], size=n), dtype=object),
            "x": rng.normal(size=n),
            0: pd.Series(rng.choice(["k", "l"], size=n), dtype=object),
        }
    )
    df["target"] = rng.normal(size=n)
    model = _fit(df)
    for booster in _boosters(model):
        assert booster.params["categorical_column"] == [1, 2, 4]


# --- acceptance 3 -------------------------------------------------------------------


def test_the_valid_dataset_receives_the_same_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Any] = []
    real = lgb.Dataset

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs.get("categorical_feature"))
        return real(*args, **kwargs)

    monkeypatch.setattr(adapter_module.lgb, "Dataset", spy)
    df = _frame(1)
    X = df[[1, "x"]].astype({1: "category"})
    y = df["target"]
    adapter = LGBMAdapter(task="regression", params={"n_estimators": 3, "verbose": -1})
    adapter.set_categorical_features([1])
    adapter.fit(X.iloc[:120], y.iloc[:120], X.iloc[120:], y.iloc[120:])
    assert seen == [[0], [0]]


# --- acceptance 4 -------------------------------------------------------------------


def test_names_and_positions_train_the_same_booster_bytes() -> None:
    """The assumption that keeps string-named models byte-identical."""
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "c": pd.Categorical(rng.choice(["a", "b", "c"], size=n)),
            "d": pd.Categorical(rng.choice(["p", "q"], size=n)),
        }
    )
    y = rng.normal(size=n)
    params = {"objective": "regression", "verbose": -1, "num_threads": 1, "seed": 0}
    texts = [
        lgb.train(
            params,
            lgb.Dataset(X, label=y, categorical_feature=cat, free_raw_data=False),
            num_boost_round=5,
        ).model_to_string()
        for cat in (["c", "d"], [1, 2])
    ]
    assert texts[0] == texts[1]


# --- acceptance 5 -------------------------------------------------------------------


def test_matching_is_by_exact_type() -> None:
    columns = [1, True, 1.0, np.int64(1), "1"]
    assert categorical_positions(columns, [True]) == [1]
    assert categorical_positions(columns, [1.0]) == [2]
    assert categorical_positions(columns, [np.int64(1), "1"]) == [3, 4]
    assert categorical_positions(columns, [1]) == [0]


@pytest.mark.parametrize(
    ("columns", "declared"),
    [(["a", "b"], ["z"]), (["a", "a", "b"], ["a"])],
    ids=["no column", "two columns"],
)
def test_a_declaration_that_is_not_one_column_is_refused(
    columns: list[Any], declared: list[Any]
) -> None:
    with pytest.raises(LizyMLError) as caught:
        categorical_positions(columns, declared)
    assert caught.value.code == ErrorCode.DATA_SCHEMA_INVALID
    assert caught.value.context == {
        "column": declared[0],
        "reason": "categorical_feature_not_unique",
    }
