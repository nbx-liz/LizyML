"""Feature names LightGBM refuses, and duplicate column names, are refused
before fit and tune train (H-0148, #409).

LightGBM 4.6.0 ``Dataset::set_feature_names`` (include/LightGBM/dataset.h)
refuses a name holding one of ``" , : [ ] { }`` and a name that repeats an
earlier one once spaces become ``_``; the Python package hands it the
``str()`` of each pandas column name as a C string, so a NUL cuts the name.
Before H-0148 ``Model.fit`` raised that ``LightGBMError`` uncoded and
``Model.tune`` failed every trial into ``TUNING_FAILED``; a duplicate column
name crashed inside LizyML with ``AttributeError``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import pytest

from lizyml.core._feature_name_check import check_feature_names_for_training
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.model import Model
from lizyml.estimators.lgbm.feature_names import lightgbm_refused_feature_name
from lizyml.estimators.lgbm.provider import LGBMProvider
from lizyml.estimators.provider import FeatureNameRuleProvider, RefusedFeatureName
from tests._helpers import make_config

N = 120
JSON = '",:[]{}'
JSON_REASON = "json_special_character_in_feature_name"
DUPLICATE_REASON = "duplicate_feature_name"

#: Name pairs that collide once LightGBM rewrites them (str(), NUL, spaces).
COLLISIONS: list[tuple[Any, Any, str]] = [
    ("a b", "a_b", "a_b"),
    ("a  b", "a__b", "a__b"),
    (0, "0", "0"),
    (1.5, "1.5", "1.5"),
    ("a\x00b", "a", "a"),
]

#: Names LightGBM accepts.
ADMITTED: list[Any] = ["z", "a'b", "a\\b", "a;b", "a b"]


def _frame(names: list[Any]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(N, len(names))))
    df.columns = pd.Index(names, dtype=object)
    df["target"] = (rng.normal(size=N) > 0).astype(int)
    return df


def _config(**extra: Any) -> dict[str, Any]:
    raw = make_config("binary", n_estimators=5)
    raw.update(extra)
    return raw


def _tune_config() -> dict[str, Any]:
    return _config(tuning={"optuna": {"params": {"n_trials": 2}, "space": {}}})


def _fit(df: pd.DataFrame, **extra: Any) -> Model:
    model = Model(_config(**extra))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data=df)
    return model


def _refusal(call: Any) -> LizyMLError:
    with pytest.raises(LizyMLError) as exc, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        call()
    assert exc.value.code == ErrorCode.DATA_SCHEMA_INVALID
    return exc.value


# --- Acceptance 1: names LightGBM refuses, in fit and tune -------------------


@pytest.mark.parametrize("char", list(JSON))
def test_fit_and_tune_refuse_a_json_special_character(char: str) -> None:
    name = f"a{char}b"
    df = _frame(["x", name])
    expected = {
        "column": name,
        "position": 1,
        "estimator_name": name,
        "reason": JSON_REASON,
    }
    assert _refusal(lambda: Model(_config()).fit(data=df)).context == expected
    assert _refusal(lambda: Model(_tune_config()).tune(data=df)).context == expected


@pytest.mark.parametrize(("first", "second", "seen"), COLLISIONS, ids=repr)
def test_fit_and_tune_refuse_a_collision(first: Any, second: Any, seen: str) -> None:
    df = _frame(["x", first, second])
    expected = {
        "column": second,
        "position": 2,
        "estimator_name": seen,
        "reason": DUPLICATE_REASON,
        "first_column": first,
        "first_position": 1,
    }
    assert _refusal(lambda: Model(_config()).fit(data=df)).context == expected
    assert _refusal(lambda: Model(_tune_config()).tune(data=df)).context == expected


def test_tune_with_storage_writes_no_study(tmp_path: Path) -> None:
    storage = f"sqlite:///{tmp_path / 'optuna.db'}"
    df = _frame(["x", "a,b"])
    _refusal(
        lambda: Model(_tune_config()).tune(data=df, storage=storage, study_name="s")
    )
    if (tmp_path / "optuna.db").exists():
        assert optuna.study.get_all_study_summaries(storage) == []


# --- Acceptance 2: duplicate column names ------------------------------------


def _role_frame(columns: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    values = {
        "x": rng.normal(size=N),
        "target": (rng.normal(size=N) > 0).astype(int),
        "g": np.repeat(np.arange(12), N // 12),
        "t": np.arange(N),
        "a": rng.normal(size=N),
    }
    df = pd.DataFrame({i: values[c] for i, c in enumerate(columns)})
    df.columns = pd.Index(columns, dtype=object)
    return df


KFOLD = {"method": "kfold", "n_splits": 3}
DUPLICATES: dict[str, tuple[list[str], dict[str, Any], str]] = {
    "feature": (["x", "a", "a", "target"], {}, "feature"),
    "target": (["x", "target", "target"], {}, "data.target"),
    "target, excluded": (
        ["x", "target", "target"],
        {"features": {"exclude": ["target"]}},
        "data.target",
    ),
    "group_col, kfold": (
        ["x", "g", "g", "target"],
        {"data": {"target": "target", "group_col": "g"}},
        "data.group_col",
    ),
    "group_col, group_kfold": (
        ["x", "g", "g", "target"],
        {
            "data": {"target": "target", "group_col": "g"},
            "split": {"method": "group_kfold", "n_splits": 3},
        },
        "data.group_col",
    ),
    "time_col, kfold": (
        ["x", "t", "t", "target"],
        {"data": {"target": "target", "time_col": "t"}},
        "data.time_col",
    ),
    "time_col, time_series": (
        ["x", "t", "t", "target"],
        {
            "data": {"target": "target", "time_col": "t"},
            "split": {"method": "time_series", "n_splits": 3},
        },
        "data.time_col",
    ),
}


@pytest.mark.parametrize("case", sorted(DUPLICATES))
def test_fit_and_tune_refuse_a_duplicate_column(case: str) -> None:
    columns, extra, role = DUPLICATES[case]
    df = _role_frame(columns)
    label = columns[1]
    positions = [i for i, c in enumerate(columns) if c == label]
    expected = {
        "column": label,
        "positions": positions,
        "role": role,
        "reason": "duplicate_column_name",
    }
    split = extra.get("split", KFOLD)
    fit_cfg = {**_config(), **extra, "split": split}
    tune_cfg = {**_tune_config(), **extra, "split": split}
    assert _refusal(lambda: Model(fit_cfg).fit(data=df)).context == expected
    assert _refusal(lambda: Model(tune_cfg).tune(data=df)).context == expected


@pytest.mark.parametrize("label", [np.nan, 0, 1.5, True], ids=repr)
def test_a_repeated_non_string_label_is_refused(label: Any) -> None:
    """Grouped as pandas groups labels: two NaN labels are one repeated label
    (implementation review round 1; ``nan == nan`` is false)."""
    df = _frame(["x", label, label])
    for call in (
        lambda: Model(_config()).fit(data=df),
        lambda: Model(_tune_config()).tune(data=df),
    ):
        context = _refusal(call).context
        assert context["positions"] == [1, 2]
        assert context["role"] == "feature"
        assert context["reason"] == "duplicate_column_name"
        column = context["column"]
        if isinstance(label, float) and np.isnan(label):
            assert isinstance(column, float) and np.isnan(column)
        else:
            assert column == label


#: Column labels, the reason fit refuses them with, and the refused positions.
#: Equal labels group as pandas hashes them; every missing label (None, NaN)
#: is one label, because ``select_training_columns`` (``df[list(columns)]``)
#: turns None into NaN before LightGBM sees the names (owner ruling,
#: implementation review round 2; ``missing-labels.out``).
LABEL_CASES: list[tuple[list[Any], str, list[int]]] = [
    (["x", 0, False], "duplicate_column_name", [1, 2]),
    (["x", 1, 1.0], "duplicate_column_name", [1, 2]),
    (["x", ("a", 1), ("a", 1)], "duplicate_column_name", [1, 2]),
    (["x", "0", 0], "duplicate_feature_name", [2]),
    (["x", "a\x00b", "a"], "duplicate_feature_name", [2]),
]


@pytest.mark.parametrize(("columns", "reason", "positions"), LABEL_CASES, ids=repr)
def test_repeated_labels_are_grouped(
    columns: list[Any], reason: str, positions: list[int]
) -> None:
    context = _refusal(lambda: Model(_config()).fit(data=_frame(columns))).context
    assert context["reason"] == reason
    if reason == "duplicate_column_name":
        assert context["positions"] == positions
    else:
        assert context["position"] == positions[0]


def _merged_by_selection(features: list[Any]) -> bool:
    """Whether the running pandas's list selection (what
    ``select_training_columns`` does) gives two of *features* one label."""
    X = pd.DataFrame(np.zeros((1, len(features))))
    X.columns = pd.Index(features, dtype=object)
    picked = X[list(X.columns)]
    return len({str(c) for c in picked.columns}) < len(features)


@pytest.mark.parametrize(
    "features",
    [["x", None, np.nan], [None, np.nan], [0, None, np.nan]],
    ids=repr,
)
def test_none_beside_nan_follows_the_running_pandas(features: list[Any]) -> None:
    """None beside NaN is refused exactly when the running pandas's list
    selection merges them before LightGBM (pandas 3.0 with a string label
    beside them; pandas 2.x never), as ``duplicate_feature_name`` on the
    selected names (owner ruling after PR #420's CI). Otherwise it trains."""
    df = _frame(features)
    if _merged_by_selection(features):
        context = _refusal(lambda: Model(_config()).fit(data=df)).context
        assert context["reason"] == DUPLICATE_REASON
        assert context["column"] is features[-1] or np.isnan(context["column"])
        assert context["first_column"] is None
    else:
        assert _fit(df)._refit_result is not None


def test_a_repeated_none_label_is_refused_everywhere() -> None:
    """Two None labels are one repeated label on every pandas."""
    df = _frame(["x", None, None])
    context = _refusal(lambda: Model(_config()).fit(data=df)).context
    assert context["reason"] == "duplicate_column_name"
    assert context["positions"] == [1, 2]


def test_missing_labels_beside_an_excluded_duplicate() -> None:
    """The reviewer's case: ``a`` repeats and is excluded; None and NaN are
    distinct column labels, so the duplicate check passes, and training
    meets them as the running pandas selects them."""
    df = _frame([None, np.nan, "a", "a"])
    config = _config(features={"exclude": ["a"]})
    if _merged_by_selection([None, np.nan]):
        assert _refusal(lambda: Model(config).fit(data=df)).context["reason"] == (
            DUPLICATE_REASON
        )
    else:
        assert _fit(df, features={"exclude": ["a"]})._refit_result is not None


def test_the_selected_names_match_lightgbm() -> None:
    """The names the rule judges are the ones LightGBM meets: after the list
    selection, LightGBM refuses None beside NaN exactly when they merged."""
    features = ["x", None, np.nan]
    X = pd.DataFrame(np.zeros((40, 3)))
    X.columns = pd.Index(features, dtype=object)
    picked = X[list(X.columns)]
    assert _lightgbm_refuses(list(picked.columns)) is _merged_by_selection(features)


def test_an_excluded_duplicate_feature_still_fits() -> None:
    model = _fit(_role_frame(["x", "a", "a", "target"]), features={"exclude": ["a"]})
    assert model._refit_result is not None


# --- Acceptance 3: admitted names --------------------------------------------


@pytest.mark.parametrize("name", ADMITTED, ids=repr)
def test_admitted_names_fit(name: Any) -> None:
    model = _fit(_frame(["x", name]))
    assert model._refit_result is not None


def test_admitted_names_tune() -> None:
    df = _frame(["x", "a'b", "a b"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = Model(_tune_config()).tune(data=df)
    assert result is not None


# --- Acceptance 4: the rule against the installed LightGBM -------------------

RULE_LISTS: list[list[Any]] = [
    ["x", "z"],
    *[["x", f"a{c}b"] for c in JSON],
    *[[f"{c}b", "x"] for c in JSON],
    *[["x", a, b] for a, b, _ in COLLISIONS],
    *[[b, "x", a] for a, b, _ in COLLISIONS],
    ["x", "a'b", "a\\b", "a;b"],
    ["x", "a b"],
    ["x", "a b", "c d"],
    ["x", " ", "_"],
    ["x", "a\x00", "a\x00b"],
    ["x", 1, 2],
    ["x", True, "True"],
]


def _lightgbm_refuses(names: list[Any]) -> bool:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, len(names))))
    X.columns = pd.Index(names, dtype=object)
    y = (rng.normal(size=40) > 0).astype(int)
    try:
        lgb.Dataset(X, y, params={"verbose": -1}).construct()
    except lgb.basic.LightGBMError:
        return True
    return False


@pytest.mark.parametrize("names", RULE_LISTS, ids=repr)
def test_the_rule_matches_lightgbm(names: list[Any]) -> None:
    """Refused exactly when LightGBM's Dataset refuses the names. A LightGBM
    that changes its rule fails this test, the cue to revisit it (DC7)."""
    refused = lightgbm_refused_feature_name(names) is not None
    assert refused is _lightgbm_refuses(names)


def test_the_rule_answers_the_first_refused_name_in_lightgbm_order() -> None:
    assert lightgbm_refused_feature_name(["a b", "a_b", "c,d"]) == RefusedFeatureName(
        position=1,
        name="a_b",
        estimator_name="a_b",
        reason=DUPLICATE_REASON,
        first_position=0,
    )
    assert lightgbm_refused_feature_name(["c,d", "a b", "a_b"]) == RefusedFeatureName(
        position=0,
        name="c,d",
        estimator_name="c,d",
        reason=JSON_REASON,
        first_position=None,
    )


# --- Acceptance 5: the capability --------------------------------------------


def test_the_lightgbm_provider_has_the_capability() -> None:
    assert isinstance(LGBMProvider(), FeatureNameRuleProvider)


def test_a_provider_without_the_capability_checks_nothing() -> None:
    check_feature_names_for_training(
        object(), _frame(["x", "a,b"]).drop(columns="target")
    )
