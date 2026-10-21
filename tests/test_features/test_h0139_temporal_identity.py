"""H-0139 acceptance 3-9: temporal categories through fit, predict and save (#379).

- 3: a numpy timedelta64 in Y / M units is refused at fit (builder, encoder,
  generated ``train.py``) and is unseen at predict under every policy;
- 4: #379 itself (pandas 2 with pytz): the model no longer calls its own
  training rows unseen, and every exit saves it;
- 5 / 6: a temporal column pandas cannot hold by instant is rebuilt by key,
  named by the builder only, learned per fold, and reaches LightGBM as
  positions;
- 7: the runtime and the generated scripts give the same codes;
- 8: a saved state keeps the lookup its ``identity_columns`` names;
- 9: declarations H-0136 refused only because pandas compared them after JSON
  are saved.
"""

from __future__ import annotations

import datetime as dt
import json
import warnings
import zoneinfo
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml import Model
from lizyml.core.category_values import IDENTITY_OBJECT_COLUMNS, category_identity_key
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.specs.feature_spec import FeatureSpec
from lizyml.data.dataframe_builder import _apply_categorical
from lizyml.features.encoders import categorical_encoder
from lizyml.features.encoders.categorical_encoder import CategoricalEncoder
from lizyml.features.pipelines_native import NativeFeaturePipeline
from tests._helpers import make_config
from tests.test_codegen.test_h0133_codec_copy import COPY
from tests.test_codegen.test_h0138_identity_copy import (
    _as_ints,
    _batches,
    _modules,
    _project,
    _runtime_codes,
)
from tests.test_features.test_h0138_identity import _model_on, _refit

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

NY = zoneinfo.ZoneInfo("America/New_York")
FOLD0 = dt.datetime(2024, 11, 3, 1, 30, fold=0, tzinfo=NY)
FOLD1 = FOLD0.replace(fold=1)
#: pandas (and Python's ==) take the two folds of one zone for one value;
#: they are two instants, so the column is rebuilt by key on every lane.
FOLD_PAIR = [FOLD0, FOLD1]
SUB_NS = [np.datetime64(4, "ps"), np.datetime64(4, "fs"), np.datetime64(4, "as")]
CALENDAR = {"Y": np.timedelta64(2, "Y"), "M": np.timedelta64(3, "M")}


def _pytz() -> Any:
    try:
        import pytz  # type: ignore[import-untyped, unused-ignore]
    except ImportError:
        return None
    return pytz


def _pytz_pair() -> list[pd.Timestamp]:
    pytz = _pytz()
    if pytz is None:
        pytest.skip("pytz is not installed (pandas 3 does not depend on it)")
    ny = pytz.timezone("America/New_York")
    return [
        pd.Timestamp(ny.localize(dt.datetime(2024, 11, 3, 1, 30), is_dst=d))
        for d in (True, False)
    ]


def test_a_pandas_2_lane_has_pytz() -> None:
    """#379 lives on pandas 2; its tests must run there, never skip."""
    if int(pd.__version__.split(".")[0]) < 3:
        assert _pytz() is not None


# --- acceptance 3: Y / M durations ------------------------------------------------


def _assert_calendar_refusal(caught: Any, units: list[str]) -> None:
    assert caught.value.code == ErrorCode.DATA_SCHEMA_INVALID
    assert caught.value.context == {
        "column": "c",
        "units": units,
        "reason": "calendar_timedelta64",
    }
    assert "calendar-unit numpy timedelta64" in caught.value.user_message


@pytest.mark.parametrize(
    ("values", "units"),
    [
        ([CALENDAR["Y"], "x"], ["Y"]),
        ([("x", (CALENDAR["M"],)), "y"], ["M"]),
        ([frozenset({CALENDAR["Y"], 1}), CALENDAR["M"]], ["M", "Y"]),
        ([np.timedelta64(3, "2M"), "x"], ["M"]),
    ],
    ids=["row", "nested tuple", "frozenset and row", "multiple"],
)
def test_the_builder_and_the_encoder_refuse_a_calendar_duration(
    values: list[Any], units: list[str]
) -> None:
    df = pd.DataFrame({"c": pd.Series(values * 3, dtype=object)})
    with pytest.raises(LizyMLError) as caught:
        _apply_categorical(df, FeatureSpec(categorical=("c",)))
    _assert_calendar_refusal(caught, units)
    with pytest.raises(LizyMLError) as caught:
        CategoricalEncoder().fit(df, ["c"])
    _assert_calendar_refusal(caught, units)
    with pytest.raises(ValueError, match=r"calendar-unit numpy timedelta64"):
        COPY._check_calendar_durations(COPY._column_values(df["c"]), "c")


def test_an_unused_declared_calendar_duration_is_refused() -> None:
    declared = pd.CategoricalDtype(["x", "y", CALENDAR["M"]])
    df = pd.DataFrame({"c": pd.Series(["x", "y"] * 3, dtype=object).astype(declared)})
    with pytest.raises(LizyMLError) as caught:
        _apply_categorical(df, FeatureSpec(categorical=("c",)))
    _assert_calendar_refusal(caught, ["M"])
    with pytest.raises(LizyMLError) as caught:
        CategoricalEncoder().fit(df, ["c"])
    _assert_calendar_refusal(caught, ["M"])


def test_model_fit_refuses_a_calendar_duration() -> None:
    rng = np.random.default_rng(0)
    rows = [CALENDAR["Y"], "x"] * 20
    df = pd.DataFrame(
        {"c": pd.Series(rows, dtype=object), "num": rng.normal(size=len(rows))}
    )
    df["target"] = rng.normal(size=len(rows))
    with pytest.raises(LizyMLError) as caught:
        Model(make_config("regression", n_estimators=5, n_splits=2)).fit(data=df)
    _assert_calendar_refusal(caught, ["Y"])


@pytest.mark.parametrize("policy", ["mode", "nan", "error"])
def test_a_calendar_duration_at_predict_is_unseen(policy: str) -> None:
    train = pd.DataFrame({"c": pd.Series(["x", "y", "x"] * 4, dtype=object)})
    encoder = CategoricalEncoder(unseen_policy=policy).fit(train, ["c"])  # type: ignore[arg-type]
    assert encoder.get_state()["identity_columns"] == ["c"]
    batch = pd.Series([CALENDAR["Y"], (CALENDAR["M"],), None, "y"], dtype=object)
    unchanged = pd.Series(False, index=batch.index)
    cats = encoder.get_state()["categories"]["c"]
    mode = encoder.get_state()["modes"]["c"]
    if policy == "error":
        with pytest.raises(LizyMLError, match="unseen"):
            encoder.transform(pd.DataFrame({"c": batch}))
        with pytest.raises(ValueError, match="unseen"):
            COPY._keyed_codes(batch, batch, unchanged, cats, mode, policy, "c")
        return
    runtime = [
        int(c) for c in encoder.transform(pd.DataFrame({"c": batch}))["c"].cat.codes
    ]
    mode_code = cats.index(mode)
    expected = [mode_code, mode_code, -1, 1] if policy == "mode" else [-1, -1, -1, 1]
    assert runtime == expected
    copied, _ = COPY._keyed_codes(batch, batch, unchanged, cats, mode, policy, "c")
    assert _as_ints(copied) == expected


# --- acceptance 4: #379 -------------------------------------------------------------


def _h0133_model(values: list[Any]) -> tuple[Model, pd.DataFrame]:
    from tests.test_persistence.test_h0133_acceptance import _frame

    df = _frame(values)
    cfg = make_config("binary", n_estimators=8, n_splits=2)
    cfg["features"] = {"categorical": ["c"]}
    model = Model(cfg)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(data=df)
    assert not [w for w in caught if "would not answer like" in str(w.message)]
    return model, df


def test_379_the_model_does_not_call_its_own_training_rows_unseen() -> None:
    model, df = _h0133_model(_pytz_pair())
    X = df.drop(columns=["target"])
    result = model.predict(X)
    assert not [w for w in result.warnings if "unseen" in w]
    full = np.asarray(result.proba)
    for i in range(2):  # each value alone answers as inside the full batch
        alone = np.asarray(model.predict(X.iloc[[i]].reset_index(drop=True)).proba)
        assert alone[0] == full[i]


def test_379_every_exit_saves_a_model_that_answers_the_same(tmp_path: Path) -> None:
    from tests.test_codegen._retrain_harness import generated_prediction

    model, df = _h0133_model(_pytz_pair())
    X = df.drop(columns=["target"])
    model.export(tmp_path / "a")
    model.export_code(tmp_path / "c")
    expected = np.asarray(model.predict(X).proba)
    loaded = Model.load(tmp_path / "a").predict(X)
    np.testing.assert_array_equal(np.asarray(loaded.proba), expected)
    np.testing.assert_allclose(
        generated_prediction(tmp_path / "c", X, "binary"), expected, rtol=1e-7, atol=0
    )


# --- acceptance 5 / 6: rebuilt columns, the mark and the folds ---------------------


@pytest.mark.parametrize(
    "rows",
    [[*FOLD_PAIR, "x"], SUB_NS],
    ids=["fold pair beside a str", "sub-ns only"],
)
def test_the_builder_names_a_temporal_column_pandas_cannot_hold(
    rows: list[Any],
) -> None:
    built = _apply_categorical(
        pd.DataFrame({"c": pd.Series(rows * 3, dtype=object)}),
        FeatureSpec(categorical=("c",)),
    )
    assert built["c"].dtype == object
    assert built.attrs[IDENTITY_OBJECT_COLUMNS] == ["c"]


def test_a_rebuilt_temporal_column_reaches_lightgbm_as_positions() -> None:
    values = FOLD_PAIR
    model, _ = _model_on(values)
    encoder = _refit(model).pipeline_state["encoder"]
    cats = encoder["categories"]["c"]
    assert "c" in encoder["identity_columns"]
    assert "c" in encoder["stand_in_columns"]
    assert len({category_identity_key(v) for v in cats}) == len(values) + 1
    booster = _refit(model).model.get_native_model()
    assert booster.pandas_categorical == [list(range(len(cats)))]


@pytest.mark.parametrize(
    "values",
    [
        [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
        [*SUB_NS, "x"],
        [pd.Timestamp(FOLD0), pd.Timestamp(FOLD1)],
    ],
    ids=["Timestamps", "sub-ns beside a str", "Timestamp folds"],
)
def test_a_faithful_temporal_column_keeps_its_pandas_categories(
    values: list[Any],
) -> None:
    """Looked up by key, learned as pandas learns it today. Whether LightGBM
    takes positions is the estimator's rule (H-0133: it cannot write these
    values as JSON), not a new temporal rule."""
    rows = pd.Series([*values, "x"] * 3, dtype=object)
    built = _apply_categorical(pd.DataFrame({"c": rows}), FeatureSpec())
    assert IDENTITY_OBJECT_COLUMNS not in built.attrs
    today = list(rows.astype("category").cat.categories)
    model, _ = _model_on(values)
    encoder = _refit(model).pipeline_state["encoder"]
    assert "c" in encoder["identity_columns"]
    assert [category_identity_key(v) for v in encoder["categories"]["c"]] == [
        category_identity_key(v) for v in today
    ]


def test_a_temporal_key_seen_only_in_validation_rows_is_not_learned() -> None:
    """The mark names columns only: a fold fit learns from its own rows."""
    df = pd.DataFrame(
        {"c": pd.Series([*FOLD_PAIR * 3, pd.Timestamp("2030-01-01")], dtype=object)}
    )
    X = _apply_categorical(df, FeatureSpec(categorical=("c",)))
    assert X.attrs[IDENTITY_OBJECT_COLUMNS] == ["c"]
    train = X.iloc[:-1]
    pipeline = NativeFeaturePipeline(unseen_policy="error").fit(
        train, pd.Series(range(len(train)))
    )  # type: ignore[arg-type]
    learned = pipeline.get_state()["encoder"]["categories"]["c"]
    assert pd.Timestamp("2030-01-01") not in learned
    with pytest.raises(LizyMLError, match="unseen"):
        pipeline.transform(X.iloc[-1:])


# --- acceptance 7: runtime = train.py = predict.py ---------------------------------

TEMPORAL_COLUMNS = [
    FOLD_PAIR,
    SUB_NS,
    [
        pd.Timestamp("2024-01-02 08:00", tz="UTC"),
        pd.Timestamp("2024-01-02 03:00", tz=NY),
    ],
    [pd.Timedelta(hours=1), dt.timedelta(hours=1), np.timedelta64(90, "m")],
    [dt.date(2024, 1, 2), pd.Timestamp("2024-01-02")],
    [(pd.Timestamp(FOLD0), 1), (pd.Timestamp(FOLD1), 1)],
]


@pytest.mark.parametrize("values", TEMPORAL_COLUMNS, ids=repr)
def test_runtime_train_and_predict_give_the_same_temporal_codes(
    values: list[Any], tmp_path: Path
) -> None:
    model, df, project = _project(values, tmp_path)
    written = json.loads((project / "artifacts" / "pipeline_state.json").read_text())
    assert written["identity_columns"] == ["c"]
    train, predict = _modules(project)
    batches = _batches(df)
    runtime = [_runtime_codes(model, b) for b in batches]
    assert [_as_ints(predict.transform(b)["c"]) for b in batches] == runtime
    assert runtime[0] == runtime[1][::-1]
    raw = train.cast_categoricals(df[train.CFG["feature_names"]])
    train.fit_pipeline(raw)
    retrained = json.loads((project / "artifacts" / "pipeline_state.json").read_text())
    assert retrained == written


def test_runtime_and_scripts_agree_on_the_379_pair(tmp_path: Path) -> None:
    test_runtime_train_and_predict_give_the_same_temporal_codes(_pytz_pair(), tmp_path)


# --- acceptance 8: the saved route decides ------------------------------------------


def test_a_state_whose_list_omits_a_temporal_column_keeps_the_pandas_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A develop (H-0138) state: version 1, temporal columns not listed."""
    values = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"), "x"]
    train = pd.DataFrame({"c": pd.Series(values * 4, dtype=object).astype("category")})
    state = CategoricalEncoder().fit(train, ["c"]).get_state()
    assert state["identity_columns"] == ["c"]
    old = {**state, "identity_columns": []}
    keyed: list[Any] = []
    real = categorical_encoder.keyed_codes

    def spy(*args: Any, **kwargs: Any) -> Any:
        keyed.append(args[-1])
        return real(*args, **kwargs)

    monkeypatch.setattr(categorical_encoder, "keyed_codes", spy)
    batch = pd.DataFrame({"c": pd.Series(values, dtype=object)})
    restored = CategoricalEncoder().load_state(old)
    assert restored.get_state()["identity_columns"] == []
    assert restored.get_state()["category_identity_version"] == 1
    restored.transform(batch)
    assert keyed == []
    CategoricalEncoder().load_state(state).transform(batch)
    assert keyed == ["c"]


# --- acceptance 9: declarations pandas refused after JSON ---------------------------

BIG = np.uint64(9223372036854775813)
DECLARED_TEMPORAL = [
    *(np.timedelta64(4, u) for u in ("W", "D", "h", "m", "s", "ms", "us")),
    np.timedelta64(3, "15m"),
]


@pytest.mark.parametrize("value", DECLARED_TEMPORAL, ids=repr)
def test_a_large_uint64_declared_beside_a_duration_is_saved(
    value: Any, tmp_path: Path
) -> None:
    """The eight pandas-3 declarations (``p61-92``..``p61-98``, ``p61-103``)."""
    rng = np.random.default_rng(0)
    declared = pd.CategoricalDtype([BIG, value, "x"])
    rows = pd.Series([BIG, value, "x"] * 20, dtype=object).astype(declared)
    df = pd.DataFrame({"c": rows, "num": rng.normal(size=len(rows))})
    df["target"] = rng.normal(size=len(rows))
    model = Model(make_config("regression", n_estimators=5, n_splits=2))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(data=df)
    assert not [w for w in caught if "would not answer like" in str(w.message)]
    model.export(tmp_path / "a")
    X = df.drop(columns=["target"])
    np.testing.assert_array_equal(
        Model.load(tmp_path / "a").predict(X).pred, model.predict(X).pred
    )
