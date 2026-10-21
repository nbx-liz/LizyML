"""H-0136 acceptance (criteria 1-5, 7): a saved model behaves like the in-memory one.

Before anything is written, ``Model.export``, ``lizyml.persistence.export``
and ``export_code`` project every saved category state through joblib and
through the H-0133 codec JSON, and refuse with ``SERIALIZATION_FAILED`` when a
learned category becomes unseen, the projected transform raises, a code
changes or the mode's code changes. ``fit`` warns on the same rule.

The pytz cases need pandas 2 with pytz (the dependency floor; the
lowest-direct CI lane); pandas 3 does not depend on pytz.
"""

from __future__ import annotations

import datetime as dt
import warnings
import zoneinfo
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from tests._helpers import make_config
from tests.test_codegen._retrain_harness import generated_prediction

NY = zoneinfo.ZoneInfo("America/New_York")
FOLD1 = pd.Timestamp(dt.datetime(2024, 11, 3, 1, 30, fold=1, tzinfo=NY))
FIXED = pd.Timestamp("2024-11-03 01:30", tz=dt.timezone(dt.timedelta(hours=-5)))
EXITS = ("Model.export", "persistence.export", "export_code")


def _pytz_ny() -> Any:
    pytz = pytest.importorskip("pytz")
    return pytz.timezone("America/New_York")


def _pytz_standard() -> pd.Timestamp:
    """The standard-side 01:30 of 2024-11-03: pickle restores it with fold 1."""
    return pd.Timestamp(
        _pytz_ny().localize(dt.datetime(2024, 11, 3, 1, 30), is_dst=False)
    )


def _frame(
    values: list[Any],
    weights: list[int] | None = None,
    seed: int = 0,
    *,
    shuffle: bool = True,
) -> pd.DataFrame:
    """A column of *values* (each repeated per *weights*) beside strings; the
    category decides the target, so a wrong mapping changes predictions.

    ``shuffle=False`` keeps the values in blocks, in order: pandas keeps a
    mixed column's categories in the order they first appear, so every saved
    state (each fold's rows are a subset) learns them in the order given.
    The H-0136 verdict for values whose equality is not transitive depends
    on that order."""
    weights = weights or [120] * len(values)
    column = [v for v, w in zip(values, weights, strict=True) for _ in range(w)]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(column)) if shuffle else np.arange(len(column))
    column = [column[i] for i in order]
    codes = np.array([i for i, w in enumerate(weights) for _ in range(w)])[order]
    target = (codes % 2 == 0) ^ (rng.random(len(column)) < 0.1)
    return pd.DataFrame(
        {
            "c": pd.Series(column, dtype=object),
            "x": rng.normal(size=len(column)),
            "target": target.astype(int),
        }
    )


def _fit(df: pd.DataFrame, policy: str = "mode") -> tuple[Model, list[str]]:
    cfg = make_config("binary", n_estimators=8, n_splits=2)
    cfg["features"] = {"categorical": ["c"], "unseen_policy": policy}
    model = Model(cfg)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(data=df)
    return model, [str(w.message) for w in caught if "will refuse" in str(w.message)]


def _exits(model: Model, tmp_path: Path) -> dict[str, LizyMLError | Path]:
    from lizyml.persistence import export as persistence_export

    state = model._get_fit_state()
    actions = {
        "Model.export": lambda: model.export(tmp_path / "a"),
        "persistence.export": lambda: persistence_export(
            tmp_path / "b",
            state.fit_result,
            model._refit_result,
            state.cfg.model_dump(),
            state.cfg.task,
            declared_categories=state.declared_categories,
        ),
        "export_code": lambda: model.export_code(tmp_path / "c"),
    }
    targets = dict(zip(EXITS, ("a", "b", "c"), strict=True))
    out: dict[str, LizyMLError | Path] = {}
    for name, action in actions.items():
        try:
            action()
        except LizyMLError as exc:
            out[name] = exc
        else:
            out[name] = tmp_path / targets[name]
    return out


def _assert_refused(
    model: Model,
    tmp_path: Path,
    *,
    serializer: str,
    kind: str,
    exception: str | None = None,
) -> None:
    answers = _exits(model, tmp_path)
    for name in EXITS:
        exc = answers[name]
        assert isinstance(exc, LizyMLError), f"{name} accepted the model"
        assert exc.code == ErrorCode.SERIALIZATION_FAILED, name
        assert exc.context["serializer"] == serializer, (name, exc.context)
        assert exc.context["kind"] == kind, (name, exc.context)
        assert exc.context["column"] == "c"
        assert exc.context.get("exception") == exception, (name, exc.context)
    # Criterion 7: refused before anything is written.
    assert not any((tmp_path / p).exists() for p in ("a", "b", "c"))


def _assert_saved_the_same(model: Model, df: pd.DataFrame, tmp_path: Path) -> None:
    answers = _exits(model, tmp_path)
    for name in EXITS:
        assert isinstance(answers[name], Path), (name, answers[name])
    X = df.drop(columns=["target"])
    expected = model.predict(X)
    loaded = Model.load(tmp_path / "a").predict(X)
    np.testing.assert_array_equal(np.asarray(loaded.proba), np.asarray(expected.proba))
    assert loaded.warnings == expected.warnings
    np.testing.assert_allclose(
        generated_prediction(tmp_path / "c", X, "binary"),
        np.asarray(expected.proba),
        rtol=1e-7,
        atol=0.0,
    )


# --- Criterion 1: the reported pair ------------------------------------------


def test_the_372_pair_is_refused_at_every_exit_and_fit_warns(tmp_path: Path) -> None:
    model, warned = _fit(_frame([FOLD1, _pytz_standard(), "x"]))
    assert warned
    _assert_refused(model, tmp_path, serializer="joblib", kind="unseen")


# --- Criterion 8: the H-0133 case this rule refuses ---------------------------


def test_the_h0133_pytz_fold_case_is_refused_and_fit_warns(tmp_path: Path) -> None:
    """Both pytz occurrences of 01:30 on 2024-11-03, the H-0133 acceptance
    case ``Timestamp pytz fold`` in that test's own frame. pandas 2 rebuilds
    the standard side with fold 1 inside fit (#379), so the in-memory state
    holds that value and the codec JSON gives back fold 0: the JSON side
    breaks."""
    from tests.test_persistence.test_h0133_acceptance import _frame as h0133_frame

    ny = _pytz_ny()
    values = [
        pd.Timestamp(ny.localize(dt.datetime(2024, 11, 3, 1, 30), is_dst=d))
        for d in (True, False)
    ]
    model, warned = _fit(h0133_frame(values))
    assert warned
    _assert_refused(model, tmp_path, serializer="json", kind="unseen")


# --- Criterion 2: one pytz value, not the mode -------------------------------


@pytest.mark.parametrize("policy", ["mode", "nan", "error"])
def test_one_pytz_value_is_refused_under_every_policy(
    policy: str, tmp_path: Path
) -> None:
    model, warned = _fit(_frame([_pytz_standard(), "x", "y"], [20, 60, 40]), policy)
    assert warned
    _assert_refused(model, tmp_path, serializer="joblib", kind="unseen")


@pytest.mark.parametrize("policy", ["mode", "nan", "error"])
def test_an_unambiguous_pytz_value_saves_and_predicts_the_same(
    policy: str, tmp_path: Path
) -> None:
    summer = pd.Timestamp(_pytz_ny().localize(dt.datetime(2024, 6, 1, 1, 30)))
    df = _frame([summer, "x", "y"], [20, 60, 40])
    model, warned = _fit(df, policy)
    assert not warned
    _assert_saved_the_same(model, df, tmp_path)


# --- Criterion 3: the value inside containers --------------------------------


@pytest.mark.parametrize("container", ["tuple", "frozenset"])
def test_a_pytz_value_inside_a_container_is_refused(
    container: str, tmp_path: Path
) -> None:
    value = _pytz_standard()
    held = (value, 1) if container == "tuple" else frozenset({value, 1})
    model, warned = _fit(_frame([held, "x"]))
    assert warned
    _assert_refused(model, tmp_path, serializer="joblib", kind="unseen")


# --- Criterion 4: the codegen JSON collision ---------------------------------


@pytest.mark.parametrize(
    ("order", "kind", "exception"),
    [
        # JSON turns np.uint64(4) into 4; 4 == np.timedelta64(4, 'M') and both
        # hash to 4, but np.uint64(4) != np.timedelta64(4, 'M').
        pytest.param("uint64 first", "raises", "ValueError", id="not unique"),
        pytest.param("timedelta64 first", "unseen", None, id="uint64 unseen"),
    ],
)
def test_uint64_beside_timedelta64_months_is_refused(
    order: str, kind: str, exception: str | None, tmp_path: Path
) -> None:
    pair = [np.uint64(4), np.timedelta64(4, "M")]
    values = pair if order == "uint64 first" else pair[::-1]
    model, warned = _fit(_frame([*values, "x"], shuffle=False))
    assert warned
    _assert_refused(model, tmp_path, serializer="json", kind=kind, exception=exception)


def test_the_pair_as_declared_categories_is_refused(tmp_path: Path) -> None:
    values = [np.uint64(4), np.timedelta64(4, "M"), "x"]
    df = _frame(values)
    df["c"] = df["c"].astype(pd.CategoricalDtype(values))
    model, warned = _fit(df)
    assert warned
    answers = _exits(model, tmp_path)
    for name in EXITS:
        exc = answers[name]
        assert isinstance(exc, LizyMLError), name
        assert exc.code == ErrorCode.SERIALIZATION_FAILED
        assert exc.context["serializer"] == "json"
    assert not any((tmp_path / p).exists() for p in ("a", "b", "c"))


def test_a_large_uint64_declared_before_numpy_true_is_saved_by_key(
    tmp_path: Path,
) -> None:
    """In this order pandas' ``set_categories`` may raise comparing the
    JSON-projected categories (``np.uint64`` -> ``int``, ``np.bool_`` ->
    ``bool``). Since H-0138 the column is looked up by key, which the JSON
    copy keeps, so the model is saved and answers the same on every pandas
    (H-0136 refused it where pandas raised)."""
    values = [np.uint64(2**63 + 5), np.bool_(True), "x"]
    df = _frame(values, shuffle=False)
    df["c"] = df["c"].astype(pd.CategoricalDtype(values))
    model, warned = _fit(df)
    encoder = model._refit_result.pipeline_state["encoder"]  # noqa: SLF001
    assert "c" in encoder["identity_columns"]
    assert not warned
    _assert_saved_the_same(model, df, tmp_path)


def test_numpy_true_first_keeps_its_codes(tmp_path: Path) -> None:
    """The same values in pandas' own order answer the same after JSON."""
    df = _frame([np.uint64(2**63 + 5), np.bool_(True), "x"], shuffle=False)
    model, warned = _fit(df)
    assert not warned
    _assert_saved_the_same(model, df, tmp_path)


def test_a_large_uint64_beside_python_true_is_not_refused(tmp_path: Path) -> None:
    df = _frame([np.uint64(2**63 + 5), True, "x"], shuffle=False)
    model, warned = _fit(df)
    assert not warned
    _assert_saved_the_same(model, df, tmp_path)


# --- Criterion 5: equal-but-hash-apart pairs that keep their codes -----------


@pytest.mark.parametrize(
    "pair",
    [
        pytest.param([FOLD1, FIXED], id="zoneinfo fold 1 + fixed offset"),
        pytest.param([1, np.timedelta64(1, "s")], id="1 + timedelta64(1, s)"),
    ],
)
def test_stable_pairs_save_and_predict_the_same(
    pair: list[Any], tmp_path: Path
) -> None:
    df = _frame([*pair, "x"])
    model, warned = _fit(df)
    assert not warned
    _assert_saved_the_same(model, df, tmp_path)


def test_the_refusal_message_names_both_surfaces(tmp_path: Path) -> None:
    model, _ = _fit(_frame([np.uint64(4), np.timedelta64(4, "M"), "x"]))
    with pytest.raises(LizyMLError) as info:
        model.export(tmp_path / "a")
    message = info.value.user_message
    assert "Model.export" in message and "export_code" in message
    assert "'c'" in message
