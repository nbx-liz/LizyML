"""H-0125 policy 4 (#313): a LightGBM native metric takes no LizyML parameters.

``model.params["metric"]`` accepts ``{name: {param: value}}`` (H-0065). For a
native LightGBM name the parameters used to be dropped in silence:
``{"huber": {"delta": 9.0}}`` trained LightGBM's ``huber`` with its own
``alpha`` and ``delta`` did nothing. Non-empty parameters on a native name are
refused (criterion 5); an artifact that recorded them still loads and predicts,
and only what would train from the record again is refused (criterion 6).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.model import Model
from lizyml.estimators.lgbm import metric_bridge
from lizyml.estimators.lgbm.adapter import LGBMAdapter
from lizyml.estimators.lgbm.metric_bridge import resolve_metrics
from tests._helpers import make_binary_df, make_config, make_regression_df
from tests._train_spy import record_lightgbm_calls

CI = ErrorCode.CONFIG_INVALID

#: ``(task, entry, metric as written, sorted parameter names)``.
REFUSED: list[tuple[str, dict[str, Any], str, list[str]]] = [
    ("regression", {"huber": {"delta": 9.0}}, "huber", ["delta"]),
    ("regression", {"rmse": {"bogus": 1}}, "rmse", ["bogus"]),
    ("regression", {"l2": {"b": 1, "a": 2}}, "l2", ["a", "b"]),
    # A LizyML name translated to a native one is native too.
    ("binary", {"logloss": {"eps": 1e-9}}, "logloss", ["eps"]),
    ("binary", {"auc": {"k": 3}}, "auc", ["k"]),
    ("multiclass", {"multi_logloss": {"x": 1}}, "multi_logloss", ["x"]),
]


def _assert_native_params_refused(
    error: LizyMLError, metric: str, params: list[str]
) -> None:
    assert error.code == CI
    assert error.context["metric"] == metric
    assert error.context["params"] == params
    message = error.user_message
    assert "native" in message
    assert "alpha" in message
    assert "evaluation.metrics" in message


@pytest.mark.parametrize(("task", "entry", "metric", "params"), REFUSED)
def test_resolve_metrics_refuses_native_params(
    task: str, entry: dict[str, Any], metric: str, params: list[str]
) -> None:
    with pytest.raises(LizyMLError) as caught:
        resolve_metrics([entry], task)  # type: ignore[arg-type]
    _assert_native_params_refused(caught.value, metric, params)


@pytest.mark.parametrize("task", ["regression", "binary"])
def test_empty_params_equal_the_bare_name(task: str) -> None:
    name = "huber" if task == "regression" else "auc"
    assert resolve_metrics([{name: {}}], task) == resolve_metrics([name], task)  # type: ignore[arg-type]


def test_feval_params_are_unchanged() -> None:
    native, fevals, names = resolve_metrics(
        [{"precision_at_k": {"k": 5}}, "auc"],
        "binary",  # type: ignore[arg-type]
    )
    assert native == ["auc"]
    assert len(fevals) == 1
    assert names == ["precision_at_k (k=5)"]


def test_an_unknown_name_keeps_the_whitelist_message() -> None:
    """The name's error comes first, as it did."""
    with pytest.raises(LizyMLError) as caught:
        resolve_metrics([{"no_such_metric": {"x": 1}}], "regression")  # type: ignore[arg-type]
    assert caught.value.code == CI
    assert "is not a valid LightGBM metric" in caught.value.user_message
    assert "params" not in caught.value.context


def test_an_unknown_name_after_a_parameterised_one_is_named_first() -> None:
    with pytest.raises(LizyMLError) as caught:
        resolve_metrics(
            [{"huber": {"delta": 9.0}}, "no_such_metric"],
            "regression",  # type: ignore[arg-type]
        )
    assert "'no_such_metric' is not a valid LightGBM metric" in (
        caught.value.user_message
    )


@pytest.mark.parametrize("key", ["metric", "metrics", "metric_types"])
@pytest.mark.parametrize(
    "value",
    [[{"huber": {"delta": 9.0}}], {"huber": {"delta": 9.0}}],
    ids=["list", "single"],
)
def test_model_params_refuses_every_spelling(key: str, value: Any) -> None:
    cfg = make_config("regression", n_estimators=3, n_splits=2)
    cfg["model"]["params"][key] = value
    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as caught:
        Model(cfg).fit(data=make_regression_df(n=80))
    _assert_native_params_refused(caught.value, "huber", ["delta"])
    assert caught.value.context["surface"] == "model.params"
    assert caught.value.context["parameter"] == key
    assert seen["train_params"] == []


def test_fit_params_refuses() -> None:
    cfg = make_config("regression", n_estimators=3, n_splits=2)
    with pytest.raises(LizyMLError) as caught:
        Model(cfg).fit(
            data=make_regression_df(n=80),
            params={"metric": [{"rmse": {"bogus": 1}}]},
        )
    _assert_native_params_refused(caught.value, "rmse", ["bogus"])
    assert caught.value.context["surface"] == "fit(params=)"


def test_direct_adapter_refuses() -> None:
    df = make_regression_df(n=80)
    adapter = LGBMAdapter(
        task="regression",
        params={"n_estimators": 3, "metric": [{"huber": {"delta": 9.0}}]},
    )
    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as caught:
        adapter.fit(df.drop(columns=["target"]), df["target"])
    _assert_native_params_refused(caught.value, "huber", ["delta"])
    assert seen["train_params"] == []


def test_feval_params_still_fit() -> None:
    cfg = make_config(
        "binary", n_estimators=3, n_splits=2, metric=[{"precision_at_k": {"k": 5}}]
    )
    Model(cfg).fit(data=make_binary_df(n=120))


# ---------------------------------------------------------------------------
# Criterion 6: an artifact recorded before the rule
# ---------------------------------------------------------------------------


@pytest.fixture
def old_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An artifact as v0.18.0 wrote it: the parameters recorded and dropped.

    The refusal is switched off while fitting, which is the code path of the
    version that wrote such artifacts; everything recorded is real.
    """
    with monkeypatch.context() as patched:
        patched.setattr(metric_bridge, "_refuse_native_metric_params", _accept_all)
        cfg = make_config(
            "regression",
            n_estimators=5,
            n_splits=2,
            metric=[{"huber": {"delta": 9.0}}],
        )
        model = Model(cfg)
        model.fit(data=make_regression_df(n=100))
        path = tmp_path / "old"
        model.export(path)
    return path


def _accept_all(*_args: Any, **_kwargs: Any) -> None:
    return None


def test_old_artifact_loads_and_predicts(old_artifact: Path) -> None:
    df = make_regression_df(n=100)
    model = Model.load(old_artifact)
    out = model.predict(df.drop(columns=["target"]).iloc[:10])
    assert out.pred.shape == (10,)
    assert model._cfg.model.params["metric"] == [{"huber": {"delta": 9.0}}]  # type: ignore[union-attr]


def test_old_artifact_refit_is_refused(old_artifact: Path) -> None:
    model = Model.load(old_artifact)
    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as caught:
        model.fit(data=make_regression_df(n=100))
    _assert_native_params_refused(caught.value, "huber", ["delta"])
    assert seen["train_params"] == []


def test_old_artifact_export_code_is_refused(
    old_artifact: Path, tmp_path: Path
) -> None:
    model = Model.load(old_artifact)
    out = tmp_path / "code"
    with pytest.raises(LizyMLError) as caught:
        model.export_code(out)
    _assert_native_params_refused(caught.value, "huber", ["delta"])
    assert not (out / "train.py").exists()
