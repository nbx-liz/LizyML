"""H-0125 policy 5 (#307): ``Model.load`` refuses a booster trained with
``cross_entropy_lambda``.

Such a booster returns λ, not a probability, and v0.18.0 could export one (a
short fit whose outputs stayed below 1). The check reads what LightGBM recorded
inside each stored booster -- the ``objective=`` line of its model string --
and never its ``params``: a booster restored from its model string has none
(H-0125 ``probes/booster_objective_record.txt``), and ``params`` can be written
after training. (A joblib round trip does keep ``params`` on LightGBM 4.6, so
the params-less case is built with ``lgb.Booster(model_str=...)``.) Every
booster is read: each fold model of ``fit_result.pkl`` and the refit model, in
format_version 1, the v2 shape and 3 (criteria 7 and 8; H-0133 P8).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pytest

from lizyml.core import _model_persistence
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.model import Model
from lizyml.estimators.lgbm import provider
from lizyml.estimators.lgbm.defaults import TASK_COMPATIBLE_OBJECTIVES
from lizyml.estimators.lgbm.param_validation import recorded_objective
from tests._helpers import make_binary_df, make_config
from tests.test_persistence._artifact_versions import resign, to_v2_shape

DF = make_binary_df(n=150, seed=0)
X = DF.drop(columns=["target"])
N_SPLITS = 3
#: Where one forbidden booster is placed: a fold (first, middle, last) or the refit.
POSITIONS = ("fold-0", "fold-1", "fold-2", "refit")


def _accept_cross_entropy_lambda(patched: pytest.MonkeyPatch) -> None:
    """The two tables of v0.18.0, which accepted the objective.

    Both, as the import-time consistency check of ``provider.py`` requires.
    """
    patched.setitem(
        TASK_COMPATIBLE_OBJECTIVES,
        "binary",
        frozenset({"binary", "cross_entropy", "cross_entropy_lambda"}),
    )
    patched.setitem(
        provider._LGBM_OBJECTIVE_CHOICES,
        "binary",
        ("binary", "cross_entropy", "cross_entropy_lambda"),
    )


def _export(
    tmp_path: Path, name: str, *, objective: str | None = None, **fit_kwargs: Any
) -> Path:
    extra = {} if objective is None else {"objective": objective}
    model = Model(make_config("binary", n_estimators=5, n_splits=N_SPLITS, **extra))
    model.fit(data=DF, **fit_kwargs)
    path = tmp_path / name
    model.export(path)
    return path


def _resign_if_signed(path: Path) -> None:
    """Re-pickled payloads no longer match the recorded digests (H-0083).

    Re-sign them rather than drop ``checksums``: a format 3-4 artifact without
    it is refused (H-0135 P2). The v1 recipe has none to re-sign.
    """
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if "checksums" in meta:
        resign(path)


def _to_format_version_1(path: Path) -> None:
    """The recipe of ``test_v1_metadata_still_loads``: a v1 metadata and layout."""
    to_v2_shape(path)  # v1 predates H-0133's stand_in_columns too
    meta_path = path / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["format_version"] = 1
    meta.pop("checksums", None)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    fit_pkl = path / "fit_result.pkl"
    fit_result = joblib.load(fit_pkl)
    fit_result.target_encoder = None
    joblib.dump(fit_result, fit_pkl, compress=3)


def _lambda_booster(
    params_after: dict[str, Any] | None = None, *, restored: bool = False
) -> lgb.Booster:
    """A booster LightGBM trained with ``cross_entropy_lambda`` on the same features.

    *restored* rebuilds it from its model string and empties ``params``:
    LightGBM 4.6.0 drops them on that restore and 4.7.0 keeps them, so the
    test sets the "no params" case itself instead of relying on either.
    """
    booster = lgb.train(
        {"objective": "cross_entropy_lambda", "verbosity": -1, "learning_rate": 0.001},
        lgb.Dataset(X, DF["target"]),
        num_boost_round=3,
    )
    if restored:
        booster = lgb.Booster(model_str=booster.model_to_string())
        booster.params = {}
    if params_after is not None:
        booster.params = dict(params_after)
    return booster


def _replace(path: Path, position: str, booster: lgb.Booster) -> None:
    if position == "refit":
        pkl = path / "refit_model.pkl"
        refit = joblib.load(pkl)
        refit.model._model = booster
        joblib.dump(refit, pkl, compress=3)
    else:
        index = int(position.split("-")[1])
        pkl = path / "fit_result.pkl"
        fit_result = joblib.load(pkl)
        fit_result.models[index]._model = booster
        joblib.dump(fit_result, pkl, compress=3)
    _resign_if_signed(path)


def _assert_refused(path: Path) -> LizyMLError:
    with pytest.raises(LizyMLError) as caught:
        Model.load(path)
    error = caught.value
    assert error.code == ErrorCode.DESERIALIZATION_FAILED
    assert error.context["objective"] == "cross_entropy_lambda"
    assert error.context["path"] == str(path)
    message = error.user_message
    assert "cross_entropy_lambda" in message
    assert "not a probability" in message
    assert "'binary'" in message and "'cross_entropy'" in message
    return error


VERSIONS: dict[str, Callable[[Path], None]] = {
    "v5": lambda _path: None,
    "v2": to_v2_shape,  # H-0133 P8: the shape a v2 artifact has
    "v1": _to_format_version_1,
}


# ---------------------------------------------------------------------------
# Criterion 7: artifacts trained with the objective
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", sorted(VERSIONS))
@pytest.mark.parametrize("route", ["model.params", "fit(params=)", "alias"])
def test_an_artifact_trained_with_the_objective_is_refused(
    route: str, version: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as patched:
        _accept_cross_entropy_lambda(patched)
        if route == "model.params":
            path = _export(tmp_path, "a", objective="cross_entropy_lambda")
        elif route == "fit(params=)":
            path = _export(tmp_path, "a", params={"objective": "cross_entropy_lambda"})
        else:
            path = _export(
                tmp_path, "a", params={"application": "cross_entropy_lambda"}
            )
    VERSIONS[version](path)
    error = _assert_refused(path)
    assert error.context["booster"] == "fold 0"


@pytest.mark.parametrize("version", sorted(VERSIONS))
@pytest.mark.parametrize("objective", sorted(TASK_COMPATIBLE_OBJECTIVES["binary"]))
def test_an_artifact_of_an_accepted_objective_loads(
    objective: str, version: str, tmp_path: Path
) -> None:
    path = _export(tmp_path, "a", objective=objective)
    VERSIONS[version](path)
    model = Model.load(path)
    proba = model.predict(X.iloc[:10]).proba
    assert proba is not None
    assert np.all((proba >= 0.0) & (proba <= 1.0))


@pytest.mark.parametrize("version", sorted(VERSIONS))
@pytest.mark.parametrize("position", POSITIONS)
def test_one_forbidden_booster_at_any_position_is_refused(
    position: str, version: str, tmp_path: Path
) -> None:
    """Every booster is read, not a sample of them."""
    path = _export(tmp_path, "a")
    _replace(path, position, _lambda_booster())
    VERSIONS[version](path)
    error = _assert_refused(path)
    expected = "refit" if position == "refit" else f"fold {position.split('-')[1]}"
    assert error.context["booster"] == expected


@pytest.mark.parametrize("version", sorted(VERSIONS))
def test_an_unreplaced_control_loads(version: str, tmp_path: Path) -> None:
    """The same edits without a forbidden booster: re-pickled and re-signed."""
    path = _export(tmp_path, "a")
    _replace(path, "refit", joblib.load(path / "refit_model.pkl").model._model)
    VERSIONS[version](path)
    Model.load(path).predict(X.iloc[:5])


# ---------------------------------------------------------------------------
# Criterion 8: the record LightGBM keeps, not the params
# ---------------------------------------------------------------------------


def test_the_reader_uses_the_record_of_a_restored_booster() -> None:
    restored = lgb.Booster(model_str=_lambda_booster().model_to_string())
    restored.params = {}  # whether the restore keeps them depends on the version
    assert recorded_objective(restored) == "cross_entropy_lambda"


def test_the_reader_ignores_params_overwritten_to_binary() -> None:
    restored = lgb.Booster(model_str=_lambda_booster().model_to_string())
    restored.params = {"objective": "binary"}
    assert recorded_objective(restored) == "cross_entropy_lambda"


def test_the_reader_reads_the_alias_as_recorded() -> None:
    booster = lgb.train(
        {"objective": "xentlambda", "verbosity": -1},
        lgb.Dataset(X, DF["target"]),
        num_boost_round=2,
    )
    assert recorded_objective(booster) == "cross_entropy_lambda"


@pytest.mark.parametrize("objective", ["binary", "cross_entropy"])
def test_the_reader_reads_the_first_token(objective: str) -> None:
    booster = lgb.train(
        {"objective": objective, "verbosity": -1},
        lgb.Dataset(X, DF["target"]),
        num_boost_round=2,
    )
    assert recorded_objective(booster) == objective


@pytest.mark.parametrize("version", sorted(VERSIONS))
@pytest.mark.parametrize("position", ["fold-1", "refit"])
def test_a_restored_booster_without_params_is_refused(
    position: str, version: str, tmp_path: Path
) -> None:
    path = _export(tmp_path, "a")
    _replace(path, position, _lambda_booster(restored=True))
    stored = (
        joblib.load(path / "refit_model.pkl").model
        if position == "refit"
        else joblib.load(path / "fit_result.pkl").models[1]
    )
    assert not stored.get_native_model().params
    VERSIONS[version](path)
    _assert_refused(path)


@pytest.mark.parametrize("version", sorted(VERSIONS))
@pytest.mark.parametrize("restored", [False, True], ids=["trained", "restored"])
def test_params_written_as_binary_do_not_hide_the_record(
    restored: bool, version: str, tmp_path: Path
) -> None:
    path = _export(tmp_path, "a")
    _replace(
        path,
        "fold-1",
        _lambda_booster(params_after={"objective": "binary"}, restored=restored),
    )
    stored = joblib.load(path / "fit_result.pkl").models[1].get_native_model()
    assert stored.params == {"objective": "binary"}
    VERSIONS[version](path)
    _assert_refused(path)


def test_the_artifact_check_reads_no_params() -> None:
    """A booster whose ``params`` say ``cross_entropy_lambda`` but whose record
    says ``binary`` is not refused: the record decides."""
    booster = lgb.train(
        {"objective": "binary", "verbosity": -1},
        lgb.Dataset(X, DF["target"]),
        num_boost_round=2,
    )
    booster.params = {"objective": "cross_entropy_lambda"}

    class _Adapter:
        def get_native_model(self) -> lgb.Booster:
            return booster

    class _Fit:
        models = [_Adapter()]

    class _Refit:
        model = _Adapter()

    _model_persistence._refuse_non_probability_boosters(_Fit(), _Refit(), "p")
