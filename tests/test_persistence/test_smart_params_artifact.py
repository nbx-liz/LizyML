"""H-0121 amendment 2, policy 4 (criterion 5): a stored tune result's smart values.

An artifact records the tune result's ``best_smart_params``, and a re-``fit``
of the loaded Model overlays them. Before #347 a smart dimension's value was
recorded as sampled, so an artifact can carry ``"false"`` for a booster
trained with ``auto_num_leaves`` true: its record disagrees with the model.
``Model.load`` checks every recorded value with the provider's rule (the same
one a tune's dimensions get), for the artifact config's task, and refuses one
outside it. An artifact whose record is inside the rule, empty, or absent
loads and predicts as before.

The artifacts are written by the current code and their ``metadata.json``
edited, for ``format_version`` 3, rewritten to the v2 shape (H-0133 P8), and
rewritten to 1.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from tests._helpers import make_config, make_multiclass_df, make_regression_df
from tests.test_persistence._artifact_versions import (
    to_v2_shape,
    to_v3_shape,
    to_v4_shape,
)

DF = ErrorCode.DESERIALIZATION_FAILED
LR: dict[str, Any] = {"type": "float", "low": 0.05, "high": 0.3, "log": True}


def _frame(task: str) -> Any:
    return (
        make_multiclass_df(n=150) if task == "multiclass" else make_regression_df(n=120)
    )


def _export(task: str, root: Path, *, tuned: bool = True) -> Path:
    split = "kfold" if task == "regression" else "stratified_kfold"
    cfg = make_config(
        task,
        n_estimators=3,
        n_splits=2,
        split_method=split,
        tuning_n_trials=1 if tuned else None,
    )
    if tuned:
        cfg["tuning"]["optuna"]["space_mode"] = "replace"
        cfg["tuning"]["optuna"]["space"] = {"learning_rate": copy.deepcopy(LR)}
    model = Model(cfg, data=_frame(task))
    if tuned:
        model.tune()
    model.fit()
    return model.export(root / "export")


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("artifacts")
    return {
        "regression": _export("regression", root / "regression"),
        "multiclass": _export("multiclass", root / "multiclass"),
        "untuned": _export("regression", root / "untuned", tuned=False),
    }


def _edited(
    source: Path, target: Path, *, smart: dict[str, Any] | None, version: int
) -> Path:
    """A copy of *source* with ``best_smart_params`` and ``format_version`` set."""
    shutil.copytree(source, target)
    if version <= 2:
        to_v2_shape(target)  # what a v1 / v2 artifact holds (H-0133 P8)
    elif version == 3:
        to_v3_shape(target)  # no history, no identity marker (H-0134, H-0138)
    elif version == 4:
        to_v4_shape(target)  # no identity marker (H-0138)
    path = target / "metadata.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    if smart is not None:
        meta["tuning"]["best_smart_params"] = smart
    meta["format_version"] = version
    if version == 1:
        # A real v1 artifact predates the H-0083 checksums.
        meta.pop("checksums", None)
    path.write_text(json.dumps(meta), encoding="utf-8")
    return target


#: ``(artifact, best_smart_params, name, value, reason)`` refused on load.
REFUSED: list[tuple[str, dict[str, Any], str, Any, str]] = [
    (
        "regression",
        {"auto_num_leaves": "false"},
        "auto_num_leaves",
        "false",
        "smart_value_invalid",
    ),
    (
        "regression",
        {"auto_num_leaves": None},
        "auto_num_leaves",
        None,
        "smart_value_invalid",
    ),
    (
        "regression",
        {"num_leaves_ratio": 1.5},
        "num_leaves_ratio",
        1.5,
        "smart_value_invalid",
    ),
    (
        "regression",
        {"min_data_in_leaf_ratio": 0},
        "min_data_in_leaf_ratio",
        0,
        "smart_value_invalid",
    ),
    (
        "regression",
        {"feature_weights": {"a": 1.0}},
        "feature_weights",
        {"a": 1.0},
        "smart_value_invalid",
    ),
    (
        "regression",
        {"balanced": True},
        "balanced",
        True,
        "smart_value_unsupported_task",
    ),
    # No tune records a name the provider does not declare (such a dimension
    # is refused before the study), so a record with one is not LizyML's own.
    (
        "regression",
        {"not_a_smart_param": 1},
        "not_a_smart_param",
        1,
        "smart_value_invalid",
    ),
]


@pytest.mark.parametrize("edit", ["no_best_smart_params", "tuning_not_an_object"])
def test_load_refuses_a_malformed_tuning_block(
    edit: str, artifacts: dict[str, Path], tmp_path: Path
) -> None:
    """A tuning block LizyML did not write is refused, not a raw KeyError."""
    path = _edited(artifacts["regression"], tmp_path / "a", smart=None, version=2)
    meta_path = path / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if edit == "no_best_smart_params":
        del meta["tuning"]["best_smart_params"]
    else:
        meta["tuning"] = ["not", "an", "object"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(LizyMLError) as caught:
        Model.load(path)
    assert caught.value.code is DF
    assert "tuning" in caught.value.user_message


@pytest.mark.parametrize("version", [3, 2, 1])
def test_load_refuses_a_record_that_is_not_a_mapping(
    version: int, artifacts: dict[str, Path], tmp_path: Path
) -> None:
    path = _edited(artifacts["regression"], tmp_path / "a", smart=None, version=version)
    meta_path = path / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["tuning"]["best_smart_params"] = ["auto_num_leaves", False]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(LizyMLError) as caught:
        Model.load(path)
    assert caught.value.code is DF
    assert "best_smart_params" in caught.value.user_message


@pytest.mark.parametrize("version", [3, 2, 1])
@pytest.mark.parametrize(
    "row", REFUSED, ids=[f"{r[0]}-{json.dumps(r[1])}" for r in REFUSED]
)
def test_load_refuses_a_recorded_value_outside_the_rule(
    row: tuple[str, dict[str, Any], str, Any, str],
    version: int,
    artifacts: dict[str, Path],
    tmp_path: Path,
) -> None:
    artifact, smart, name, value, reason = row
    path = _edited(artifacts[artifact], tmp_path / "a", smart=smart, version=version)
    with pytest.raises(LizyMLError) as caught:
        Model.load(path)
    err = caught.value
    assert err.code is DF
    assert err.context["name"] == name
    assert err.context["value"] == value
    assert type(err.context["value"]) is type(value)
    assert err.context["reason"] == reason
    assert name in err.user_message
    assert "tune again" in err.user_message.lower()
    assert "export again" in err.user_message.lower()


#: ``(artifact, best_smart_params)`` that load; ``None`` keeps the record.
ACCEPTED: list[tuple[str, dict[str, Any] | None]] = [
    ("regression", {"auto_num_leaves": False, "balanced": None}),
    ("multiclass", {"balanced": True}),
    ("regression", {}),
    ("regression", None),
    ("untuned", None),
]


@pytest.mark.parametrize("version", [3, 2, 1])
@pytest.mark.parametrize(
    "row", ACCEPTED, ids=[f"{r[0]}-{json.dumps(r[1])}" for r in ACCEPTED]
)
def test_load_keeps_an_artifact_inside_the_rule(
    row: tuple[str, dict[str, Any] | None],
    version: int,
    artifacts: dict[str, Path],
    tmp_path: Path,
) -> None:
    artifact, smart = row
    task = "multiclass" if artifact == "multiclass" else "regression"
    X = _frame(task).drop(columns=["target"]).iloc[:10]
    expected = Model.load(artifacts[artifact]).predict(X).pred
    path = _edited(artifacts[artifact], tmp_path / "a", smart=smart, version=version)
    loaded = Model.load(path)
    np.testing.assert_array_equal(loaded.predict(X).pred, expected)
    if artifact == "untuned":
        assert loaded._tuning_result is None
    elif smart is not None:
        assert loaded._tuning_result.best_smart_params == smart
