"""H-0120 amendments 1 and 4: the records a fit leaves for ``export_code``.

Amendment 1 records the row-weight rule the refit applied (``"balanced"`` or
``"none"``, metadata key ``applied_sample_weight``); amendment 4 records the
features the fit's input frame held as ``category`` (metadata key
``declared_categories``). Both follow H-0109's ``applied_training_params``:
written by ``Model.export``, restored by ``Model.load``, replaced only by a
successful fit, and *unknown* -- not empty -- for an artifact without the key.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from tests.test_codegen._retrain_harness import make_config, make_frame

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

WEIGHT_KEY = "applied_sample_weight"
DECLARED_KEY = "declared_categories"


def _metadata(path: Path) -> dict[str, Any]:
    return json.loads((path / "metadata.json").read_text(encoding="utf-8"))


def _rewrite_metadata(path: Path, edit: Any) -> None:
    metadata = _metadata(path)
    edit(metadata)
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def _balanced_space(value: bool) -> dict[str, Any]:
    return {
        "balanced": {"type": "categorical", "choices": [value], "category": "smart"}
    }


def _tuning(cfg: dict[str, Any], space: dict[str, Any]) -> dict[str, Any]:
    return {
        **cfg,
        "tuning": {
            "optuna": {
                "params": {"n_trials": 1, "direction": "minimize"},
                "space": space,
                "space_mode": "replace",
            }
        },
    }


def _exported_weight(model: Model, tmp_path: Path, name: str = "gen") -> Any:
    project = tmp_path / name
    model.export_code(project)
    config = json.loads((project / "config.json").read_text(encoding="utf-8"))
    return config["sample_weight"]


def _refused_fit(model: Model, df: pd.DataFrame) -> None:
    """A fit a gate refuses before training."""
    with pytest.raises(LizyMLError):
        model.fit(data=df, params={"objective": "regression"})


def _failed_fit(model: Model, df: pd.DataFrame) -> None:
    """A fit that raises mid-training."""
    with (
        mock.patch(
            "lizyml.training.cv_trainer.CVTrainer.fit",
            side_effect=RuntimeError("training blew up"),
        ),
        pytest.raises(RuntimeError),
    ):
        model.fit(data=df)


# ---------------------------------------------------------------------------
# Amendment 1: the weight rule
# ---------------------------------------------------------------------------

_WEIGHT_CASES = {
    "multiclass-default": ("multiclass", {}, "balanced"),
    "multiclass-false": ("multiclass", {"balanced": False}, "none"),
    "binary": ("binary", {}, "none"),
    "regression": ("regression", {}, "none"),
}


@pytest.mark.parametrize("case", sorted(_WEIGHT_CASES))
def test_both_values_are_written_restored_and_rewritten(
    case: str, tmp_path: Path
) -> None:
    task, model_extra, expected = _WEIGHT_CASES[case]
    model = Model(make_config(task, "kfold", model_extra=model_extra))
    model.fit(data=make_frame(task))
    assert model._get_fit_state().applied_sample_weight == expected

    artifact = model.export(tmp_path / "artifact")
    assert _metadata(artifact)[WEIGHT_KEY] == expected
    loaded = Model.load(artifact)
    assert loaded._get_fit_state().applied_sample_weight == expected
    again = loaded.export(tmp_path / "again")
    assert _metadata(again)[WEIGHT_KEY] == expected
    assert _exported_weight(loaded, tmp_path) == (
        "balanced" if expected == "balanced" else None
    )


def _multiclass_with_balanced_space(value: bool) -> tuple[Model, pd.DataFrame]:
    df = make_frame("multiclass")
    cfg = _tuning(make_config("multiclass", "kfold"), _balanced_space(value))
    return Model(cfg), df


def test_a_later_fit_with_another_rule_replaces_the_record(tmp_path: Path) -> None:
    model, df = _multiclass_with_balanced_space(False)
    model.fit(data=df)
    assert model._get_fit_state().applied_sample_weight == "balanced"
    model.tune(data=df)
    model.fit(data=df)  # now with the tuned balanced=False
    assert model._get_fit_state().applied_sample_weight == "none"
    assert _exported_weight(model, tmp_path) is None


def test_a_fit_after_load_replaces_the_record(tmp_path: Path) -> None:
    model, df = _multiclass_with_balanced_space(False)
    model.fit(data=df)
    loaded = Model.load(model.export(tmp_path / "artifact"))
    loaded.tune(data=df)
    loaded.fit(data=df)
    assert loaded._get_fit_state().applied_sample_weight == "none"


@pytest.mark.parametrize("then", ["tune", "refused_fit", "failed_fit"])
@pytest.mark.parametrize("record", ["known", "unknown"])
def test_the_weight_record_survives_calls_that_do_not_replace_the_fit(
    record: str, then: str, tmp_path: Path
) -> None:
    model, df = _multiclass_with_balanced_space(False)
    model.fit(data=df)
    artifact = model.export(tmp_path / "artifact")
    if record == "unknown":
        _rewrite_metadata(artifact, lambda m: m.pop(WEIGHT_KEY))
    loaded = Model.load(artifact)
    before = loaded._get_fit_state().applied_sample_weight
    assert before == ("balanced" if record == "known" else None)

    if then == "tune":
        loaded.tune(data=df)
    elif then == "refused_fit":
        _refused_fit(loaded, df)
    else:
        _failed_fit(loaded, df)

    assert loaded._get_fit_state().applied_sample_weight == before
    out = _metadata(loaded.export(tmp_path / "after"))
    if record == "known":
        assert out[WEIGHT_KEY] == before
    else:
        assert WEIGHT_KEY not in out


def test_fit_then_tune_exports_the_rule_the_fit_used(tmp_path: Path) -> None:
    """The tuning result says ``balanced: False``; the fit used weights."""
    model, df = _multiclass_with_balanced_space(False)
    model.fit(data=df)
    model.tune(data=df)
    assert model._tuning_result.best_smart_params["balanced"] is False  # type: ignore[union-attr]
    assert _exported_weight(model, tmp_path) == "balanced"
    loaded = Model.load(model.export(tmp_path / "artifact"))
    assert _exported_weight(loaded, tmp_path, "gen-loaded") == "balanced"


@pytest.mark.parametrize(
    ("config_balanced", "tuned_balanced", "expected"),
    [(None, False, None), (False, True, "balanced")],
    ids=["config-default-tuned-false", "config-false-tuned-true"],
)
def test_an_artifact_without_the_weight_record_derives_from_the_tuning_result(
    config_balanced: bool | None,
    tuned_balanced: bool,
    expected: str | None,
    tmp_path: Path,
) -> None:
    """Unknown record: config and the restored tuning result decide, and the
    tuning result wins where they disagree (config alone would fail here)."""
    df = make_frame("multiclass")
    extra = {} if config_balanced is None else {"balanced": config_balanced}
    cfg = _tuning(
        make_config("multiclass", "kfold", model_extra=extra),
        _balanced_space(tuned_balanced),
    )
    model = Model(cfg)
    model.fit(data=df)
    model.tune(data=df)
    artifact = model.export(tmp_path / "artifact")
    _rewrite_metadata(artifact, lambda m: m.pop(WEIGHT_KEY))
    loaded = Model.load(artifact)
    assert loaded._get_fit_state().applied_sample_weight is None
    assert WEIGHT_KEY not in _metadata(loaded.export(tmp_path / "again"))
    assert _exported_weight(loaded, tmp_path) == expected


@pytest.mark.parametrize("value", [True, None, "Balanced", 1, ["balanced"]])
def test_a_malformed_weight_record_is_refused_on_load(
    value: Any, tmp_path: Path
) -> None:
    model = Model(make_config("multiclass", "kfold"))
    model.fit(data=make_frame("multiclass"))
    artifact = model.export(tmp_path / "artifact")
    _rewrite_metadata(artifact, lambda m: m.__setitem__(WEIGHT_KEY, value))
    with pytest.raises(LizyMLError) as info:
        Model.load(artifact)
    assert info.value.code == ErrorCode.DESERIALIZATION_FAILED


# ---------------------------------------------------------------------------
# Amendment 4: features the input declared `category`
# ---------------------------------------------------------------------------


def _declared_frame() -> pd.DataFrame:
    df = make_frame("binary")
    rng = np.random.default_rng(3)
    df["d"] = pd.Categorical(
        rng.choice(["z", "y"], len(df)), categories=["z", "y", "x"]
    )
    df["s"] = rng.choice(["q", "p"], len(df))
    return df


def _declared_model() -> tuple[Model, pd.DataFrame]:
    df = _declared_frame()
    model = Model(make_config("binary", "kfold"))
    model.fit(data=df)
    return model, df


def test_only_the_input_category_columns_are_recorded(tmp_path: Path) -> None:
    model, _ = _declared_model()
    # "s" is cast to category by the builder too, but the input held a string.
    assert model.fit_result.dtypes["s"] == "category"
    assert model._get_fit_state().declared_categories == {"d": ["z", "y", "x"]}


def test_the_declared_record_round_trips(tmp_path: Path) -> None:
    model, _ = _declared_model()
    artifact = model.export(tmp_path / "artifact")
    assert _metadata(artifact)[DECLARED_KEY] == {"d": ["z", "y", "x"]}
    loaded = Model.load(artifact)
    assert loaded._get_fit_state().declared_categories == {"d": ["z", "y", "x"]}
    assert _metadata(loaded.export(tmp_path / "again"))[DECLARED_KEY] == {
        "d": ["z", "y", "x"]
    }
    project = tmp_path / "gen"
    loaded.export_code(project)
    config = json.loads((project / "config.json").read_text(encoding="utf-8"))
    assert config["declared_categories"] == {"d": ["z", "y", "x"]}


def test_an_artifact_without_the_declared_record_is_unknown(tmp_path: Path) -> None:
    model, _ = _declared_model()
    artifact = model.export(tmp_path / "artifact")
    _rewrite_metadata(artifact, lambda m: m.pop(DECLARED_KEY))
    loaded = Model.load(artifact)
    assert loaded._get_fit_state().declared_categories is None
    assert DECLARED_KEY not in _metadata(loaded.export(tmp_path / "again"))
    project = tmp_path / "gen"
    loaded.export_code(project)
    config = json.loads((project / "config.json").read_text(encoding="utf-8"))
    assert config["declared_categories"] == {}


@pytest.mark.parametrize("then", ["tune", "refused_fit", "failed_fit"])
def test_the_declared_record_survives_calls_that_do_not_replace_the_fit(
    then: str,
) -> None:
    df = _declared_frame()
    model = Model(_tuning(make_config("binary", "kfold"), _balanced_space(True)))
    model.fit(data=df)
    plain = df.assign(d=df["d"].astype(str))  # a later call without the dtype
    if then == "tune":
        model.tune(data=plain)
    elif then == "refused_fit":
        _refused_fit(model, plain)
    else:
        _failed_fit(model, plain)
    assert model._get_fit_state().declared_categories == {"d": ["z", "y", "x"]}


@pytest.mark.parametrize(
    "values",
    [
        [(1, 2), (3, 4)],
        [b"x", b"y"],
        [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")],
    ],
    ids=["tuple", "bytes", "Timestamp"],
)
def test_a_record_of_tagged_values_survives_export_and_load(
    values: list[Any], tmp_path: Path
) -> None:
    """H-0133 P6: the record is written with the tagged codec, not omitted;
    both loads return the values themselves, and export_code from the loaded
    model writes them (H-0120 left the key out and refused the tuple)."""
    from lizyml.persistence import load

    df = make_frame("binary")
    df["d"] = pd.Categorical([values[i % 2] for i in range(len(df))], categories=values)
    model = Model(make_config("binary", "kfold"))
    model.fit(data=df)
    artifact = model.export(tmp_path / "artifact")
    written = _metadata(artifact)[DECLARED_KEY]["d"]
    assert written != values  # tagged JSON, not the values themselves
    assert load(artifact)[2][DECLARED_KEY] == {"d": values}
    loaded = Model.load(artifact)
    assert loaded._get_fit_state().declared_categories == {"d": values}
    loaded.export_code(tmp_path / "gen")
    config = json.loads((tmp_path / "gen" / "config.json").read_text(encoding="utf-8"))
    assert config[DECLARED_KEY]["d"] == written


def test_a_malformed_tagged_record_is_refused_by_both_loads(tmp_path: Path) -> None:
    """H-0133 criterion 6 / negative control (h): a tag the exporter never
    writes fails both loads with DESERIALIZATION_FAILED."""
    from lizyml.persistence import load

    df = make_frame("binary")
    df["d"] = pd.Categorical(["a", "b"] * (len(df) // 2), categories=["a", "b"])
    model = Model(make_config("binary", "kfold"))
    model.fit(data=df)
    artifact = model.export(tmp_path / "artifact")
    meta_path = artifact / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta[DECLARED_KEY]["d"] = [{"$lizyml": "bytes", "b64": "YQ==", "extra": 1}]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    for read in (load, Model.load):
        with pytest.raises(LizyMLError) as info:
            read(artifact)
        assert info.value.code == ErrorCode.DESERIALIZATION_FAILED
        assert info.value.context["key"] == "d"


@pytest.mark.parametrize("read", ["Model.load", "persistence.load"])
@pytest.mark.parametrize(
    "record",
    [
        ["d"],
        {"d": "z"},
        {"d": [["z"]]},
        {"d": [None]},
        {"d": [{"a": 1}]},
        {"d": [float("nan")]},
        {"d": [{"$lizyml": "decimal", "str": "NaN"}]},
    ],
    ids=[
        "not-object",
        "not-list",
        "nested-list",
        "null",
        "object",
        "json-nan",
        "decimal-nan",
    ],
)
def test_a_malformed_declared_record_is_refused_on_load(
    record: Any, read: str, tmp_path: Path
) -> None:
    """Both public loads refuse it (H-0133 criterion 6): the shape check and
    the decode live in ``persistence.load``."""
    from lizyml.persistence import load

    model, _ = _declared_model()
    artifact = model.export(tmp_path / "artifact")
    _rewrite_metadata(artifact, lambda m: m.__setitem__(DECLARED_KEY, record))
    with pytest.raises(LizyMLError) as info:
        (Model.load if read == "Model.load" else load)(artifact)
    assert info.value.code == ErrorCode.DESERIALIZATION_FAILED
