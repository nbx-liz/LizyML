"""The training overlay a fit applied is recorded in the artifact (H-0109, #281).

``params_table()`` and ``export_code()`` answer "what did *this* fit use?"
(H-0094 decision 13). The patience comes from the trained adapter, which the
artifact pickles. The inner-validation ratio is not on the adapter, so it came
from in-session state only, and after ``Model.load()`` both surfaces fell back to
the configured ratio -- wrong after ``tune -> fit -> export -> load``, where the
fit trained with the tuned one.

The tuning block H-0086 writes cannot stand in for the record: ``export()``
writes the model's *current* tuning result, and after ``fit -> tune`` that
result was consumed by no fit. Measured: ``best_training_params`` was the same
dict in both orderings (``results/pr8b_metadata_keys_before.txt``).

Three states are kept apart: ``None`` (an artifact without the record -- unknown),
``{}`` (the fit applied no overlay), and the applied dict. Collapsing the first
two would write a definite ``{}`` when a pre-H-0109 artifact is re-exported.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.model import Model
from tests._helpers import (
    make_binary_df,
    make_config,
    make_multiclass_df,
    make_regression_df,
)

KEY = "applied_training_params"
CONFIG_RATIO = 0.2
TUNED_RATIO = 0.45
TUNED_PATIENCE = 2

#: Both training dimensions, as caller-written categoricals, so the tuned values
#: are known and differ from the config.
SPACE: dict[str, Any] = {
    "early_stopping_rounds": {
        "type": "categorical",
        "choices": [TUNED_PATIENCE],
        "category": "training",
    },
    "validation_ratio": {
        "type": "categorical",
        "choices": [TUNED_RATIO],
        "category": "training",
    },
}

DATA = {
    "regression": make_regression_df,
    "binary": make_binary_df,
    "multiclass": make_multiclass_df,
}


def _model(
    task: str = "binary",
    *,
    space: dict[str, Any] | None = None,
    space_mode: str = "replace",
) -> Model:
    cfg = make_config(
        task, n_estimators=10, n_splits=2, num_threads=1, tuning_n_trials=1
    )
    cfg["training"]["early_stopping"] = {
        "enabled": True,
        "rounds": 7,
        "validation_ratio": CONFIG_RATIO,
    }
    cfg["tuning"]["optuna"]["space"] = dict(SPACE if space is None else space)
    cfg["tuning"]["optuna"]["space_mode"] = space_mode
    return Model(cfg, data=DATA[task](n=200))


def _run(model: Model, lifecycle: str) -> Model:
    steps = {
        "fit": ("fit",),
        "tune_fit": ("tune", "fit"),
        "fit_tune": ("fit", "tune"),
    }[lifecycle]
    for step in steps:
        getattr(model, step)()
    return model


def _metadata(path: Path) -> dict[str, Any]:
    return json.loads((path / "metadata.json").read_text(encoding="utf-8"))


def _rewrite_metadata(path: Path, edit: Any) -> None:
    metadata = _metadata(path)
    edit(metadata)
    (path / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def _reported_ratio(model: Model) -> Any:
    return model.params_table().loc["validation_ratio", "value"]


def _exported_ratio(model: Model) -> Any:
    with mock.patch("lizyml.codegen.generator.generate_code") as generate:
        model.export_code("not-written")
    return generate.call_args.kwargs["validation_ratio"]


@pytest.mark.parametrize("lifecycle", ["fit", "tune_fit", "fit_tune"])
def test_export_records_the_overlay_the_fit_applied(
    lifecycle: str, tmp_path: Path
) -> None:
    """``fit_tune`` is the case the tuning block gets wrong: it carries an
    overlay, and the fit that produced the artifact applied none."""
    model = _run(_model(), lifecycle)
    out = model.export(tmp_path / "artifact")
    metadata = _metadata(out)

    assert KEY in metadata
    if lifecycle == "tune_fit":
        expected = metadata["tuning"]["best_training_params"]
        assert expected == {
            "early_stopping_rounds": TUNED_PATIENCE,
            "validation_ratio": TUNED_RATIO,
        }
    else:
        expected = {}
    assert metadata[KEY] == expected
    if lifecycle == "fit_tune":
        # The overlay the tuning block carries is not the one recorded.
        assert metadata["tuning"]["best_training_params"] != metadata[KEY]


@pytest.mark.parametrize(
    ("space", "space_mode"),
    [
        (SPACE, "replace"),
        # The default space's integer dimension for the patience, with the
        # caller's ratio merged in: the other way a value reaches the record.
        ({"validation_ratio": SPACE["validation_ratio"]}, "merge"),
    ],
    ids=["caller_categoricals", "default_int_dimension"],
)
def test_recorded_values_keep_their_types(
    space: dict[str, Any], space_mode: str, tmp_path: Path
) -> None:
    model = _run(_model(space=space, space_mode=space_mode), "tune_fit")
    out = model.export(tmp_path / "artifact")
    record = _metadata(out)[KEY]

    assert set(record) == {"early_stopping_rounds", "validation_ratio"}
    patience = record["early_stopping_rounds"]
    assert isinstance(patience, int) and not isinstance(patience, bool)
    assert isinstance(record["validation_ratio"], float)

    loaded = Model.load(out)
    assert loaded._get_fit_state().applied_training_params == record


@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
def test_a_loaded_model_reports_the_ratio_its_fit_applied(
    task: str, tmp_path: Path
) -> None:
    """#281's row: the fit trained with the tuned ratio, so both surfaces say so."""
    model = _run(_model(task), "tune_fit")
    assert _reported_ratio(model) == TUNED_RATIO

    loaded = Model.load(model.export(tmp_path / "artifact"))

    assert _reported_ratio(loaded) == TUNED_RATIO
    assert _exported_ratio(loaded) == TUNED_RATIO


def test_reexport_carries_the_record_forward(tmp_path: Path) -> None:
    model = _run(_model(), "tune_fit")
    first = model.export(tmp_path / "first")

    second = Model.load(first).export(tmp_path / "second")

    assert _metadata(second)[KEY] == _metadata(first)[KEY]
    assert _reported_ratio(Model.load(second)) == TUNED_RATIO


def _artifact_without_the_record(tmp_path: Path, lifecycle: str = "tune_fit") -> Path:
    """An artifact as written before H-0109, as far as ``load()`` can tell.

    Deleting the key leaves the key set an export wrote before H-0109
    (measured: ``pr8b_metadata_keys_before.txt``), and
    ``test_the_record_is_the_only_new_key`` checks the exporter adds nothing
    else. The values under the other keys come from the current exporter; the
    pickles are untouched by H-0109.
    """
    model = _run(_model(), lifecycle)
    out = model.export(tmp_path / "legacy")
    _rewrite_metadata(out, lambda metadata: metadata.pop(KEY, None))
    return out


#: The top-level keys an export wrote before H-0109, as measured at ``036cd18``
#: (``results/pr8b_metadata_keys_before.txt``).
LEGACY_KEYS = frozenset(
    {
        "checksums",
        "config",
        "feature_names",
        "format_version",
        "lizyml_version",
        "metrics",
        "python_version",
        "run_id",
        "task",
        "timestamp",
    }
)


@pytest.mark.parametrize("lifecycle", ["fit", "tune_fit"])
def test_the_record_is_the_only_new_key(lifecycle: str, tmp_path: Path) -> None:
    """Deleting the record is then a faithful pre-H-0109 key set."""
    out = _run(_model(), lifecycle).export(tmp_path / "artifact")
    tuning = {"tuning"} if lifecycle == "tune_fit" else set()

    assert set(_metadata(out)) == LEGACY_KEYS | tuning | {KEY}


def test_an_artifact_without_the_record_falls_back_to_the_config(
    tmp_path: Path,
) -> None:
    """The bound H-0094 decision 13 stated survives only here."""
    loaded = Model.load(_artifact_without_the_record(tmp_path))

    assert loaded._get_fit_state().applied_training_params is None
    assert _reported_ratio(loaded) == CONFIG_RATIO
    assert _exported_ratio(loaded) == CONFIG_RATIO


def test_an_unknown_record_is_not_rewritten_as_empty(tmp_path: Path) -> None:
    """Unknown is not "no overlay": re-exporting must not invent a record."""
    loaded = Model.load(_artifact_without_the_record(tmp_path))

    out = loaded.export(tmp_path / "again")

    assert KEY not in _metadata(out)


@pytest.mark.parametrize(
    "record",
    [
        ["validation_ratio", 0.45],
        {"seed": 1},
        {"validation_ratio": True},
        # The patience is an ``int``, and ``bool`` is an ``int`` subclass: this
        # case is refused by the bool clause alone (the ratio case above is
        # also outside (0, 1)).
        {"early_stopping_rounds": True},
        {"validation_ratio": "0.45"},
        {"validation_ratio": None},
        {"validation_ratio": math.nan},
        {"early_stopping_rounds": math.inf},
        # A fit records the patience as ``int()`` gives it.
        {"early_stopping_rounds": 2.5},
        # JSON integers are unbounded; ``math.isfinite(10**400)`` would raise
        # ``OverflowError`` rather than the deserialization error.
        {"validation_ratio": 10**400},
        # No inner-validation strategy accepts a ratio outside (0, 1), so no fit
        # can have applied one.
        {"validation_ratio": 0.0},
        {"validation_ratio": 1.0},
    ],
    ids=[
        "not_a_dict",
        "unknown_key",
        "bool_ratio",
        "bool_patience",
        "string",
        "null",
        "nan",
        "inf",
        "float_patience",
        "huge_int",
        "ratio_zero",
        "ratio_one",
    ],
)
def test_a_malformed_record_is_refused_on_load(record: Any, tmp_path: Path) -> None:
    """Accepting it would fail later, inside a report, as a raw error.

    The context names the path and the offending value's type, and the key
    when there is one.
    """
    out = _run(_model(), "fit").export(tmp_path / "artifact")

    def put(metadata: dict[str, Any]) -> None:
        metadata[KEY] = record

    _rewrite_metadata(out, put)

    with pytest.raises(LizyMLError) as excinfo:
        Model.load(out)
    assert excinfo.value.code == ErrorCode.DESERIALIZATION_FAILED
    assert KEY in excinfo.value.user_message
    context = excinfo.value.context
    assert context["path"] == str(out)
    if isinstance(record, dict):
        (name, value), *_ = record.items()
        assert context["key"] == name
        assert context["type"] == type(value).__name__
    else:
        assert context["type"] == type(record).__name__


@pytest.mark.parametrize(
    ("dimension", "choice", "recorded"),
    [
        ("validation_ratio", "0.45", 0.45),
        ("early_stopping_rounds", True, 1),
        ("early_stopping_rounds", 3.0, 3),
    ],
    ids=["string_ratio", "bool_patience", "float_patience"],
)
def test_the_record_holds_what_training_applied(
    dimension: str, choice: Any, recorded: Any, tmp_path: Path
) -> None:
    """Training converts each value as it reads it, and the record follows.

    Recording the raw choice let a categorical ``"0.45"`` or ``True`` -- both
    trained with -- reach an artifact ``Model.load()`` then refused (code review
    round 1). The record is the converted value, so the artifact loads.
    """
    space = {
        dimension: {"type": "categorical", "choices": [choice], "category": "training"}
    }
    model = _run(_model(space=space), "tune_fit")

    out = model.export(tmp_path / "artifact")

    record = _metadata(out)[KEY]
    assert record == {dimension: recorded}
    assert type(record[dimension]) is type(recorded)
    loaded = Model.load(out)
    assert loaded._get_fit_state().applied_training_params == {dimension: recorded}
    if dimension == "validation_ratio":
        assert _reported_ratio(loaded) == recorded


@pytest.mark.parametrize(
    ("source", "consumes_tuning"),
    [
        ("untuned_without_record", False),
        ("tuned_without_record", True),
        ("tuned_with_record", True),
    ],
)
def test_a_fit_after_load_records_its_own_overlay(
    source: str, consumes_tuning: bool, tmp_path: Path
) -> None:
    """A fit replaces the record, whatever the artifact said.

    The loaded model's tuning result is the one H-0086 restores from the tuning
    block, so a fit after load consumes it when the artifact was tuned, and
    applies no overlay when it was not -- including from an artifact whose
    record was unknown.
    """
    if source == "untuned_without_record":
        artifact = _artifact_without_the_record(tmp_path, "fit")
    elif source == "tuned_without_record":
        artifact = _artifact_without_the_record(tmp_path, "tune_fit")
    else:
        artifact = _run(_model(), "tune_fit").export(tmp_path / "artifact")
    loaded = Model.load(artifact)
    tuning = loaded._tuning_result
    assert (tuning is not None) is consumes_tuning
    expected = dict(tuning.best_training_params) if tuning is not None else {}

    loaded.fit(data=make_binary_df(n=200))

    assert loaded._get_fit_state().applied_training_params == expected
    assert _metadata(loaded.export(tmp_path / "refit"))[KEY] == expected


def _refused_fit(model: Model) -> None:
    """A fit a gate refuses before training: a regression objective on binary."""
    with pytest.raises(LizyMLError):
        model.fit(data=make_binary_df(n=90), params={"objective": "regression"})


def _failed_fit(model: Model) -> None:
    """A fit that raises mid-training."""
    with (
        mock.patch(
            "lizyml.training.cv_trainer.CVTrainer.fit",
            side_effect=RuntimeError("training blew up"),
        ),
        pytest.raises(RuntimeError),
    ):
        model.fit(data=make_binary_df(n=90))


@pytest.mark.parametrize("record", ["known", "unknown"])
@pytest.mark.parametrize("then", ["tune", "refused_fit", "failed_fit"])
def test_a_loaded_record_survives_calls_that_do_not_replace_the_fit(
    record: str, then: str, tmp_path: Path
) -> None:
    """Only a successful fit replaces the record (H-0094 decision 14's commit).

    A later ``tune()`` replaces the tuning result and leaves the fitted adapters
    alone; a refused or failed fit leaves the retained model in place. In each
    case the record -- known, or unknown -- must still describe the retained fit,
    and a re-export must still write it (or still omit it).
    """
    if record == "known":
        artifact = _run(_model(), "tune_fit").export(tmp_path / "artifact")
    else:
        artifact = _artifact_without_the_record(tmp_path)
    before = _metadata(artifact).get(KEY)
    loaded = Model.load(artifact)

    if then == "tune":
        loaded.tune(data=make_binary_df(n=200))
    elif then == "refused_fit":
        _refused_fit(loaded)
    else:
        _failed_fit(loaded)

    state = loaded._get_fit_state().applied_training_params
    out = _metadata(loaded.export(tmp_path / "after"))
    if record == "known":
        assert state == before
        assert out[KEY] == before
        assert _reported_ratio(loaded) == TUNED_RATIO
    else:
        assert state is None
        assert KEY not in out
        assert _reported_ratio(loaded) == CONFIG_RATIO
