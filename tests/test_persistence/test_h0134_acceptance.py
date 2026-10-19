"""H-0134 acceptance: the tuning history survives export and load (#315).

The criteria are numbered as in HISTORY.md H-0134 "受け入れ基準":

- 1: the load census, ``_load_census.DECLARED`` being empty.
- 2: history-less artifacts.
- 3: every shape the writers can produce.
- 4: strict reading.
- 6: a stale file.
- 8: the public ``persistence.export``.
- 5 and 7: ``test_mixin_state_isolation.py`` and ``test_artifact_compat.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.tuning_codec import (
    HISTORY_FILE,
    HISTORY_SCHEMA,
    TRIAL_STATES,
    TuningHistory,
    history_bytes,
)
from lizyml.core.types.search_dim import CategoricalDim, FloatDim, IntDim
from lizyml.core.types.tuning_result import (
    BoundaryDimStatus,
    BoundaryReport,
    RoundSummary,
    TrialResult,
    TuningResult,
)
from lizyml.persistence import export as persistence_export
from lizyml.persistence import load as persistence_load
from tests._tuning_compare import same_value
from tests.test_persistence._artifact_versions import (
    to_v1_shape,
    to_v2_shape,
    to_v3_shape,
)
from tests.test_persistence._load_census import build

SURFACES = ("tuning_table", "tuning_plot", "boundary_table")
NAN = float("nan")


@pytest.fixture(autouse=True)
def _quiet() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _readings(model: Model) -> dict[str, Any]:
    """Each surface's output, or the code it raised (P7: the outputs)."""
    out: dict[str, Any] = {}
    for surface in SURFACES:
        try:
            value = getattr(model, surface)()
        except LizyMLError as exc:
            out[surface] = ("raises", exc.code, exc.context.get("missing"))
            continue
        out[surface] = value.to_json() if surface == "tuning_plot" else value
    return out


def _assert_same_readings(before: dict[str, Any], after: dict[str, Any]) -> None:
    for surface in SURFACES:
        a, b = before[surface], after[surface]
        if isinstance(a, pd.DataFrame):
            assert isinstance(b, pd.DataFrame), surface
            pd.testing.assert_frame_equal(a, b, check_exact=True)
        else:
            assert a == b, surface


@pytest.fixture(scope="module")
def resumed(tmp_path_factory: pytest.TempPathFactory) -> tuple[Model, Path]:
    """A real tune, a resumed round with a boundary report, a fit and an export."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, _ = build("regression")
        model.tune()
        model.tune(resume=True, expand_boundary=True)
        model.fit()
        return model, model.export(tmp_path_factory.mktemp("h0134") / "a")


def _copy(artifact: Path, dest: Path) -> Path:
    dest.mkdir()
    for f in artifact.iterdir():
        (dest / f.name).write_bytes(f.read_bytes())
    return dest


def _meta(artifact: Path) -> dict[str, Any]:
    meta: dict[str, Any] = json.loads((artifact / "metadata.json").read_text())
    return meta


def _write_meta(artifact: Path, meta: dict[str, Any]) -> None:
    (artifact / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")


# ---------------------------------------------------------------------------
# Criterion 1 (real tune): the surfaces answer as before export
# ---------------------------------------------------------------------------


def test_a_resumed_tune_answers_the_same_after_load(
    resumed: tuple[Model, Path],
) -> None:
    model, artifact = resumed
    loaded = Model.load(artifact)
    before, after = _readings(model), _readings(loaded)
    assert isinstance(before["boundary_table"], pd.DataFrame)  # a reading to lose
    _assert_same_readings(before, after)
    assert same_value(loaded._tuning_result, model._tuning_result)


# ---------------------------------------------------------------------------
# Criterion 2: no history -> MODEL_NOT_FIT naming it; the overlay still works
# ---------------------------------------------------------------------------

SHAPES: dict[str, Callable[[Path], None]] = {
    "v1": to_v1_shape,
    "v2": to_v2_shape,
    "v3": to_v3_shape,
    "v4 null": lambda a: _write_meta(
        a, {**_meta(a), "tuning": {**_meta(a)["tuning"], "history": None}}
    ),
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_an_artifact_without_history_says_so_on_every_surface(
    shape: str, resumed: tuple[Model, Path], tmp_path: Path
) -> None:
    model, artifact = resumed
    copy = _copy(artifact, tmp_path / "a")
    SHAPES[shape](copy)
    loaded = Model.load(copy)
    for surface in SURFACES:
        with pytest.raises(LizyMLError) as info:
            getattr(loaded, surface)()
        assert info.value.code == ErrorCode.MODEL_NOT_FIT
        assert info.value.context == {
            "method": surface,
            "missing": "tuning_history",
            "loaded_from_artifact": True,
        }
        assert "restore_tuning(storage=" in info.value.user_message
    original = model._tuning_result
    assert original is not None and loaded._tuning_result is not None
    assert loaded._tuning_result.best_model_params == original.best_model_params


def test_reexporting_a_history_less_model_keeps_it_unknown(
    resumed: tuple[Model, Path], tmp_path: Path
) -> None:
    _, artifact = resumed
    copy = _copy(artifact, tmp_path / "a")
    to_v3_shape(copy)
    once = Model.load(copy).export(tmp_path / "b")
    twice = Model.load(once).export(tmp_path / "c")
    for out in (once, twice):
        assert _meta(out)["tuning"]["history"] is None
        assert not (out / HISTORY_FILE).exists()
        with pytest.raises(LizyMLError):
            Model.load(out).tuning_table()


def test_a_history_less_load_refits_with_the_tuned_overlay(
    resumed: tuple[Model, Path], tmp_path: Path
) -> None:
    model, artifact = resumed
    copy = _copy(artifact, tmp_path / "a")
    to_v3_shape(copy)
    loaded = Model.load(copy)
    loaded.fit(data=build("regression")[0]._data)
    assert loaded._applied_training_params == model._applied_training_params


# ---------------------------------------------------------------------------
# Criterion 3: every shape the writers can produce
# ---------------------------------------------------------------------------


def _every_shape() -> TuningResult:
    space = (
        FloatDim("learning_rate", 0.01, 0.3, log=True, category="model"),
        IntDim("max_depth", 3, 9, category="model", step=2),
        CategoricalDim("pick", (None, True, 1, 1.5, "s", NAN), category="model"),
    )
    trials = [
        # complete, with every scalar type; params in one insertion order
        TrialResult(
            0, {"max_depth": 5, "learning_rate": 0.1, "pick": "s"}, 0.5, "complete", 1
        ),
        # NaN in a param and in the score, a None param, another order
        TrialResult(1, {"pick": None, "learning_rate": NAN}, NAN, "fail", 1),
        TrialResult(2, {"pick": True, "max_depth": 3}, 0.7, "pruned", 2),
        TrialResult(3, {"pick": 1, "learning_rate": math.inf}, NAN, "running", 3),
        TrialResult(4, {"pick": 1.5, "learning_rate": -math.inf}, NAN, "waiting", 4),
    ]
    rounds = (
        # round 1 asked for 3 trials and ran 2; round 3 failed, so it has no
        # summary; round 2 expanded max_depth
        RoundSummary(1, 2, None, 0.5, (), space),
        RoundSummary(2, 1, 0.5, 0.5, ("max_depth",), space),
        RoundSummary(4, 1, 0.5, 0.5, (), space),
    )
    report = BoundaryReport(
        dims=(
            BoundaryDimStatus("max_depth", 9, 3, 9, 1.0, "upper", True, 3, 13, True),
            BoundaryDimStatus("pick", "s", None, None, None, "none", False, None, None),
        ),
        expanded_names=("max_depth",),
    )
    return TuningResult(
        best_model_params={"max_depth": 5, "learning_rate": 0.1},
        best_smart_params={},
        best_training_params={},
        best_score=0.5,
        trials=trials,
        metric_name="rmse",
        direction="minimize",
        rounds=rounds,
        boundary_report=report,
    )


def test_every_shape_answers_the_same_after_load(tmp_path: Path) -> None:
    model, _ = build("regression")
    model.fit()
    model._tuning_result = _every_shape()
    assert {t.state for t in model._tuning_result.trials} == TRIAL_STATES
    before = _readings(model)
    assert all(isinstance(v, pd.DataFrame | str) for v in before.values())
    loaded = Model.load(model.export(tmp_path / "a"))
    _assert_same_readings(before, _readings(loaded))
    assert same_value(loaded._tuning_result, model._tuning_result)
    assert list(loaded.tuning_table().columns) == list(model.tuning_table().columns)


# ---------------------------------------------------------------------------
# Criterion 4: strict reading
# ---------------------------------------------------------------------------


def _resign(artifact: Path, raw: bytes) -> None:
    """Write *raw* as the history and record its digest, so the codec is reached."""
    (artifact / HISTORY_FILE).write_bytes(raw)
    meta = _meta(artifact)
    meta["checksums"]["files"][HISTORY_FILE] = hashlib.sha256(raw).hexdigest()
    _write_meta(artifact, meta)


def _payload_edit(fn: Callable[[dict[str, Any]], None]) -> Callable[[Path], None]:
    def apply(artifact: Path) -> None:
        payload = json.loads((artifact / HISTORY_FILE).read_bytes())
        fn(payload)
        _resign(artifact, json.dumps(payload).encode("utf-8"))

    return apply


def _decl(value: Any) -> Callable[[Path], None]:
    def apply(artifact: Path) -> None:
        meta = _meta(artifact)
        meta["tuning"]["history"] = value
        _write_meta(artifact, meta)

    return apply


def _drop_history_key(artifact: Path) -> None:
    meta = _meta(artifact)
    del meta["tuning"]["history"]
    _write_meta(artifact, meta)


def _unlist(artifact: Path) -> None:
    meta = _meta(artifact)
    del meta["checksums"]["files"][HISTORY_FILE]
    _write_meta(artifact, meta)


def _flip_byte(artifact: Path) -> None:
    raw = bytearray((artifact / HISTORY_FILE).read_bytes())
    raw[-2] ^= 1
    (artifact / HISTORY_FILE).write_bytes(bytes(raw))


GOOD = {"file": HISTORY_FILE, "schema": HISTORY_SCHEMA}
BROKEN: dict[str, Callable[[Path], None]] = {
    "payload unknown schema": _payload_edit(lambda p: p.update(schema="other/v1")),
    "payload extra key": _payload_edit(lambda p: p.update(extra=1)),
    "payload unknown state": _payload_edit(
        lambda p: p["trials"][0].update(state="cancelled")
    ),
    "payload extra report key": _payload_edit(
        lambda p: p["boundary_report"].update(extra=1)
    ),
    "payload not json": lambda a: _resign(a, b"{not json"),
    "digest mismatch": _flip_byte,
    "declared file missing": lambda a: (a / HISTORY_FILE).unlink(),
    "file not in checksums": _unlist,
    "checksums not an object": lambda a: _write_meta(a, {**_meta(a), "checksums": []}),
    "checksums other algorithm": lambda a: _write_meta(
        a, {**_meta(a), "checksums": {**_meta(a)["checksums"], "algorithm": "md5"}}
    ),
    "format 4 without history key": _drop_history_key,
    "declaration is a list": _decl([HISTORY_FILE]),
    "declaration is a string": _decl(HISTORY_FILE),
    "declaration is a number": _decl(1),
    "declaration missing file": _decl({"schema": HISTORY_SCHEMA}),
    "declaration missing schema": _decl({"file": HISTORY_FILE}),
    "declaration extra key": _decl({**GOOD, "extra": 1}),
    "declaration other file": _decl({**GOOD, "file": "other.json"}),
    "declaration path escape": _decl({**GOOD, "file": "../tuning_history.json"}),
    "declaration other schema": _decl({**GOOD, "schema": "lizyml-tuning-history/v2"}),
}


@pytest.mark.parametrize("case", sorted(BROKEN))
def test_a_broken_history_or_declaration_is_refused(
    case: str, resumed: tuple[Model, Path], tmp_path: Path
) -> None:
    _, artifact = resumed
    copy = _copy(artifact, tmp_path / "a")
    BROKEN[case](copy)
    with pytest.raises(LizyMLError) as info:
        Model.load(copy)
    assert info.value.code == ErrorCode.DESERIALIZATION_FAILED


# ---------------------------------------------------------------------------
# Criterion 6: an undeclared file in a reused directory is not read
# ---------------------------------------------------------------------------


def test_a_stale_history_file_is_not_read(
    resumed: tuple[Model, Path], tmp_path: Path
) -> None:
    _, artifact = resumed
    reused = _copy(artifact, tmp_path / "reused")
    old = _copy(artifact, tmp_path / "old")
    to_v3_shape(old)
    Model.load(old).export(reused)  # writes "history": null over the old export
    assert (reused / HISTORY_FILE).exists()
    assert _meta(reused)["tuning"]["history"] is None
    with pytest.raises(LizyMLError) as info:
        Model.load(reused).tuning_table()
    assert info.value.context["missing"] == "tuning_history"


# ---------------------------------------------------------------------------
# Criterion 8: the public persistence.export writes the history too
# ---------------------------------------------------------------------------


def test_public_export_writes_the_history(
    resumed: tuple[Model, Path], tmp_path: Path
) -> None:
    model, _ = resumed
    state = model._get_fit_state()
    assert state.refit_result is not None and state.tuning_result is not None
    persistence_export(
        tmp_path / "p",
        state.fit_result,
        state.refit_result,
        state.cfg.model_dump(),
        state.cfg.task,
        tuning=state.tuning_result,
    )
    history = persistence_load(tmp_path / "p")[2]["tuning"]["history"]
    tr = state.tuning_result
    expected = TuningHistory(tr.trials, tr.rounds, tr.boundary_report)
    assert isinstance(history, TuningHistory)
    assert same_value(history, expected)
    raw = (tmp_path / "p" / HISTORY_FILE).read_bytes()
    assert raw == history_bytes(expected)
