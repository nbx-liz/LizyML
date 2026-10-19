"""H-0124 amendment 1: bounds and steps within what Optuna can sample.

Criteria 13 (the magnitude refusals of ``Tuner.tune`` and of the round
admission check; the two config surfaces are rows of ``test_space_format``'s
``REFUSED`` / ``ACCEPTED`` tables), 14 (an expansion past the limits is not
made, and is reported as not expanded) and 15 (a stored study that breaks
policy 9 is ``unreadable`` / ``inconsistent``, never a raw ``OverflowError``).
"""

from __future__ import annotations

import copy
import dataclasses
import math
import sys
from pathlib import Path
from typing import Any

import optuna
import pytest

from lizyml import Model
from lizyml.core import _tuning_plan
from lizyml.core import magnitude as _magnitude
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.search_dim import FloatDim, IntDim, SearchDim
from lizyml.tuning._step_grid import check_dim_magnitude, check_dim_step
from lizyml.tuning.lineage import STUDY_KEY, classify_study, round_key
from lizyml.tuning.lineage_codec import LineageCodecError, decode_dim, encode_dim
from lizyml.tuning.search_space import (
    detect_boundary,
    expand_dims,
    parse_space,
    suggest_params,
)
from lizyml.tuning.tuner import Tuner
from tests.test_tuning._resume_helpers import (
    fast_config,
    frame,
    load,
    snapshot_study,
    url,
)

CI = ErrorCode.CONFIG_INVALID
MAX = sys.float_info.max
LIMIT = 2**52

# ---------------------------------------------------------------------------
# Criterion 13: parse_space, Tuner.tune and the round admission check
# ---------------------------------------------------------------------------

#: The criterion-13 specs: (spec, key, reason). Both surfaces are run through
#: ``Model.tune`` in ``test_space_format``; here ``parse_space`` alone, since
#: the round admission check would refuse the config surface's space too.
PARSE_REFUSED: dict[str, tuple[dict[str, Any], str, str]] = {
    "float_width": (
        {"type": "float", "low": -1e308, "high": 1e308},
        "high",
        "range_not_finite",
    ),
    "float_width_step": (
        {"type": "float", "low": -1.5e308, "high": 1e308, "step": 5e307},
        "high",
        "range_not_finite",
    ),
    "int_high": (
        {"type": "int", "low": 0, "high": LIMIT + 1},
        "high",
        "bound_too_large",
    ),
    "int_low": (
        {"type": "int", "low": -LIMIT - 1, "high": 0},
        "low",
        "bound_too_large",
    ),
    "int_2_60": (
        {"type": "int", "low": 2**60, "high": 2**61},
        "low",
        "bound_too_large",
    ),
    "int_10_309": (
        {"type": "int", "low": 10**309, "high": 2 * 10**309, "step": 10**309},
        "low",
        "bound_too_large",
    ),
    "int_step": (
        {"type": "int", "low": 0, "high": LIMIT, "step": 2**53},
        "step",
        "bound_too_large",
    ),
}


@pytest.mark.parametrize("surface", ["tuning.optuna.space", "tune(space=)"])
@pytest.mark.parametrize("case", list(PARSE_REFUSED))
def test_parse_space_refuses_past_the_magnitude_limit(case: str, surface: str) -> None:
    spec, key, reason = PARSE_REFUSED[case]
    with pytest.raises(LizyMLError) as caught:
        parse_space({"d": spec}, surface=surface)
    err = caught.value
    assert err.code is CI
    assert {k: err.context[k] for k in ("surface", "reason", "dim", "key")} == {
        "surface": surface,
        "reason": reason,
        "dim": "d",
        "key": key,
    }


TUNER_REFUSED: dict[str, tuple[SearchDim, str, str]] = {
    "float_width_no_step": (FloatDim("x", -1e308, 1e308), "high", "range_not_finite"),
    "float_width_step": (
        FloatDim("x", -1.5e308, 1e308, step=5e307),
        "high",
        "range_not_finite",
    ),
    "int_high_no_step": (IntDim("n", 0, 2**53), "high", "bound_too_large"),
    "int_low_no_step": (IntDim("n", -(2**52) - 1, 0), "low", "bound_too_large"),
    "int_both_no_step": (IntDim("n", 2**60, 2**61), "low", "bound_too_large"),
    "int_step": (IntDim("n", 0, LIMIT, step=2**53), "step", "bound_too_large"),
    # low is reported before step when both are past the limit (policy 9).
    "int_low_and_step": (
        IntDim("n", 2**53, 2**54, step=2**53),
        "low",
        "bound_too_large",
    ),
    # int-typed float bounds: the width Optuna computes is float(high) -
    # float(low), which overflows although the exact width is the float max.
    "float_width_int_bounds": (
        FloatDim("x", -(2**970 - 1), int(sys.float_info.max) - 2**970 + 1),  # type: ignore[arg-type]
        "high",
        "range_not_finite",
    ),
    "float_low_not_finite": (
        FloatDim("x", float("-inf"), 1.0),
        "low",
        "bound_not_finite",
    ),
    # Text is read as a number only from a mapping; a dim built in Python
    # would hand it to Optuna, which samples 2**53.. coarsely without a word.
    "int_text_no_step": (
        IntDim("n", "9007199254740992", "9007199254740998"),  # type: ignore[arg-type]
        "low",
        "bound_not_a_number",
    ),
}


def _never_called(trial: Any) -> float:
    raise AssertionError("the objective ran")


@pytest.mark.parametrize("case", list(TUNER_REFUSED))
def test_tuner_tune_refuses_past_the_magnitude_limit(case: str, tmp_path: Path) -> None:
    """Policy 9 on every numeric dim, stepped or not, before a study exists."""
    dim, key, reason = TUNER_REFUSED[case]
    storage = url(tmp_path)
    tuner = Tuner(dims=[dim], n_trials=1, storage=storage, study_name="s")
    with pytest.raises(LizyMLError) as caught:
        tuner.tune(_never_called)
    err = caught.value
    assert err.code is CI
    assert {k: err.context[k] for k in ("surface", "reason", "dim", "key")} == {
        "surface": "Tuner.tune",
        "reason": reason,
        "dim": dim.name,
        "key": key,
    }
    assert optuna.get_all_study_summaries(storage) == []


@pytest.mark.parametrize(
    "dim",
    [
        FloatDim("x", -8e307, 8e307),
        FloatDim("x", 0.0, MAX),
        IntDim("n", LIMIT - 10, LIMIT),
        IntDim("n", -LIMIT, -LIMIT + 10),
        IntDim("n", 0, LIMIT, step=LIMIT),
    ],
    ids=["float_8e307", "float_width_max", "int_limit", "int_neg_limit", "step_limit"],
)
def test_tuner_tune_samples_at_the_limit(dim: SearchDim) -> None:
    """At the limit Optuna samples the range, and the check lets it through."""
    seen: list[Any] = []

    def objective(trial: Any) -> float:
        seen.append(suggest_params(trial, [dim])[dim.name])
        return 0.0

    Tuner(dims=[dim], n_trials=3, seed=0).tune(objective)
    assert len(seen) == 3
    assert all(dim.low <= v <= dim.high for v in seen)


def test_a_dim_built_in_python_refuses_text_bounds() -> None:
    """Policy 9 covers every numeric dim built in Python, stepped or not.

    Text is read as a number only in a config or ``space=`` mapping; on a dim
    built in Python it would reach Optuna as it is.
    """
    with pytest.raises(LizyMLError) as caught:
        check_dim_magnitude(FloatDim("x", "0.1", 1.0), surface="Tuner.tune")  # type: ignore[arg-type]
    assert (caught.value.context["reason"], caught.value.context["key"]) == (
        "bound_not_a_number",
        "low",
    )


def _space(**space: Any) -> dict[str, Any]:
    cfg = fast_config(n_trials=1)
    cfg["tuning"]["optuna"]["space"] = space
    return cfg


@pytest.mark.parametrize(
    ("name", "edit", "key", "reason"),
    [
        ("learning_rate", {"low": -1e308, "high": 1e308}, "high", "range_not_finite"),
        ("num_leaves", {"high": 2**53}, "high", "bound_too_large"),
    ],
    ids=["float_width", "int_magnitude"],
)
def test_round_admission_refuses_an_effective_space_past_the_limit(
    name: str,
    edit: dict[str, Any],
    key: str,
    reason: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An expansion that broke policy 9 is refused before the round record."""
    storage = url(tmp_path)
    model = Model(
        _space(
            num_leaves={"type": "int", "low": 4, "high": 8},
            learning_rate={"type": "float", "low": 0.1, "high": 0.5},
        ),
        data=frame(),
    )
    model.tune(storage=storage, study_name="s")
    real = _tuning_plan.expand_for_next_round

    def broken(*args: Any, **kwargs: Any) -> Any:
        dims, report = real(*args, **kwargs)
        return (
            tuple(
                dataclasses.replace(d, **edit) if d.name == name else d for d in dims
            ),
            report,
        )

    monkeypatch.setattr(_tuning_plan, "expand_for_next_round", broken)
    before = snapshot_study(storage, "s")
    with pytest.raises(LizyMLError) as caught:
        model.tune(
            resume=True,
            storage=storage,
            study_name="s",
            n_trials=1,
            expand_boundary=True,
        )
    err = caught.value
    assert err.code is CI
    assert (err.context["reason"], err.context["dim"], err.context["key"]) == (
        reason,
        name,
        key,
    )
    assert snapshot_study(storage, "s") == before
    assert round_key(2) not in load(storage, "s").user_attrs


# ---------------------------------------------------------------------------
# Criterion 14: an expansion past the limits is not made
# ---------------------------------------------------------------------------

LOW_A, HIGH_A = 2.218281e287, 2.9098924049999997e307
LOW_B, HIGH_B = 1.9120500772236246e307, 2.48319537762e307

#: id -> (dim, best value). Each edge's candidate breaks one policy-10 limit.
NOT_EXPANDED: dict[str, tuple[SearchDim, Any]] = {
    "log_upper_past_float": (FloatDim("x", 1e307, 1e308, log=True), 1e308),
    "linear_upper_past_float": (FloatDim("x", 0.0, 1e308), 1e308),
    "linear_upper_edge_and_width": (FloatDim("x", -1e308, 5e307), 5e307),
    "width_past_limit_under_bound": (
        FloatDim("x", -1e308, 0.0, max_allowed=1.5e308),
        0.0,
    ),
    "int_upper_past_limit": (IntDim("n", 2**51, LIMIT), LIMIT),
    "int_step_upper_past_limit": (IntDim("n", 3, 3 * 2**50, step=3), 3 * 2**50),
    "int_upper_past_limit_under_bound": (
        IntDim("n", 2**51, LIMIT, max_allowed=2**60),
        LIMIT,
    ),
    "log_step_upper_past_float": (
        FloatDim("x", 1e307, 1e308, log=True, step=1e307),
        1e308,
    ),
    "step_upper_past_float": (FloatDim("x", 0.0, 9e307, step=3e281), 9e307),
    "step_upper_off_grid_as_float": (
        FloatDim("x", LOW_A, HIGH_A, step=3e281),
        HIGH_A,
    ),
    "log_lower_rounds_to_zero": (FloatDim("x", 5e-324, 1e-323, log=True), 5e-324),
    "step_lower_off_grid_as_float": (
        FloatDim("x", LOW_B, HIGH_B, step=3e281),
        LOW_B,
    ),
    "bound_then_width_past_limit": (
        FloatDim("x", -1e308, 0.0, max_allowed=9e307),
        0.0,
    ),
    "int_candidate_past_limit_under_bound": (
        IntDim("n", 2**51, 7 * 2**49, max_allowed=2**60),
        7 * 2**49,
    ),
}

#: id -> (dim, best value, new range, clamped_to_bound).
EXPANDED: dict[str, tuple[SearchDim, Any, tuple[Any, Any], bool]] = {
    "bound_then_width_inside": (
        FloatDim("x", -1e308, 0.0, max_allowed=5e307),
        0.0,
        (-1e308, 5e307),
        True,
    ),
    "int_bound_at_limit": (
        IntDim("n", 2**51, 7 * 2**49, max_allowed=LIMIT),
        7 * 2**49,
        (2**51, LIMIT),
        True,
    ),
    "linear_width_inside": (FloatDim("x", 0.0, 5e307), 5e307, (0.0, 1e308), False),
    "step_upper_on_grid": (
        FloatDim("x", 0.0, 9e306, step=3e281),
        9e306,
        (0.0, 1.8e307),
        False,
    ),
}


def _admitted(dim: SearchDim) -> None:
    """Policy 2 and policy 9 hold, as the round admission check reads them."""
    check_dim_magnitude(dim, surface="test")
    check_dim_step(dim, surface="test")


def _detects_finitely(dim: SearchDim) -> None:
    """The next round's edge detection is finite and raises nothing."""
    for best in (dim.low, dim.high):
        (status,) = detect_boundary([dim], {dim.name: best}).dims
        assert status.position_pct is not None
        assert math.isfinite(status.position_pct)


def test_the_cases_are_inside_policy_9() -> None:
    for dim, _ in NOT_EXPANDED.values():
        _admitted(dim)
    for dim, *_ in EXPANDED.values():
        _admitted(dim)


@pytest.mark.parametrize("case", list(NOT_EXPANDED))
def test_an_edge_past_the_limits_is_not_expanded(case: str) -> None:
    dim, best = NOT_EXPANDED[case]
    report = detect_boundary([dim], {dim.name: best})
    (status,) = report.dims
    assert status.edge == ("lower" if best == dim.low else "upper")
    assert status.position_pct is not None
    assert status.expanded is False
    assert (status.new_low, status.new_high) == (None, None)
    assert status.clamped_to_bound is False
    assert report.expanded_names == ()
    (same,) = expand_dims([dim], report)
    assert same is dim
    _detects_finitely(same)


@pytest.mark.parametrize("case", list(EXPANDED))
def test_an_edge_inside_the_limits_is_expanded(case: str) -> None:
    dim, best, new_range, clamped = EXPANDED[case]
    report = detect_boundary([dim], {dim.name: best})
    (status,) = report.dims
    assert status.expanded is True
    assert (status.new_low, status.new_high) == new_range
    assert status.clamped_to_bound is clamped
    assert report.expanded_names == (dim.name,)
    (wider,) = expand_dims([dim], report)
    assert (wider.low, wider.high) == new_range
    assert wider.low <= dim.low and wider.high >= dim.high
    assert wider.step == dim.step
    _admitted(wider)
    _detects_finitely(wider)


def test_the_limit_is_checked_after_the_bound_clamps() -> None:
    """The candidate 1e308 breaks the width limit; clamped to 5e307 it does not."""
    unclamped = FloatDim("x", -1e308, 0.0)
    (status,) = detect_boundary([unclamped], {"x": 0.0}).dims
    assert status.expanded is False
    bounded = dataclasses.replace(unclamped, max_allowed=5e307)
    (status,) = detect_boundary([bounded], {"x": 0.0}).dims
    assert (status.expanded, status.new_high) == (True, 5e307)


def test_a_clamp_back_to_the_old_edge_keeps_the_h0111_report() -> None:
    """Not a policy-10 veto: the edge sits on the bound (H-0111, #342)."""
    dim = FloatDim("ff", 0.5, 1.0, max_allowed=1.0)
    (status,) = detect_boundary([dim], {"ff": 0.99}).dims
    assert (status.expanded, status.clamped_to_bound) == (False, True)


# ---------------------------------------------------------------------------
# Criterion 15: a stored study that breaks policy 9
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dim",
    [
        IntDim("n", 0, 10**400),
        IntDim("n", -(LIMIT + 1), 0),
        IntDim("n", 0, LIMIT, step=LIMIT + 1),
        FloatDim("x", -1e308, 1e308),
        FloatDim("x", 0.0, float("inf")),
        FloatDim("x", float("nan"), 1.0),
        FloatDim("x", 0.0, 1.0, step=float("inf")),
        FloatDim("x", 0.0, 1.0, min_allowed=-(10**400)),
        FloatDim("x", 0.0, 1.0, max_allowed=float("nan")),
        FloatDim("x", 0.0, 1.0, max_allowed=float("inf")),
        IntDim("n", 0, 10, min_allowed=-(10**400)),
    ],
    ids=[
        "int_high",
        "int_low",
        "int_step",
        "float_width",
        "float_inf",
        "float_nan",
        "float_step_inf",
        "min_allowed_huge",
        "max_allowed_nan",
        "max_allowed_inf",
        "int_min_allowed_huge",
    ],
)
def test_decode_refuses_a_dim_past_policy_9(dim: SearchDim) -> None:
    with pytest.raises(LineageCodecError):
        decode_dim(encode_dim(dim))


@pytest.mark.parametrize(
    "dim",
    [
        IntDim("n", -LIMIT, LIMIT, step=LIMIT),
        IntDim("n", 2**51, LIMIT, max_allowed=2**60),
        FloatDim("x", 0.0, MAX, min_allowed=-MAX, max_allowed=MAX),
    ],
    ids=["int_limit", "int_bound_past_limit", "float_width_max"],
)
def test_decode_reads_a_dim_inside_policy_9(dim: SearchDim) -> None:
    """``min_allowed`` / ``max_allowed`` need only be finite floats."""
    assert decode_dim(encode_dim(dim)) == dim


HUGE = {"type": "int", "low": 2, "high": 10**400}


@pytest.fixture
def huge_study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A study tuned with the code before this amendment: a 10**400 int dim.

    Lifting the limit restores that code's reading of the space; Optuna then
    fails every sample (``int too large to convert to float``), and the study
    keeps the records that declare the dim.
    """
    storage = url(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(_magnitude, "INT_LIMIT", 10**500)
        cfg = _space(num_leaves=dict(HUGE))
        with pytest.raises(LizyMLError) as failed:
            Model(cfg, data=frame()).tune(storage=storage, study_name="s")
        assert failed.value.code is ErrorCode.TUNING_FAILED
    return storage


def _refused(storage: str, state: str) -> None:
    """``tune(resume=True)`` and ``restore_tuning`` refuse with H-0122's state.

    Run with an in-limit config (the user's space now passes policy 9), so it
    is the stored record that is refused. The study's own space is refused
    by policy 9 before any study is read (``bound_too_large``).
    """
    fixed = _space(num_leaves={"type": "int", "low": 4, "high": 8})
    for call in (
        lambda m: m.tune(resume=True, storage=storage, study_name="s"),
        lambda m: m.restore_tuning(storage, "s"),
    ):
        with pytest.raises(LizyMLError) as caught:
            call(Model(copy.deepcopy(fixed), data=frame()))
        assert caught.value.code is CI
        assert caught.value.context["state"] == state
    with pytest.raises(LizyMLError) as caught:
        Model(_space(num_leaves=dict(HUGE)), data=frame()).tune(
            resume=True, storage=storage, study_name="s"
        )
    assert caught.value.context["reason"] == "bound_too_large"


def test_a_stored_study_with_a_huge_int_dim_is_unreadable(huge_study: str) -> None:
    study = load(huge_study, "s")
    lineage = classify_study(study.user_attrs, study.trials)
    assert lineage.state == "unreadable"
    assert "num_leaves" in lineage.reason
    _refused(huge_study, "unreadable")


@pytest.mark.parametrize("name", ["feature_fraction", "max_bin"])
def test_a_round_record_with_a_huge_min_allowed_is_inconsistent(
    name: str, tmp_path: Path
) -> None:
    storage = url(tmp_path)
    cfg = _space(
        num_leaves={"type": "int", "low": 4, "high": 8},
        feature_fraction={"type": "float", "low": 0.5, "high": 0.9},
        max_bin={"type": "int", "low": 16, "high": 32},
    )
    Model(copy.deepcopy(cfg), data=frame()).tune(storage=storage, study_name="s")
    study = load(storage, "s")
    record = copy.deepcopy(study.user_attrs[round_key(1)])
    (dim,) = [d for d in record["effective"] if d["name"] == name]
    assert dim["min_allowed"][0] in ("int", "float")
    dim["min_allowed"] = ["int", -(10**400)]
    study.set_user_attr(round_key(1), record)
    study = load(storage, "s")
    lineage = classify_study(study.user_attrs, study.trials)
    assert lineage.state == "inconsistent"
    assert name in lineage.reason
    assert STUDY_KEY in study.user_attrs
    for call in (
        lambda m: m.tune(
            resume=True, storage=storage, study_name="s", expand_boundary=True
        ),
        lambda m: m.restore_tuning(storage, "s"),
    ):
        with pytest.raises(LizyMLError) as caught:
            call(Model(copy.deepcopy(cfg), data=frame()))
        assert caught.value.code is CI
        assert caught.value.context["state"] == "inconsistent"
