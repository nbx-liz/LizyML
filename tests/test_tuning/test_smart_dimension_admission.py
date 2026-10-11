"""H-0121 (#299): smart search dimensions that nothing consumes.

A ``category: smart`` dimension reaches training only through smart
resolution, which reads a smart parameter only when it is declared and, for a
parameter with an owner, only while the owner is on. ``num_leaves_ratio`` is
read only when ``auto_num_leaves`` is true. Before H-0121 such a dimension was
sampled, recorded in ``best_params`` and consumed by nothing.

* Decision 3: a user-written dimension whose owner cannot be true in this tune
  is refused before the study (``CONFIG_INVALID``).
* Decision 4: the same dimension coming from the provider's default space is
  dropped from the space, with an INFO log naming it.

Both apply only to a tune that creates a new study (an absent name, or a
stored study in state E/N, H-0122 §7.4). ``resume=False`` onto a stored study
is refused (H-0122 criterion 9).

Decision 2 (undeclared smart names) is covered by
``test_search_space_name_validation.py``, whose strict xfails H-0121 flips.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import patch

import optuna
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.search_dim import CategoricalDim, FloatDim, IntDim
from lizyml.estimators.lgbm.provider import LGBMProvider
from lizyml.estimators.lgbm.smart_params import resolve_smart_params
from tests._helpers import make_binary_df, make_config
from tests._train_spy import record_lightgbm_calls

RATIO = "num_leaves_ratio"
OWNER = "auto_num_leaves"

_RATIO_DIM = {"type": "float", "low": 0.1, "high": 0.9, "category": "smart"}
#: Keeps merge-mode tunes cheap: the default ``n_estimators`` range is 600-2500.
_SMALL_TREES = {"type": "int", "low": 3, "high": 5, "category": "model"}


def _config(
    *,
    owner: bool,
    space: dict[str, Any],
    mode: str = "merge",
    n_trials: int = 2,
) -> dict[str, Any]:
    cfg = make_config("binary", n_estimators=3, n_splits=2, tuning_n_trials=n_trials)
    cfg["model"][OWNER] = owner
    cfg["tuning"]["optuna"]["space_mode"] = mode
    cfg["tuning"]["optuna"]["space"] = space
    return cfg


def _assert_refused_before_study(cfg: dict[str, Any]) -> LizyMLError:
    with (
        patch("optuna.create_study") as study,
        record_lightgbm_calls() as seen,
        pytest.raises(LizyMLError) as exc,
    ):
        Model(cfg, data=make_binary_df(n=120)).tune()
    study.assert_not_called()
    assert not seen["train_params"]
    err = exc.value
    assert err.code is ErrorCode.CONFIG_INVALID
    for name in (RATIO, OWNER, "tuning.optuna.space"):
        assert name in err.user_message, err.user_message
    return err


def _sampled_names(model: Model) -> set[str]:
    assert model._tuning_result is not None
    names = set(model._tuning_result.best_params)
    for trial in model._tuning_result.trials:
        names |= set(trial.params)
    return names


# ---------------------------------------------------------------------------
# Decision 3: user-written dimensions (criterion 5)
# ---------------------------------------------------------------------------


def test_replace_space_ratio_with_owner_off_is_refused() -> None:
    _assert_refused_before_study(
        _config(owner=False, space={RATIO: _RATIO_DIM}, mode="replace")
    )


def test_merge_override_of_the_default_ratio_with_owner_off_is_refused() -> None:
    _assert_refused_before_study(
        _config(owner=False, space={RATIO: _RATIO_DIM, "n_estimators": _SMALL_TREES})
    )


def test_an_owner_dimension_that_can_be_true_admits_the_ratio() -> None:
    """The config says off, but every trial overlays the sampled owner."""
    space = {
        OWNER: {"type": "categorical", "choices": [True, False], "category": "smart"},
        RATIO: _RATIO_DIM,
    }
    model = Model(
        _config(owner=False, space=space, mode="replace", n_trials=3),
        data=make_binary_df(n=120),
    )
    model.tune()
    assert RATIO in _sampled_names(model)


def test_an_owner_dimension_that_cannot_be_true_refuses_the_ratio() -> None:
    """The config says on, but every trial overlays ``False``.

    An implementation reading only the config's value would admit this.
    """
    space = {
        OWNER: {"type": "categorical", "choices": [False], "category": "smart"},
        RATIO: _RATIO_DIM,
    }
    _assert_refused_before_study(_config(owner=True, space=space, mode="replace"))


# ---------------------------------------------------------------------------
# Decision 4: dimensions from the default space (criterion 6)
# ---------------------------------------------------------------------------


def test_the_default_ratio_is_dropped_when_the_owner_is_off(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = Model(
        _config(owner=False, space={"n_estimators": _SMALL_TREES}),
        data=make_binary_df(n=120),
    )
    with caplog.at_level(logging.INFO, logger="lizyml"):
        result = model.tune()

    assert RATIO not in result.best_params
    assert RATIO not in _sampled_names(model)
    assert RATIO not in {dim.name for dim in model._space or []}
    assert RATIO not in model._study.best_trial.distributions
    dropped = [
        r for r in caplog.records if r.levelno == logging.INFO and RATIO in r.message
    ]
    assert dropped, "the dropped dimension must be named in an INFO log"


def test_the_default_ratio_is_kept_when_the_owner_is_on() -> None:
    model = Model(
        _config(owner=True, space={"n_estimators": _SMALL_TREES}),
        data=make_binary_df(n=120),
    )
    result = model.tune()
    assert RATIO in result.best_params


def _storage(tmp_path: Any) -> str:
    return f"sqlite:///{tmp_path / 'studies.db'}"


def test_a_new_name_in_storage_is_a_new_study_and_drops(tmp_path: Any) -> None:
    """One side of the decision point: storage given, no study of that name."""
    storage = _storage(tmp_path)
    optuna.create_study(storage=storage, study_name="other", direction="minimize")
    model = Model(
        _config(owner=False, space={"n_estimators": _SMALL_TREES}),
        data=make_binary_df(n=120),
    )
    result = model.tune(storage=storage, study_name="fresh")
    assert RATIO not in result.best_params
    assert RATIO not in _sampled_names(model)


def test_resume_false_onto_a_stored_study_is_refused(tmp_path: Any) -> None:
    """The other side: ``resume=False`` onto an existing stored study.

    Formerly the H-0121 temporary pin (criterion 6), which reattached and kept
    the ratio. H-0122 criterion 9 makes this path ``CONFIG_INVALID``: nothing
    is sampled and nothing is written.
    """
    storage = _storage(tmp_path)
    optuna.create_study(storage=storage, study_name="kept", direction="minimize")
    model = Model(
        _config(owner=False, space={"n_estimators": _SMALL_TREES}),
        data=make_binary_df(n=120),
    )
    with pytest.raises(LizyMLError) as exc:
        model.tune(storage=storage, study_name="kept")
    assert exc.value.code is ErrorCode.CONFIG_INVALID
    assert "resume=True" in exc.value.user_message
    stored = optuna.load_study(storage=storage, study_name="kept")
    assert stored.trials == []
    assert not [k for k in stored.user_attrs if k.startswith("lizyml:")]


def test_resume_true_onto_an_uninitialised_study_is_a_new_study_and_drops(
    tmp_path: Any,
) -> None:
    """A stored study with no trials and no LizyML record (state E) counts as
    new (H-0122 §7.4): decision 4 drops the default ratio, and the study gets
    its ``lizyml:study`` record."""
    storage = _storage(tmp_path)
    optuna.create_study(storage=storage, study_name="kept", direction="minimize")
    model = Model(
        _config(owner=False, space={"n_estimators": _SMALL_TREES}),
        data=make_binary_df(n=120),
    )
    result = model.tune(storage=storage, study_name="kept", resume=True)
    assert RATIO not in result.best_params
    assert RATIO not in _sampled_names(model)
    stored = optuna.load_study(storage=storage, study_name="kept")
    assert "lizyml:study" in stored.user_attrs
    assert [r.round for r in result.rounds] == [1]


# ---------------------------------------------------------------------------
# The provider's declaration
# ---------------------------------------------------------------------------


def test_the_provider_declares_the_ratio_owner() -> None:
    provider = LGBMProvider()
    owners = provider.smart_param_owners()
    assert owners == {RATIO: OWNER}
    assert set(owners) | set(owners.values()) <= provider.smart_param_names()


@pytest.mark.parametrize("ratio", [0.2, 0.9])
def test_the_declared_owner_is_true_of_the_resolver(ratio: float) -> None:
    """Executed rather than read: the ratio moves the result only with its owner."""

    def resolve(smart: dict[str, Any]) -> dict[str, Any]:
        resolved, _ = resolve_smart_params(
            smart=smart,
            effective_params={"max_depth": 6},
            n_rows=100,
            feature_names=["feat_a", "feat_b"],
            y=make_binary_df(n=100)["target"],
            task="binary",
        )
        return resolved

    off = {OWNER: False, "balanced": False}
    on = {OWNER: True, "balanced": False}
    assert resolve({**off, RATIO: ratio}) == resolve({**off, RATIO: 1.0})
    assert resolve({**on, RATIO: ratio}) != resolve({**on, RATIO: 1.0})


def test_without_the_declaration_both_decisions_go_inert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative control: the declaration is what criteria 5 and 6 rest on."""
    monkeypatch.setattr(LGBMProvider, "smart_param_owners", lambda self: {})

    accepted = Model(
        _config(owner=False, space={RATIO: _RATIO_DIM}, mode="replace"),
        data=make_binary_df(n=120),
    )
    accepted.tune()
    assert RATIO in _sampled_names(accepted)

    kept = Model(
        _config(owner=False, space={"n_estimators": _SMALL_TREES}),
        data=make_binary_df(n=120),
    )
    assert RATIO in kept.tune().best_params


# ---------------------------------------------------------------------------
# The rule itself, on hand-built spaces
# ---------------------------------------------------------------------------

_RATIO_SPACE_DIM = FloatDim(RATIO, 0.1, 0.9, category="smart")


@pytest.mark.parametrize(
    ("owner_dim", "config_owner", "inactive"),
    [
        (None, False, True),
        (None, None, True),
        (None, True, False),
        (CategoricalDim(OWNER, (True, False), category="smart"), False, False),
        (CategoricalDim(OWNER, (False,), category="smart"), True, True),
        (CategoricalDim(OWNER, (False, None, 0), category="smart"), True, True),
        (IntDim(OWNER, 0, 0, category="smart"), True, True),
        (IntDim(OWNER, 0, 1, category="smart"), False, False),
    ],
)
def test_the_owner_rule(
    owner_dim: Any, config_owner: bool | None, inactive: bool
) -> None:
    from lizyml.core._tuning_validation import inactive_smart_dimensions

    space = [_RATIO_SPACE_DIM] + ([owner_dim] if owner_dim is not None else [])
    got = inactive_smart_dimensions(LGBMProvider(), space, {OWNER: config_owner})
    assert got == ([_RATIO_SPACE_DIM] if inactive else [])
