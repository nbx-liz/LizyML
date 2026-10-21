"""Unit tests for the ``tune(space=...)`` declaration rules (H-0122 §5.2, §4.1b, U2).

Pure functions of ``lizyml.core._tuning_space`` over hand-built dims and
lineage records: resolution by ``space_mode`` (5.2-1), alias normalisation
(design D-b), the category rule (4.1b), the unchangeable changes (5.2-3), the
per-dim carry-forward (5.2-2), the out-of-range freeze (5.3) and the open-round
comparison (7.2 U2).
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import optuna
import pytest

from lizyml import Model
from lizyml.core._tuning_plan import resolve_config_space
from lizyml.core._tuning_space import (
    admit_space_routes,
    carry_forward,
    normalise_names,
    outside_range,
    refuse_unchangeable,
    resolve_declaration,
    resolve_space,
    same_declared_space,
)
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.search_dim import CategoricalDim, FloatDim, IntDim, SearchDim
from lizyml.estimators.lgbm.provider import LGBMProvider
from lizyml.tuning.lineage import Provenance, Route
from lizyml.tuning.lineage_codec import dim_signature
from lizyml.tuning.search_space import parse_space
from tests.test_tuning._lineage_fixtures import stored_round
from tests.test_tuning._resume_helpers import fast_config

PROVIDER = LGBMProvider()
LR = FloatDim("learning_rate", 0.05, 0.3, log=True)
ETA_ROUTE = {"eta": Route("eta", "learning_rate", "model")}


def _sigs(dims: Any) -> list[str]:
    return [dim_signature(d) for d in dims]


# ---------------------------------------------------------------------------
# resolve_space (5.2-1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["merge", "replace"])
def test_resolve_space_matches_the_config_resolution(mode: str) -> None:
    cfg = Model(fast_config(mode=mode))._cfg
    assert cfg.tuning is not None
    old = resolve_config_space(cfg, PROVIDER)
    new = resolve_space(
        cfg.tuning.optuna.space_mode,
        parse_space(cfg.tuning.optuna.space),
        PROVIDER,
        cfg.task,
    )
    assert _sigs(new.declared) == _sigs(old.declared)
    assert new.provenance == old.provenance
    assert new.used_default == old.used_default
    assert new.fixed == old.fixed
    assert new.default_dims == old.default_dims


def test_merge_empty_space_declares_the_defaults() -> None:
    resolved = resolve_space("merge", [], PROVIDER, "regression")
    assert _sigs(resolved.declared) == _sigs(PROVIDER.default_space("regression"))
    assert resolved.used_default is True
    assert {p.origin for p in resolved.provenance} == {"default"}


def test_replace_empty_space_declares_nothing() -> None:
    resolved = resolve_space("replace", [], PROVIDER, "regression")
    assert resolved.declared == ()
    assert resolved.used_default is False


# ---------------------------------------------------------------------------
# normalise_names (design D-b) and admit_space_routes (4.1b)
# ---------------------------------------------------------------------------


def test_alias_is_renamed_to_the_routed_name() -> None:
    dims = [LR, FloatDim("num_leaves_ratio", 0.5, 1.0, category="smart")]
    out = normalise_names(dims, ETA_ROUTE, PROVIDER)
    assert [d.name for d in out] == ["eta", "num_leaves_ratio"]
    assert replace(out[0], name="learning_rate") == LR


def test_unrouted_names_are_kept() -> None:
    dims = [FloatDim("eta", 0.05, 0.3), IntDim("max_depth", 3, 6)]
    assert normalise_names(dims, {}, PROVIDER) == dims


@pytest.mark.parametrize(
    ("routes", "dim"),
    [
        pytest.param(
            {
                "num_leaves_ratio": Route(
                    "num_leaves_ratio", "num_leaves_ratio", "smart"
                )
            },
            FloatDim("num_leaves_ratio", 0.5, 1.0, category="model"),
            id="raw_name",
        ),
        pytest.param(
            ETA_ROUTE,
            FloatDim("learning_rate", 0.05, 0.3, category="training"),
            id="canonical",
        ),
    ],
)
def test_category_change_is_refused(routes: dict[str, Route], dim: SearchDim) -> None:
    with pytest.raises(LizyMLError) as info:
        admit_space_routes([dim], routes, PROVIDER)
    assert info.value.code == ErrorCode.CONFIG_INVALID
    assert info.value.context["surface"] == "tune(space=)"
    assert "category" in info.value.user_message


def test_alias_of_a_routed_model_dim_is_admitted() -> None:
    admit_space_routes([FloatDim("learning_rate", 0.1, 0.2)], ETA_ROUTE, PROVIDER)


# ---------------------------------------------------------------------------
# refuse_unchangeable (5.2-3)
# ---------------------------------------------------------------------------


def _round_with(*dims: SearchDim, k: int = 1) -> Any:
    return stored_round(k, 0, declared=tuple(dims), effective=tuple(dims))


def _routes_for(*dims: SearchDim) -> dict[str, Route]:
    return {
        d.name: Route(
            d.name,
            PROVIDER.canonical_param_names([d.name])[d.name]
            if d.category == "model"
            else d.name,
            d.category,
        )
        for d in dims
    }


@pytest.mark.parametrize(
    ("recorded", "given", "changed"),
    [
        pytest.param(LR, IntDim("learning_rate", 1, 3), "kind", id="kind"),
        pytest.param(LR, replace(LR, log=False), "log", id="log"),
        pytest.param(
            CategoricalDim("c", (1, 2)), CategoricalDim("c", (2, 1)), "choices",
            id="choice_order",
        ),
        pytest.param(
            CategoricalDim("c", (True, False)), CategoricalDim("c", (1, 0)), "choices",
            id="true_to_1",
        ),
        pytest.param(
            CategoricalDim("c", (1, 2)), CategoricalDim("c", (1.0, 2.0)), "choices",
            id="1_to_1p0",
        ),
    ],
)  # fmt: skip
def test_unchangeable_changes_are_refused(
    recorded: SearchDim, given: SearchDim, changed: str
) -> None:
    rounds = (_round_with(recorded),)
    with pytest.raises(LizyMLError) as info:
        refuse_unchangeable([given], rounds, _routes_for(recorded))
    err = info.value
    assert err.code == ErrorCode.CONFIG_INVALID
    assert err.context["surface"] == "tune(space=)"
    assert err.context["dim"] == given.name
    assert err.context["changed"] == changed
    assert "new study" in err.user_message


@pytest.mark.parametrize(
    ("recorded", "given"),
    [
        pytest.param(LR, replace(LR, low=1e-4, high=2e-4), id="range"),
        pytest.param(
            CategoricalDim("c", (0.0, math.nan)),
            CategoricalDim("c", (0.0, float("nan"))),
            id="nan_equals_nan",
        ),
        pytest.param(
            FloatDim("eta", 0.05, 0.3), FloatDim("learning_rate", 0.01, 0.9), id="alias"
        ),
    ],
)
def test_range_changes_and_equal_choices_pass(
    recorded: SearchDim, given: SearchDim
) -> None:
    refuse_unchangeable([given], (_round_with(recorded),), _routes_for(recorded))


def test_alias_with_another_kind_is_refused() -> None:
    eta = FloatDim("eta", 0.05, 0.3)
    with pytest.raises(LizyMLError) as info:
        refuse_unchangeable(
            [IntDim("learning_rate", 1, 2)], (_round_with(eta),), _routes_for(eta)
        )
    assert info.value.context["changed"] == "kind"


def test_a_removed_dim_is_still_history() -> None:
    """Optuna checks against every distribution a name ever had, so a dim
    removed in round 2 and re-added in round 3 is compared with round 1's."""
    other = IntDim("num_leaves", 4, 8)
    rounds = (_round_with(LR, other), _round_with(other, k=2))
    with pytest.raises(LizyMLError) as info:
        refuse_unchangeable(
            [replace(LR, log=False), other], rounds, _routes_for(LR, other)
        )
    assert info.value.context["changed"] == "log"


def test_optuna_itself_passes_the_type_only_choice_changes() -> None:
    """Why LizyML compares choice types itself (5.2-3, DC7 pin).

    Executed against the installed Optuna: ``True`` -> ``1`` and ``1`` ->
    ``1.0`` complete without error (the study then mixes the two), while an
    order change fails the trial -- after Optuna has written it.
    """
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    def run(first: tuple[Any, ...], second: tuple[Any, ...]) -> str:
        study = optuna.create_study()
        study.optimize(lambda t: float(bool(t.suggest_categorical("c", first))), 1)
        study.optimize(
            lambda t: float(bool(t.suggest_categorical("c", second))),
            1,
            catch=(ValueError,),
        )
        return study.trials[-1].state.name

    assert run((True, False), (1, 0)) == "COMPLETE"
    assert run((1, 2), (1.0, 2.0)) == "COMPLETE"
    assert run((1, 2), (2, 1)) == "FAIL"


# ---------------------------------------------------------------------------
# carry_forward (5.2-2) and outside_range (5.3)
# ---------------------------------------------------------------------------

BOUNDS = PROVIDER.parameter_bounds("regression")


def test_carry_forward_per_dim() -> None:
    kept = IntDim("num_leaves", 4, 8)
    kept_eff = replace(kept, low=2, high=8)  # expanded in an earlier round
    moved = LR
    removed = FloatDim("feature_fraction", 0.5, 1.0)
    previous = stored_round(
        2,
        4,
        declared=(kept, moved, removed),
        effective=(
            kept_eff,
            replace(moved, min_allowed=1e-8, max_allowed=1.0),
            removed,
        ),
    )
    new_lr = replace(LR, low=1e-4, high=2e-4)
    fresh = IntDim("max_depth", 3, 6)
    base, changed = carry_forward(
        [new_lr, kept, fresh], previous, BOUNDS, _routes_for(kept, moved, removed)
    )
    assert [d.name for d in base] == ["learning_rate", "num_leaves", "max_depth"]
    assert base[1] == kept_eff
    assert base[0] == replace(new_lr, min_allowed=1e-8, max_allowed=1.0)
    # H-0140: max_depth keeps its lower bound and has no upper bound.
    assert base[2] == replace(fresh, min_allowed=-1, max_allowed=None)
    assert changed == frozenset({"learning_rate", "max_depth"})


def test_a_recorded_max_depth_keeps_its_recorded_ceiling() -> None:
    """H-0140 acceptance 12: removing the provider's ceiling 30 does not
    rewrite a recorded study; an unchanged dimension keeps its clamp."""
    depth = IntDim("max_depth", 3, 12)
    recorded = replace(depth, low=3, high=30, min_allowed=-1, max_allowed=30)
    previous = stored_round(2, 4, declared=(depth,), effective=(recorded,))
    base, changed = carry_forward([depth], previous, BOUNDS, _routes_for(depth))
    assert list(base) == [recorded]
    assert changed == frozenset()


def test_outside_range() -> None:
    dims = [
        FloatDim("a", 0.1, 0.2),
        IntDim("b", 3, 5),
        FloatDim("c", 0.1, 0.2),
        CategoricalDim("d", (1, 2)),
    ]
    best = {"a": 0.05, "b": 4, "d": 9}
    assert outside_range(dims, best) == frozenset({"a"})


# ---------------------------------------------------------------------------
# same_declared_space (U2)
# ---------------------------------------------------------------------------


def _parse(space: dict[str, Any]) -> list[SearchDim]:
    return parse_space(space)


SPACE = {
    "learning_rate": {"type": "float", "low": 0.05, "high": 0.3, "log": False},
    "lambda_l1": {"type": "categorical", "choices": [0.0, 1.0]},
}


@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(dict(reversed(list(SPACE.items()))), id="reordered"),
        pytest.param(
            {
                **SPACE,
                "learning_rate": {"type": "float", "low": 0.05, "high": 0.3},
            },
            id="omitted_log",
        ),
    ],
)
def test_same_declared_space_ignores_spelling(variant: dict[str, Any]) -> None:
    assert same_declared_space(_parse(variant), _parse(SPACE), {})


def test_same_declared_space_alias() -> None:
    recorded = _parse({"eta": SPACE["learning_rate"]})
    assert same_declared_space(_parse(SPACE)[:1], recorded, ETA_ROUTE)


@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            {**SPACE, "learning_rate": {"type": "float", "low": 0.05, "high": 0.4}},
            id="range",
        ),
        pytest.param(
            {**SPACE, "lambda_l1": {"type": "categorical", "choices": [0, 1.0]}},
            id="choice_type",
        ),
        pytest.param({"learning_rate": SPACE["learning_rate"]}, id="dim_missing"),
    ],
)
def test_same_declared_space_differs(variant: dict[str, Any]) -> None:
    assert not same_declared_space(_parse(variant), _parse(SPACE), {})


def test_same_declared_space_nan() -> None:
    a = [CategoricalDim("lambda_l1", (0.0, float("nan")))]
    b = [CategoricalDim("lambda_l1", (0.0, math.nan))]
    assert same_declared_space(a, b, {})


# ---------------------------------------------------------------------------
# resolve_declaration
# ---------------------------------------------------------------------------


def test_resolve_declaration_merge_drops_inactive_default_and_routes_new() -> None:
    routes = _routes_for(LR)
    space_dims = _parse({"max_depth": {"type": "int", "low": 3, "high": 6}})
    decl = resolve_declaration(
        space_dims,
        space_mode="merge",
        provider=PROVIDER,
        task="regression",
        routes=routes,
        base_smart={"auto_num_leaves": False},
    )
    names = [d.name for d in decl.declared]
    assert "num_leaves_ratio" not in names
    assert "max_depth" in names
    assert decl.space_arg == tuple(space_dims)
    assert Provenance("model", "max_depth", "user") in decl.provenance
    assert Provenance("model", "learning_rate", "default") in decl.provenance
    assert {r.name for r in decl.routes_added} == set(names) - {"learning_rate"}


def test_resolve_declaration_refuses_a_user_inactive_smart_dim() -> None:
    space_dims = _parse(
        {
            "num_leaves_ratio": {
                "type": "float",
                "low": 0.5,
                "high": 1.0,
                "category": "smart",
            }
        }
    )
    with pytest.raises(LizyMLError) as info:
        resolve_declaration(
            space_dims,
            space_mode="replace",
            provider=PROVIDER,
            task="regression",
            routes={},
            base_smart={"auto_num_leaves": False},
        )
    assert info.value.code == ErrorCode.CONFIG_INVALID


def test_resolve_declaration_renames_an_alias_in_space_and_defaults() -> None:
    """Merge mode: both a ``space=`` alias and the provider default meet the
    routed raw name, so the route stays single (D-b)."""
    decl = resolve_declaration(
        [],
        space_mode="merge",
        provider=PROVIDER,
        task="regression",
        routes=ETA_ROUTE,
        base_smart={"auto_num_leaves": True},
    )
    names = [d.name for d in decl.declared]
    assert "eta" in names
    assert "learning_rate" not in names
    assert "eta" not in {r.name for r in decl.routes_added}
