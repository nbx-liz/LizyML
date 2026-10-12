"""H-0123: a search-space dimension is checked against the documented format.

``tuning.optuna.space`` and ``tune(space=...)`` take the format docs/config-
reference.md and BLUEPRINT §11.1 state. Before H-0123 ``parse_space`` read a
dimension through ``float()`` / ``int()`` / ``bool()`` and silently turned a
spec outside that format into another search (``log: "false"`` searched on a
log scale, ``step`` was dropped, an ``int`` bound ``1.7`` became ``1``). Each
form is now refused as ``CONFIG_INVALID`` on both surfaces, before any study is
written and before ``lgb.train`` runs.

Criteria: 1 (each form, both surfaces), 2 (a ``UserDict`` spec in the config),
3 (a non-mapping spec in the config), 4 / 4b (a YAML config through
``load_config``), 7 (in-format specs resolve as before).
"""

from __future__ import annotations

import math
from collections import UserDict
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pytest
import yaml

from lizyml import Model
from lizyml.config.loader import load_config
from lizyml.config.schema import LizyMLConfig, OptunaConfig
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.search_dim import CategoricalDim, FloatDim, IntDim
from lizyml.tuning.search_space import parse_space
from tests._train_spy import record_lightgbm_calls
from tests.test_tuning._resume_helpers import (
    fast_config,
    frame,
    round_record,
    snapshot_model,
    snapshot_study,
    url,
)

CI = ErrorCode.CONFIG_INVALID
CONFIG = "tuning.optuna.space"
SPACE_ARG = "tune(space=)"

#: The dimension name each refused form is declared under, by its ``type``.
NAMES = {"float": "learning_rate", "int": "num_leaves", "categorical": "extra_trees"}
LR = {"type": "float", "low": 0.05, "high": 0.3, "log": True}
NL = {"type": "int", "low": 4, "high": 8}


def _gen() -> Any:
    return (i for i in range(3))


#: id -> (spec, the key the refusal names, its reason). The forms are H-0123's
#: 目的 and 互換性 lists (the reviewed controls of firing_rate_344_controls.py).
REFUSED: dict[str, tuple[Any, str | None, str]] = {
    "log_str": ({**LR, "log": "false"}, "log", "log_not_bool"),
    "log_int": ({**LR, "log": 1}, "log", "log_not_bool"),
    "log_numpy_bool": ({**LR, "log": np.bool_(True)}, "log", "log_not_bool"),
    "step_key": ({**NL, "step": 2}, "step", "step_not_supported"),
    "typo_key": ({**LR, "lo": 3}, "lo", "unknown_key"),
    "categorical_with_low": (
        {"type": "categorical", "choices": [True, False], "low": 1},
        "low",
        "unknown_key",
    ),
    "int_fraction": ({**NL, "low": 1.7}, "low", "bound_not_integral"),
    "int_fraction_str": ({**NL, "low": "1.7"}, "low", "bound_not_integral"),
    "int_fraction_numpy": (
        {**NL, "high": np.float64(8.5)},
        "high",
        "bound_not_integral",
    ),
    "bad_category": ({**LR, "category": "booster"}, "category", "unknown_category"),
    "category_not_str": ({**LR, "category": 1}, "category", "unknown_category"),
    "choices_str": (
        {"type": "categorical", "choices": "abc"},
        "choices",
        "choices_not_a_list",
    ),
    "choices_set": (
        {"type": "categorical", "choices": {1, 2, 3}},
        "choices",
        "choices_not_a_list",
    ),
    "choices_dict": (
        {"type": "categorical", "choices": {"a": 1, "b": 2}},
        "choices",
        "choices_not_a_list",
    ),
    "choices_range": (
        {"type": "categorical", "choices": range(3)},
        "choices",
        "choices_not_a_list",
    ),
    "choices_ndarray": (
        {"type": "categorical", "choices": np.array(["a", "b"], dtype=object)},
        "choices",
        "choices_not_a_list",
    ),
    "choices_gen": (
        {"type": "categorical", "choices": "<generator>"},
        "choices",
        "choices_not_a_list",
    ),
    "missing_choices": ({"type": "categorical"}, "choices", "malformed_dim"),
    "missing_high": ({"type": "float", "low": 0.1}, "high", "malformed_dim"),
    "nan_low": ({**LR, "low": float("nan")}, "low", "bound_not_finite"),
    "inf_high": ({**LR, "high": float("inf")}, "high", "bound_not_finite"),
    "neg_inf_low": ({**LR, "log": False, "low": -math.inf}, "low", "bound_not_finite"),
    "nan_str": ({**LR, "low": "nan"}, "low", "bound_not_a_number"),
    "inf_str": ({**LR, "high": "inf"}, "high", "bound_not_a_number"),
    "overflow_str": ({**LR, "high": "1e999"}, "high", "bound_not_finite"),
    "int_overflow_str": ({**NL, "high": "1e999"}, "high", "bound_not_finite"),
    "float_overflow_int": ({**LR, "high": 10**400}, "high", "bound_not_finite"),
    "hex_str": ({**NL, "low": "0x10"}, "low", "bound_not_a_number"),
    "comma_str": ({**NL, "high": "1,000"}, "high", "bound_not_a_number"),
    "word_str": ({**LR, "low": "abc"}, "low", "bound_not_a_number"),
    "trailing_junk_str": ({**LR, "low": "1e0x"}, "low", "bound_not_a_number"),
    "unicode_digit_str": ({**NL, "low": "١"}, "low", "bound_not_a_number"),
    "newline_str": ({**NL, "low": "4\n"}, "low", "bound_not_a_number"),
    "bool_bound": ({**NL, "low": False, "high": True}, "low", "bound_not_a_number"),
    "numpy_bool_bound": ({**NL, "low": np.bool_(False)}, "low", "bound_not_a_number"),
    "bool_bound_float": (
        {**LR, "log": False, "low": False, "high": True},
        "low",
        "bound_not_a_number",
    ),
    "numpy_bool_bound_float": (
        {**LR, "low": np.bool_(False)},
        "low",
        "bound_not_a_number",
    ),
    "bytes_bound": ({**LR, "low": b"0.1"}, "low", "bound_not_a_number"),
    "decimal_bound": ({**LR, "low": Decimal("0.1")}, "low", "bound_not_a_number"),
    "fraction_bound": ({**LR, "low": Fraction(1, 10)}, "low", "bound_not_a_number"),
    "none_bound": ({**LR, "low": None}, "low", "bound_not_a_number"),
    "not_a_mapping": ([("type", "float")], None, "dim_not_a_mapping"),
}


def _spec(case: str) -> Any:
    """A fresh copy of the refused spec (a generator is single-use)."""
    spec = REFUSED[case][0]
    if case == "choices_gen":
        return {**spec, "choices": _gen()}
    return spec


def _name(spec: Any) -> str:
    kind = spec.get("type") if isinstance(spec, dict) else "float"
    return NAMES[kind]


def _assert_context(err: LizyMLError, case: str, surface: str) -> None:
    spec, key, reason = REFUSED[case]
    assert err.code is CI
    expected = {"surface": surface, "reason": reason, "dim": _name(spec)}
    if key is not None:
        expected["key"] = key
    assert {k: err.context.get(k) for k in expected} == expected
    assert surface in err.user_message
    assert _name(spec) in err.user_message


def _config_with(space: dict[str, Any]) -> dict[str, Any]:
    cfg = fast_config()
    cfg["tuning"]["optuna"]["space"] = space
    return cfg


def _study_names(storage: str) -> list[str]:
    return [s.study_name for s in optuna.get_all_study_summaries(storage)]


# ---------------------------------------------------------------------------
# Criterion 1 (and 3): each form, on both surfaces, before any study write
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", list(REFUSED))
def test_config_space_refuses_form(case: str, tmp_path: Path) -> None:
    """``tuning.optuna.space``: refused before a study or a Booster exists."""
    space = {_name(REFUSED[case][0]): _spec(case)}
    if case == "choices_gen":
        # The loader deep-copies a dict config, and a generator cannot be
        # copied; a validated config instance reaches tune() as it is.
        raw = LizyMLConfig.model_validate(load_config(fast_config()).model_dump())
        dumped = raw.model_dump()
        dumped["tuning"]["optuna"]["space"] = space
        model = Model(LizyMLConfig.model_validate(dumped), data=frame())
    else:
        model = Model(_config_with(space), data=frame())
    storage = url(tmp_path)
    before = snapshot_model(model)
    with record_lightgbm_calls() as calls, pytest.raises(LizyMLError) as caught:
        model.tune(storage=storage, study_name="s")
    _assert_context(caught.value, case, CONFIG)
    assert calls["train_params"] == []
    assert _study_names(storage) == []
    assert snapshot_model(model) == before


@pytest.mark.parametrize("case", list(REFUSED))
def test_space_arg_refuses_form(case: str) -> None:
    """``tune(space=...)`` after a first tune: refused, nothing changes."""
    model = Model(fast_config(), data=frame())
    model.tune()
    study = model._study
    before = (snapshot_study(study), snapshot_model(model))
    space = {_name(REFUSED[case][0]): _spec(case)}
    with record_lightgbm_calls() as calls, pytest.raises(LizyMLError) as caught:
        model.tune(resume=True, space=space)
    _assert_context(caught.value, case, SPACE_ARG)
    assert calls["train_params"] == []
    assert (snapshot_study(study), snapshot_model(model)) == before


def test_config_space_non_mapping_spec_is_config_invalid() -> None:
    """Criterion 3: not a bare ``AttributeError`` from ``spec.get``."""
    model = Model(_config_with({"learning_rate": [0.05, 0.3]}), data=frame())
    with pytest.raises(LizyMLError) as caught:
        model.tune()
    assert caught.value.code is CI
    assert caught.value.context["reason"] == "dim_not_a_mapping"
    assert caught.value.context["surface"] == CONFIG


def test_config_format_is_checked_before_the_name_checks() -> None:
    """Policy 4: the format refusal wins over a check that reads raw specs.

    The name checks read ``category`` from each raw spec; ``not_a_param`` would
    be refused there as an unknown parameter if the format were checked later.
    """
    spec = {"type": "float", "low": 0.1, "high": 1.0, "step": 0.1}
    model = Model(_config_with({"not_a_param": spec}), data=frame())
    with pytest.raises(LizyMLError) as caught:
        model.tune()
    assert caught.value.context["reason"] == "step_not_supported"
    assert caught.value.context["dim"] == "not_a_param"


def test_unknown_type_keeps_its_context_and_names_the_surface() -> None:
    with pytest.raises(LizyMLError) as caught:
        parse_space({"lr": {"type": "uniform", "low": 0.1, "high": 0.2}})
    context = caught.value.context
    assert (context["param"], context["type"]) == ("lr", "uniform")
    assert (context["surface"], context["reason"]) == (CONFIG, "unknown_type")


def test_step_refusal_says_dimensions_have_no_step() -> None:
    with pytest.raises(LizyMLError) as caught:
        parse_space({"num_leaves": {**NL, "step": 2}})
    assert "step" in caught.value.user_message
    assert "no step" in caught.value.user_message


# ---------------------------------------------------------------------------
# Criterion 2: a mapping that is not a dict, in the config
# ---------------------------------------------------------------------------


def test_config_copies_the_space_and_each_spec_to_dict() -> None:
    spec = UserDict({"type": "float", "low": 0.5, "high": 0.9, "category": "smart"})
    optuna_cfg = OptunaConfig(space=UserDict({"num_leaves_ratio": spec}))
    assert type(optuna_cfg.space) is dict
    assert type(optuna_cfg.space["num_leaves_ratio"]) is dict
    assert optuna_cfg.space["num_leaves_ratio"] == dict(spec)


def test_config_leaves_a_non_mapping_spec_for_parse_space() -> None:
    optuna_cfg = OptunaConfig(space={"learning_rate": [0.05, 0.3]})
    assert optuna_cfg.space == {"learning_rate": [0.05, 0.3]}


def test_userdict_smart_dim_in_config_is_read_as_smart() -> None:
    """The name check must not take a ``UserDict`` smart dim for a model one."""
    cfg = fast_config(n_trials=1)
    cfg["model"]["auto_num_leaves"] = True
    cfg["tuning"]["optuna"]["space"] = {
        "learning_rate": dict(LR),
        "num_leaves_ratio": UserDict(
            {"type": "float", "low": 0.5, "high": 0.9, "category": "smart"}
        ),
    }
    model = Model(cfg, data=frame())
    model.tune()
    declared = {d.name: d for d in round_record(model._study, 1).declared}
    assert declared["num_leaves_ratio"].category == "smart"


# ---------------------------------------------------------------------------
# Criteria 4 and 4b: a YAML config through load_config
# ---------------------------------------------------------------------------


def _yaml_config(tmp_path: Path, flow_space: str) -> Path:
    """A YAML config whose ``tuning.optuna.space`` is *flow_space*, verbatim.

    The space is written as text so that YAML 1.1 reads ``1e-4`` the way a
    user's file does (as a string), not as a float ``safe_dump`` produced.
    """
    cfg = fast_config()
    cfg["tuning"]["optuna"]["space"] = {}
    text = yaml.safe_dump(cfg, sort_keys=False)
    assert text.count("space: {}") == 1
    path = tmp_path / "config.yaml"
    path.write_text(text.replace("space: {}", f"space: {flow_space}"), "utf-8")
    return path


def test_yaml_exponent_bound_resolves_to_a_float(tmp_path: Path) -> None:
    """Criterion 4: ``low: 1e-4`` (a YAML 1.1 string) is the number 0.0001."""
    path = _yaml_config(
        tmp_path,
        "{learning_rate: {type: float, low: 1e-4, high: 1E-1, log: true}}",
    )
    assert load_config(path).tuning.optuna.space["learning_rate"]["low"] == "1e-4"
    model = Model(path, data=frame())
    model.tune()
    (dim,) = round_record(model._study, 1).declared
    assert isinstance(dim, FloatDim)
    assert (dim.low, dim.high) == (0.0001, 0.1)


@pytest.mark.parametrize(
    ("literal", "reason"),
    [
        (".nan", "bound_not_finite"),
        ("nan", "bound_not_a_number"),
        ('"abc"', "bound_not_a_number"),
    ],
)
def test_yaml_bound_outside_the_format_is_refused(
    literal: str, reason: str, tmp_path: Path
) -> None:
    path = _yaml_config(
        tmp_path, f"{{learning_rate: {{type: float, low: {literal}, high: 0.3}}}}"
    )
    model = Model(path, data=frame())
    with record_lightgbm_calls() as calls, pytest.raises(LizyMLError) as caught:
        model.tune()
    assert caught.value.code is CI
    assert caught.value.context["reason"] == reason
    assert caught.value.context["key"] == "low"
    assert calls["train_params"] == []


def test_yaml_int_number_literals_resolve_to_an_int_dim(tmp_path: Path) -> None:
    """Criterion 4b, config surface: ``"16.0"`` and ``1e2`` -> 16 and 100."""
    path = _yaml_config(
        tmp_path,
        '{num_leaves: {type: int, low: "16.0", high: 1e2}, '
        "learning_rate: {type: float, low: 0.05, high: 0.3}}",
    )
    model = Model(path, data=frame())
    model.tune()
    declared = {d.name: d for d in round_record(model._study, 1).declared}
    dim = declared["num_leaves"]
    assert isinstance(dim, IntDim)
    assert (dim.low, dim.high) == (16, 100)
    assert (type(dim.low), type(dim.high)) == (int, int)


def test_space_arg_int_number_literals_resolve_to_an_int_dim() -> None:
    """Criterion 4b, ``tune(space=)`` surface."""
    model = Model(fast_config(), data=frame())
    model.tune()
    space = {
        "num_leaves": {"type": "int", "low": "16.0", "high": "1e2"},
        "learning_rate": LR,
    }
    model.tune(resume=True, space=space)
    declared = {d.name: d for d in round_record(model._study, 2).declared}
    dim = declared["num_leaves"]
    assert isinstance(dim, IntDim)
    assert (dim.low, dim.high) == (16, 100)
    assert (type(dim.low), type(dim.high)) == (int, int)


# ---------------------------------------------------------------------------
# Criterion 7: in-format specs resolve as before
# ---------------------------------------------------------------------------

ACCEPTED: dict[str, tuple[dict[str, Any], Any]] = {
    "omitted_log_and_category": (
        {"type": "float", "low": 0.1, "high": 1.0},
        FloatDim(name="d", low=0.1, high=1.0, log=False, category="model"),
    ),
    "explicit_log_and_category": (
        {"type": "float", "low": 0.1, "high": 1.0, "log": True, "category": "smart"},
        FloatDim(name="d", low=0.1, high=1.0, log=True, category="smart"),
    ),
    "int_16_0": (
        {"type": "int", "low": 16.0, "high": 20},
        IntDim(name="d", low=16, high=20),
    ),
    "numpy_bounds": (
        {"type": "int", "low": np.int64(3), "high": np.float64(8.0)},
        IntDim(name="d", low=3, high=8),
    ),
    "numpy_float32": (
        {"type": "float", "low": np.float32(0.5), "high": 1.0},
        FloatDim(name="d", low=0.5, high=1.0),
    ),
    "float_number_strings": (
        {"type": "float", "low": "1e-4", "high": "1E-1", "log": True},
        FloatDim(name="d", low=0.0001, high=0.1, log=True),
    ),
    "dot_strings": (
        {"type": "float", "low": ".5", "high": "5."},
        FloatDim(name="d", low=0.5, high=5.0),
    ),
    "int_strings": (
        {"type": "int", "low": "16", "high": "+32"},
        IntDim(name="d", low=16, high=32),
    ),
    "huge_int": (
        {"type": "int", "low": 10**400, "high": 10**400 + 1},
        IntDim(name="d", low=10**400, high=10**400 + 1),
    ),
    "list_choices": (
        {"type": "categorical", "choices": ["gbdt", "dart"], "category": "model"},
        CategoricalDim(name="d", choices=("gbdt", "dart")),
    ),
    "tuple_choices": (
        {"type": "categorical", "choices": (0.5, 1.0)},
        CategoricalDim(name="d", choices=(0.5, 1.0)),
    ),
    "userdict_spec": (
        UserDict(
            {"type": "categorical", "choices": [True, False], "category": "smart"}
        ),  # type: ignore[dict-item]
        CategoricalDim(name="d", choices=(True, False), category="smart"),
    ),
}


@pytest.mark.parametrize("case", list(ACCEPTED))
def test_in_format_spec_resolves_as_before(case: str) -> None:
    spec, expected = ACCEPTED[case]
    (dim,) = parse_space({"d": spec})
    assert dim == expected
    if isinstance(dim, (FloatDim, IntDim)):
        plain = float if isinstance(dim, FloatDim) else int
        assert (type(dim.low), type(dim.high)) == (plain, plain)
