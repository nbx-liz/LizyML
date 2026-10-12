"""H-0121 amendment 2, policy 1 (criteria 7 and 8): the smart value rule.

The provider declares, once per smart parameter, the values it can hold
(``EstimatorProvider.smart_value_rules``). The ratio ranges are the config's
own (``lizyml.config.schema.SMART_RATIO_RANGES``, which ``LGBMConfig``
validates with), so the config field and a search dimension cannot disagree
about a ratio (DC3): changing the shared range changes both checks.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from pydantic import ValidationError

from lizyml.config import schema
from lizyml.config.schema import LGBMConfig, RatioRange
from lizyml.core.types.search_dim import CategoricalDim, FloatDim, IntDim
from lizyml.estimators.lgbm.provider import LGBMProvider

RATIOS = ("num_leaves_ratio", "min_data_in_leaf_ratio", "min_data_in_bin_ratio")
PROBES: tuple[Any, ...] = (
    -1.0,
    0.0,
    1e-12,
    0.5,
    1.0,
    1.0 + 1e-12,
    2.0,
    math.nan,
    math.inf,
)


def _config_accepts(name: str, value: Any) -> bool:
    try:
        LGBMConfig(name="lgbm", **{name: value})
    except ValidationError:
        return False
    return True


def _rule_accepts(name: str, value: Any) -> bool:
    rule = LGBMProvider().smart_value_rules()[name]
    return rule.check_value(value, "binary") is None


def test_every_smart_parameter_has_one_rule() -> None:
    provider = LGBMProvider()
    assert set(provider.smart_value_rules()) == set(provider.smart_param_names())


@pytest.mark.parametrize("name", RATIOS)
def test_the_config_field_and_the_rule_agree_on_every_probe(name: str) -> None:
    for value in PROBES:
        assert _config_accepts(name, value) == _rule_accepts(name, value), value


def test_changing_the_shared_range_changes_both_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _config_accepts("num_leaves_ratio", 1.0)
    assert _rule_accepts("num_leaves_ratio", 1.0)
    monkeypatch.setitem(
        schema.SMART_RATIO_RANGES,
        "num_leaves_ratio",
        RatioRange(low=0.0, high=1.0, upper_inclusive=False),
    )
    assert not _config_accepts("num_leaves_ratio", 1.0)
    assert not _rule_accepts("num_leaves_ratio", 1.0)


def test_the_expansion_bound_is_inside_the_rule() -> None:
    """Policy 3: expansion never leaves the rule, so never reaches a refusal."""
    provider = LGBMProvider()
    rules = provider.smart_value_rules()
    for task in ("regression", "binary", "multiclass"):
        for name, bound in provider.parameter_bounds(task).items():
            if name in rules:
                for edge in (bound["min"], bound["max"]):
                    assert rules[name].check_value(edge, task) is None, (name, edge)


@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
def test_the_default_space_is_inside_the_rule(task: str) -> None:
    provider = LGBMProvider()
    rules = provider.smart_value_rules()
    for dim in provider.default_space(task):
        if dim.category == "smart":
            assert rules[dim.name].check_dimension(dim, task) is None, dim


@pytest.mark.parametrize(
    ("name", "dim", "task", "expected"),
    [
        (
            "auto_num_leaves",
            IntDim("auto_num_leaves", 0, 1, category="smart"),
            "binary",
            ("low", 0, "smart_value_invalid"),
        ),
        (
            "balanced",
            CategoricalDim("balanced", (False, True), category="smart"),
            "regression",
            ("choices", True, "smart_value_unsupported_task"),
        ),
        (
            "num_leaves_ratio",
            FloatDim("num_leaves_ratio", 0.5, 2.0, category="smart"),
            "binary",
            ("high", 2.0, "smart_value_invalid"),
        ),
        (
            "feature_weights",
            CategoricalDim("feature_weights", (None,), category="smart"),
            "binary",
            ("choices", None, "smart_value_invalid"),
        ),
    ],
)
def test_check_dimension_names_the_first_value_outside(
    name: str, dim: Any, task: str, expected: tuple[str, Any, str]
) -> None:
    rule = LGBMProvider().smart_value_rules()[name]
    assert rule.check_dimension(dim, task) == expected


def test_the_rule_does_not_accept_numpy_scalars_or_bool_as_numbers() -> None:
    np = pytest.importorskip("numpy")
    rule = LGBMProvider().smart_value_rules()["num_leaves_ratio"]
    assert rule.check_value(np.float64(0.5), "binary") == "smart_value_invalid"
    assert rule.check_value(True, "binary") == "smart_value_invalid"
    assert rule.check_value(0.5, "binary") is None
    assert rule.check_value(1, "binary") is None
