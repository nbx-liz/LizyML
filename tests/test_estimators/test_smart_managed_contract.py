"""The ``smart_managed_param_names`` contract, executed on LightGBM (H-0121 am. 1).

``EstimatorProvider.smart_managed_param_names`` promises that its result
depends on each smart value only through whether it is ``None`` and its
truthiness. The tune admission checks rely on it: they try one representative
value per state of each ``category: smart`` dimension
(``lizyml.core._smart_combinations``). Criterion 5d: replacing any smart
parameter's value with another of the same state never changes the result, for
every smart parameter, every task, and every state of the others.
"""

from __future__ import annotations

import itertools
from typing import Any

import pytest

from lizyml.estimators.lgbm.provider import LGBMProvider

PROVIDER = LGBMProvider()
SMART_NAMES = sorted(PROVIDER.smart_param_names())

#: Values of each state, of the types a config or a search dimension can give.
SAME_STATE: dict[Any, tuple[Any, ...]] = {
    None: (None,),
    False: (False, 0, 0.0, {}),
    True: (True, 1, 0.5, 2.0, {"feat_a": 1.0}),
}


def test_every_smart_parameter_is_covered() -> None:
    """The quantifier is over the declaration, not a copy of it (DC7)."""
    assert set(SMART_NAMES) == set(PROVIDER.extract_smart_params(_lgbm_config()))


def _lgbm_config() -> Any:
    from lizyml.config.schema import LGBMConfig

    return LGBMConfig(name="lgbm")


@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
@pytest.mark.parametrize("name", SMART_NAMES)
def test_result_depends_only_on_state(task: str, name: str) -> None:
    others = [n for n in SMART_NAMES if n != name]
    for states in itertools.product(SAME_STATE, repeat=len(others)):
        base = {n: SAME_STATE[s][0] for n, s in zip(others, states, strict=True)}
        for state, values in SAME_STATE.items():
            results = {
                value_repr: PROVIDER.smart_managed_param_names(
                    {**base, name: value}, task
                )
                for value_repr, value in ((repr(v), v) for v in values)
            }
            first = next(iter(results.values()))
            assert all(r == first for r in results.values()), (
                f"{task}: '{name}' in state {state!r} gives different managed "
                f"names for {sorted(results)} with the others at {base}"
            )
