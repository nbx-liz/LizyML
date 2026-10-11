"""Controls for the H-0123 predicate (firing_rate_344._reasons), v3 generic rule."""
import sys
from collections import UserDict
from decimal import Decimal
from fractions import Fraction

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4]))
from firing_rate_344 import _reasons  # noqa: E402


class OnlyInt:
    def __int__(self) -> int:
        return 3


class OnlyFloat:
    def __float__(self) -> float:
        return 0.5


flagged = {
    "log_str": {"type": "float", "low": 0.1, "high": 1.0, "log": "false"},
    "extra_keys_str_low": {"type": "float", "low": "0.1", "high": 1.0, "step": 0.1, "lo": 3},
    "int_fraction": {"type": "int", "low": 1.7, "high": 5},
    "int_longdouble_half": {"type": "int", "low": np.longdouble(2) ** 53 + np.longdouble("0.5"), "high": 2**60},
    "bad_category": {"type": "float", "low": 0.1, "high": 1.0, "category": "booster"},
    "choices_str": {"type": "categorical", "choices": "abc"},
    "nan_low": {"type": "float", "low": float("nan"), "high": 1.0},
    "inf_high": {"type": "float", "low": 0.0, "high": float("inf")},
    "decimal_overflow": {"type": "float", "low": Decimal("1"), "high": Decimal("1e999")},
    "gen_choices": {"type": "categorical", "choices": (i for i in range(3))},
    "bool_bound": {"type": "int", "low": False, "high": True},
    "bytes_bound": {"type": "float", "low": b"0.1", "high": 1.0},
    "int_only_object": {"type": "int", "low": OnlyInt(), "high": 9},
    "set_choices": {"type": "categorical", "choices": {1, 2, 3}},
    "dict_choices": {"type": "categorical", "choices": {"a": 1, "b": 2}},
}
accepted = {
    "clean": {"type": "float", "low": 0.1, "high": 1.0, "log": True, "category": "model"},
    "int_16_0": {"type": "int", "low": 16.0, "high": 20},
    "decimal_int": {"type": "int", "low": Decimal("16"), "high": Decimal("20")},
    "decimal_float": {"type": "float", "low": Decimal("0.1"), "high": Decimal("0.5")},
    "fraction": {"type": "float", "low": Fraction(1, 10), "high": 1},
    "numpy": {"type": "int", "low": np.int64(3), "high": np.float64(8.0)},
    "big_int": {"type": "int", "low": 2**53 + 1, "high": 2**53 + 3},
    "huge_int": {"type": "int", "low": 10**400, "high": 10**400 + 1},
    "float_only_object": {"type": "float", "low": OnlyFloat(), "high": 1.0},
    "range_choices": {"type": "categorical", "choices": range(3)},
    "tuple_choices": {"type": "categorical", "choices": (0.5, 1.0)},
    "ndarray_choices": {"type": "categorical", "choices": np.array(["a", "b"], dtype=object)},
    "userdict_spec": UserDict({"type": "categorical", "choices": [True, False], "category": "smart"}),
}
bad = 0
for label, spec in flagged.items():
    r = _reasons(spec)
    bad += not r
    print("FLAG  ", label, r)
for label, spec in accepted.items():
    r = _reasons(spec)
    bad += bool(r)
    print("ACCEPT", label, r)
print("CONTROL FAILURES:", bad)
