"""Controls for the H-0123 v4 predicate (firing_rate_344._reasons): the documented format."""

import sys
from collections import UserDict
from decimal import Decimal
from fractions import Fraction

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4]))
from firing_rate_344 import _reasons  # noqa: E402

flagged = {
    "log_str": {"type": "float", "low": 0.1, "high": 1.0, "log": "false"},
    "step_key": {"type": "int", "low": 1, "high": 9, "step": 2},
    "typo_key": {"type": "float", "low": 0.1, "high": 1.0, "lo": 3},
    "int_fraction": {"type": "int", "low": 1.7, "high": 5},
    "int_fraction_str": {"type": "int", "low": "1.7", "high": 5},
    "bad_category": {"type": "float", "low": 0.1, "high": 1.0, "category": "booster"},
    "choices_str": {"type": "categorical", "choices": "abc"},
    "choices_set": {"type": "categorical", "choices": {1, 2, 3}},
    "choices_dict": {"type": "categorical", "choices": {"a": 1, "b": 2}},
    "choices_gen": {"type": "categorical", "choices": (i for i in range(3))},
    "choices_range": {"type": "categorical", "choices": range(3)},
    "choices_ndarray": {"type": "categorical", "choices": np.array(["a", "b"], dtype=object)},
    "nan_low": {"type": "float", "low": float("nan"), "high": 1.0},
    "inf_high": {"type": "float", "low": 0.0, "high": float("inf")},
    "nan_str": {"type": "float", "low": "nan", "high": 1.0},
    "overflow_str": {"type": "float", "low": 0.0, "high": "1e999"},
    "hex_str": {"type": "int", "low": "0x10", "high": 99},
    "comma_str": {"type": "int", "low": 1, "high": "1,000"},
    "word_str": {"type": "float", "low": "abc", "high": 1.0},
    "bool_bound": {"type": "int", "low": False, "high": True},
    "numpy_bool_bound": {"type": "int", "low": np.bool_(False), "high": 3},
    "bytes_bound": {"type": "float", "low": b"0.1", "high": 1.0},
    "decimal_bound": {"type": "float", "low": Decimal("0.1"), "high": 1.0},
    "fraction_bound": {"type": "float", "low": Fraction(1, 10), "high": 1},
    "not_a_mapping": [("type", "float")],
}
accepted = {
    "clean": {"type": "float", "low": 0.1, "high": 1.0, "log": True, "category": "model"},
    "int_16_0": {"type": "int", "low": 16.0, "high": 20},
    "numpy": {"type": "int", "low": np.int64(3), "high": np.float64(8.0)},
    "numpy_float32": {"type": "float", "low": np.float32(0.1), "high": 1.0},
    "yaml_exponent": {"type": "float", "low": "1e-4", "high": "1E-1", "log": True},
    "int_str": {"type": "int", "low": "16", "high": "+32"},
    "dot_str": {"type": "float", "low": ".5", "high": "5."},
    "huge_int": {"type": "int", "low": 10**400, "high": 10**400 + 1},
    "tuple_choices": {"type": "categorical", "choices": (0.5, 1.0)},
    "list_choices": {"type": "categorical", "choices": ["gbdt", "dart"], "category": "model"},
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
