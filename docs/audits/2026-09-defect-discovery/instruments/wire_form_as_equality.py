"""Is `wire(a) == wire(b)` a usable stand-in for `not values_differ(a, b)`?

LightGBM stringifies every parameter before the C++ side sees it, so the wire
form is the only thing that can affect training. If two spellings produce the
same wire text, no observable difference exists. This executes that on the
exact values rounds 5-26 argued about.
"""

from lightgbm.basic import _param_dict_to_str
import numpy as np

PAIRS = [
    ("round 5: int vs float", 1, 1.0),
    ("bool vs int", True, 1),
    ("str vs float", "0.5", 0.5),
    ("equal floats", 0.5, 0.5),
    ("equal strings", "binary", "binary"),
    ("list vs comma text", [1.0, 2.0], "1.0,2.0"),
    ("list vs tuple", [1.0, 2.0], (1.0, 2.0)),
    ("list int vs float", [1, 2], [1.0, 2.0]),
    ("numpy scalar vs python", np.int64(1), 1),
    ("numpy float vs python", np.float64(0.5), 0.5),
    ("ndarray vs list", np.array([1.0, 2.0]), [1.0, 2.0]),
    ("numpy str vs str", np.str_("binary"), "binary"),
    ("None vs None", None, None),
    ("different", 0.5, 0.9),
]

print(f"{'case':28} {'wire(a)':24} {'wire(b)':24} same?")
for label, a, b in PAIRS:
    try:
        wa = _param_dict_to_str({"p": a})
    except Exception as exc:
        wa = f"<raise {type(exc).__name__}>"
    try:
        wb = _param_dict_to_str({"p": b})
    except Exception as exc:
        wb = f"<raise {type(exc).__name__}>"
    print(f"{label:28} {wa:24} {wb:24} {wa == wb}")

print()
print("values LightGBM refuses to serialise at all:")
for label, value in [
    ("dict", {"a": 1}),
    ("object", object()),
    ("nested list", [[1, 2], [3, 4]]),
    ("set", {1, 2}),
    ("2-D ndarray", np.zeros((2, 2))),
    ("surrogate str", "\ud800"),
]:
    try:
        print(f"  {label:16} -> {_param_dict_to_str({'p': value})!r}")
    except Exception as exc:
        print(f"  {label:16} -> raises {type(exc).__name__}: {exc}")
