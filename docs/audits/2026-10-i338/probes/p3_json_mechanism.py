"""Q1: why the message says 'Circular reference detected'."""

import json
import json.encoder as enc

from lightgbm.basic import _json_default_with_numpy

print("C encoder in use:", enc.c_make_encoder is not None)
for label, kw in [
    ("no default", {}),
    ("lgb default (returns obj unchanged)", {"default": _json_default_with_numpy}),
    ("identity default", {"default": lambda o: o}),
    ("identity default, check_circular=False", {"default": lambda o: o, "check_circular": False}),
    ("default=str", {"default": str}),
]:
    try:
        print(label, "->", json.dumps([[b"a"]], **kw))
    except Exception as exc:  # noqa: BLE001
        print(label, "->", type(exc).__name__, str(exc)[:80])
_obj = b"a"
print("lgb default(b'a') returns the same object:", _json_default_with_numpy(_obj) is _obj)
