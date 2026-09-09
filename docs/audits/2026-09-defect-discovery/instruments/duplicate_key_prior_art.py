"""Execute the prior-art claims rather than quoting them."""

import json

print("--- Python: one parameter under two spellings at a call site")
try:
    exec("def f(**kw): pass\nf(**{'a': 1}, **{'a': 1})")
except TypeError as exc:
    print(f"  equal values     -> TypeError: {exc}")
try:
    exec("def f(**kw): pass\nf(**{'a': 1}, **{'a': 2})")
except TypeError as exc:
    print(f"  different values -> TypeError: {exc}")

print()
print("--- Python dict literal: duplicate key")
d = {"a": 1, "a": 2}  # noqa: F601
print(f"  {{'a': 1, 'a': 2}} -> {d}  (last wins, silent)")

print()
print("--- json: duplicate key in a document")
doc = '{"a":1,"a":2}'
print("  json.loads(" + doc + ") -> " + repr(json.loads(doc)) + "  (last wins, silent)")

print()
print("--- pydantic: field populated by both its name and its alias")
try:
    from pydantic import BaseModel, ConfigDict, Field

    class M(BaseModel):
        model_config = ConfigDict(populate_by_name=True)
        learning_rate: float = Field(default=0.1, alias="eta")

    for payload in ({"eta": 0.5, "learning_rate": 0.9}, {"eta": 0.5, "learning_rate": 0.5}):
        try:
            print(f"  {payload} -> {M(**payload).learning_rate}")
        except Exception as exc:
            print(f"  {payload} -> {type(exc).__name__}: {str(exc)[:90]}")
except ImportError as exc:
    print(f"  pydantic unavailable: {exc}")

print()
print("--- sklearn: set_params with an unknown / duplicate name")
try:
    from sklearn.linear_model import Ridge

    r = Ridge()
    try:
        r.set_params(alpha=1.0, nope=2.0)
    except ValueError as exc:
        print(f"  unknown param -> ValueError: {str(exc)[:100]}")
except ImportError as exc:
    print(f"  sklearn unavailable: {exc}")
