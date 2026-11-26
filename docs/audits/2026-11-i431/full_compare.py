"""Compare two H-0138 oracles field by field after masking only allowed changes.

    python full_compare.py <old.json.gz> <new.json.gz> <expected_hash_changes>

Masks ``meta.lightgbm`` and each model case's ``model.model_sha256``, then
requires the two documents to be equal everywhere else (every case, every
field, every temporal case; types are compared, and floats by ``float.hex``
so the sign of zero counts). Exits 1 on any other difference, or when the
number of model cases whose hash changed is not *expected_hash_changes*.
"""

from __future__ import annotations

import copy
import gzip
import json
import sys
from typing import Any

MASK = "<masked>"


def load(path: str) -> dict[str, Any]:
    with gzip.open(path) as fh:
        data: dict[str, Any] = json.load(fh)
    return data


def masked(doc: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(doc)
    out["meta"]["lightgbm"] = MASK
    for case in out["cases"]:
        if "model" in case:
            case["model"]["model_sha256"] = MASK
    return out


def first_difference(a: Any, b: Any, path: str = "") -> str | None:
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return f"{path}: keys {sorted(a.keys() ^ b.keys())}"
        for k in a:
            found = first_difference(a[k], b[k], f"{path}.{k}")
            if found:
                return found
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            found = first_difference(x, y, f"{path}[{i}]")
            if found:
                return found
        return None
    if isinstance(a, float):
        # float.hex keeps what == drops: the sign of zero (-0.0 == 0.0), and
        # it compares NaN with NaN as the same value (round-2 finding 1).
        return None if a.hex() == b.hex() else f"{path}: {a.hex()} != {b.hex()}"
    return None if a == b else f"{path}: {a!r} != {b!r}"


def main() -> int:
    old, new, expected = load(sys.argv[1]), load(sys.argv[2]), int(sys.argv[3])
    models = [(a, b) for a, b in zip(old["cases"], new["cases"]) if "model" in a]
    changed = sum(a["model"]["model_sha256"] != b["model"]["model_sha256"] for a, b in models)
    diff = first_difference(masked(old), masked(new))
    print(
        f"model cases={len(models)} hash changes={changed} (expected {expected}); "
        f"meta.lightgbm {old['meta']['lightgbm']} -> {new['meta']['lightgbm']}; "
        f"other differences: {diff or 'none'}"
    )
    return 0 if diff is None and changed == expected else 1


if __name__ == "__main__":
    sys.exit(main())
