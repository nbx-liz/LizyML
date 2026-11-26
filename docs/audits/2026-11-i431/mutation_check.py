"""Negative control for full_compare.py (round-1 review finding 1).

    python mutation_check.py <oracle.json.gz> <out_dir>

Writes copies of *oracle* that each change one non-masked field (an encoder
field of the last model case, a model case's ``pred_hex``, a temporal case,
one non-lightgbm ``meta`` value) and prints full_compare.py's exit code for
each; every mutant must exit 1, the unmutated copy 0.
"""

from __future__ import annotations

import copy
import gzip
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent


def write(doc: dict[str, Any], path: Path) -> None:
    with gzip.open(path, "wt") as fh:
        json.dump(doc, fh)


def flip_first_leaf(node: Any) -> bool:
    """Change the first scalar found inside *node*; True when one changed."""
    items = node.items() if isinstance(node, dict) else enumerate(node)
    for key, value in items:
        if isinstance(value, (dict, list)):
            if flip_first_leaf(value):
                return True
        else:
            node[key] = f"{value!r}-mutated"
            return True
    return False


def flip_zero_sign(node: Any) -> bool:
    """Flip the sign of the first float zero inside *node* (round-2 finding 1)."""
    items = node.items() if isinstance(node, dict) else enumerate(node)
    for key, value in items:
        if isinstance(value, (dict, list)):
            if flip_zero_sign(value):
                return True
        elif isinstance(value, float) and value == 0.0:
            node[key] = -value
            return True
    return False


def sign_mutant(base: dict[str, Any], field: str) -> tuple[str, dict[str, Any]]:
    """A copy whose first model case holding a float zero in *field* has it negated."""
    m = copy.deepcopy(base)
    for case in m["cases"]:
        if "model" not in case:
            continue
        node = case["model"][field] if field == "pandas_categorical" else case[field]
        if node is not None and flip_zero_sign(node):
            return case["id"], m
    raise AssertionError(f"no model case holds a float zero in {field}")


def main() -> int:
    with gzip.open(sys.argv[1]) as fh:
        base: dict[str, Any] = json.load(fh)
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    last_model = max(i for i, c in enumerate(base["cases"]) if "model" in c)
    mutants: dict[str, dict[str, Any]] = {"unmutated": copy.deepcopy(base)}
    m = copy.deepcopy(base)
    case = m["cases"][last_model]
    assert flip_first_leaf(case["encoder"])
    mutants["encoder_field_of_last_model_case"] = m
    m = copy.deepcopy(base)
    m["cases"][last_model]["dtype"] += "-mutated"
    mutants["dtype_of_last_model_case"] = m
    m = copy.deepcopy(base)
    m["cases"][last_model]["model"]["pred_hex"][0] = "0x0p+0"
    mutants["pred_hex"] = m
    m = copy.deepcopy(base)
    assert flip_first_leaf(m["temporal_cases"][-1])
    mutants["temporal_case"] = m
    for field in ("encoder", "pandas_categorical"):
        case_id, m = sign_mutant(base, field)
        mutants[f"zero_sign_in_{field}_of_{case_id}"] = m
    m = copy.deepcopy(base)
    m["meta"]["numpy"] = "0.0.0"
    mutants["meta_numpy"] = m
    status = 0
    for name, doc in mutants.items():
        path = out / f"{name}.json.gz"
        write(doc, path)
        rc = subprocess.run(
            [sys.executable, str(HERE / "full_compare.py"), sys.argv[1], str(path), "0"],
            capture_output=True,
            check=False,
        ).returncode
        want = 0 if name == "unmutated" else 1
        print(f"{name}: exit {rc} (want {want})")
        status |= rc != want
    return status


if __name__ == "__main__":
    sys.exit(main())
