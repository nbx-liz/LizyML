"""Summarize the H-0132 firing-rate log written by ``sitecustomize.py``.

Counts, per process role (pytest, notebook kernel):

- ``evaluate`` records: the calls whose splits put a covered row in two or more
  validation folds (where the overlap refusal fires), and the regression folds
  whose outer-train scale is zero or not finite;
- ``bind`` records: every ``MASE.bind_train`` call and the ones that refused.

Events inside the tests that build these inputs on purpose are counted apart.
Any other event, and any ``error`` record, makes the exit status 1.
``--self-check`` feeds one synthetic record per unexpected category and fails
unless each one turns the exit status to 1 (and a clean log to 0).

Usage: python3 summarize.py <log.jsonl> [<log.jsonl> ...]
       python3 summarize.py --self-check
"""

from __future__ import annotations

import json
import sys
from collections import Counter

#: Tests that construct overlapping validation folds to exercise the refusal.
DELIBERATE_OVERLAP = (
    "tests/test_evaluation/test_mase_evaluator.py::test_overlapping_",
    "tests/test_core/test_mase_model.py::test_fit_refuses_overlapping_",
    "tests/test_core/test_mase_model.py::test_tune_fails_every_trial_on_overlapping_",
)
#: Tests that give MASE a constant or non-finite training target on purpose.
DELIBERATE_ZERO = (
    "tests/test_evaluation/test_mase_evaluator.py::test_a_constant_outer_train_",
    "tests/test_metrics/test_mase.py::test_a_zero_or_non_finite_scale_is_refused",
    "tests/test_estimators/test_mase_feval.py::test_a_constant_training_target_is_refused_",
)


def summarize(records: list[dict]) -> tuple[list[str], int]:
    c: Counter[str] = Counter()
    unexpected: list[str] = []
    for rec in records:
        where = rec.get("test") or rec.get("argv0", "")
        role = rec.get("role", "?")
        if "error" in rec:
            unexpected.append(f"error ({role}): {rec['error']} at {where}")
            continue
        if rec.get("kind") == "evaluate":
            c[f"evaluate[{role}]"] += 1
            if rec["overlapping_rows"] > 0:
                if where.startswith(DELIBERATE_OVERLAP):
                    c["overlap deliberate"] += 1
                else:
                    unexpected.append(f"overlap ({role}): {where}")
            zero = rec.get("zero_scale_folds")
            if zero is not None:
                c["regression evaluate"] += 1
                c["regression folds"] += rec["n_folds"]
                if zero:
                    if where.startswith(DELIBERATE_ZERO):
                        c["zero-scale folds deliberate"] += zero
                    else:
                        unexpected.append(f"zero-scale folds x{zero} ({role}): {where}")
        elif rec.get("kind") == "bind":
            c[f"bind[{role}]"] += 1
            if rec.get("refused"):
                if where.startswith(DELIBERATE_ZERO):
                    c["bind refused deliberate"] += 1
                else:
                    unexpected.append(f"bind refused ({role}): {where}")
        else:
            unexpected.append(f"unknown record kind: {rec.get('kind')!r} at {where}")
    lines = [f"{key}: {c[key]}" for key in sorted(c)]
    lines.append(f"unexpected events: {len(unexpected)}")
    lines += [f"  {item}" for item in unexpected]
    return lines, 1 if unexpected else 0


def _self_check() -> int:
    base = {"role": "pytest", "test": "tests/x.py::t (call)", "argv0": "pytest"}
    clean = {**base, "kind": "evaluate", "n_folds": 2, "overlapping_rows": 0, "zero_scale_folds": 0}
    cases = {
        "overlap": {**clean, "overlapping_rows": 1},
        "zero-scale fold": {**clean, "zero_scale_folds": 1},
        "bind refused": {**base, "kind": "bind", "n_train": 3, "refused": "LizyMLError"},
        "error": {**base, "kind": "bind", "error": "TypeError: x"},
        "unknown kind": {**base, "kind": "other"},
    }
    ok = summarize([clean])[1] == 0
    print(f"self-check clean log -> {'0 (ok)' if ok else 'NONZERO (broken)'}")
    for name, rec in cases.items():
        status = summarize([clean, rec])[1]
        print(f"self-check {name} -> {'1 (ok)' if status == 1 else '0 (BROKEN)'}")
        ok = ok and status == 1
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    if argv == ["--self-check"]:
        return _self_check()
    records = [json.loads(line) for path in argv for line in open(path, encoding="utf-8")]
    lines, status = summarize(records)
    print("\n".join(lines))
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
