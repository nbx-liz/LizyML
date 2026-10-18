"""Summarize the H-0132 firing-rate log written by ``sitecustomize.py``.

Counts every ``Evaluator.evaluate`` call, the calls whose splits put a covered
row in two or more validation folds (where MASE's refusal would fire), and,
separately, the tests that build such splits on purpose (the refusal's own
tests). Error lines are reported, not skipped.

Usage: python3 summarize.py <log.jsonl> [<log.jsonl> ...]
"""

from __future__ import annotations

import json
import sys
from collections import Counter

#: Tests that construct overlapping validation folds to exercise the refusal.
DELIBERATE = (
    "tests/test_evaluation/test_mase_evaluator.py::test_overlapping_",
    "tests/test_core/test_mase_model.py::test_fit_refuses_overlapping_",
    "tests/test_core/test_mase_model.py::test_tune_fails_every_trial_on_overlapping_",
)
#: Tests that construct a constant outer-train target to exercise the refusal.
DELIBERATE_ZERO = ("tests/test_evaluation/test_mase_evaluator.py::test_a_constant_outer_train_",)


def main(paths: list[str]) -> int:
    calls = errors = overlapping = deliberate = with_mase = 0
    regression_calls = regression_folds = zero_deliberate = zero_elsewhere = 0
    sources: Counter[str] = Counter()
    unexpected: list[str] = []
    for path in paths:
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            calls += 1
            if "error" in rec:
                errors += 1
                unexpected.append(f"error: {rec['error']} ({rec.get('test') or rec.get('argv0')})")
                continue
            with_mase += bool(rec["needs_train_target"])
            sources["pytest" if rec["test"] else "notebook kernel"] += 1
            zero = rec.get("zero_scale_folds")
            if zero is not None:
                regression_calls += 1
                regression_folds += rec["n_folds"]
                if zero:
                    if rec["test"].startswith(DELIBERATE_ZERO):
                        zero_deliberate += zero
                    else:
                        zero_elsewhere += zero
                        unexpected.append(f"zero scale x{zero}: {rec['test'] or rec['argv0']}")
            if rec["overlapping_rows"] > 0:
                if rec["test"].startswith(DELIBERATE):
                    deliberate += 1
                else:
                    overlapping += 1
                    unexpected.append(f"overlap: {rec['test'] or rec['argv0']}")
    print(f"Evaluator.evaluate calls: {calls} ({dict(sources)}), errors: {errors}")
    print(f"calls requesting a training-scaled metric (mase): {with_mase}")
    print(f"calls with overlapping validation folds, deliberate refusal tests: {deliberate}")
    print(f"calls with overlapping validation folds, elsewhere: {overlapping}")
    print(f"regression calls: {regression_calls}, their folds: {regression_folds}")
    print(f"folds with a zero or non-finite outer-train scale, deliberate refusal tests: {zero_deliberate}")
    print(f"folds with a zero or non-finite outer-train scale, elsewhere: {zero_elsewhere}")
    for item in unexpected:
        print(f"  {item}")
    return 1 if errors or overlapping else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
