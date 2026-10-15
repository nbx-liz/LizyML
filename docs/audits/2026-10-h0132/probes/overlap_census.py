"""H-0132 measurement: can a shipped splitter put one row in two validation folds?

MASE's pooled OOF value scales each row by the training scale of the fold that
validated it. That needs each covered row to belong to exactly one validation
fold. This probe runs every splitter in ``lizyml/splitters/`` over a grid of
its parameters on synthetic indices (no training) and reports, per splitter,
how many configurations produced a row in two or more validation folds.

Usage (repository root): uv run python docs/audits/2026-10-h0132/probes/overlap_census.py
Exit status 1 if any configuration overlaps.
"""

from __future__ import annotations

import itertools
import sys
import warnings
from collections import Counter

import numpy as np

from lizyml.splitters.blocked_group_kfold import BlockedGroupKFoldSplitter
from lizyml.splitters.group_kfold import GroupKFoldSplitter, StratifiedGroupKFoldSplitter
from lizyml.splitters.group_time_series import GroupTimeSeriesSplitter
from lizyml.splitters.kfold import KFoldSplitter, StratifiedKFoldSplitter
from lizyml.splitters.purged_time_series import PurgedTimeSeriesSplitter
from lizyml.splitters.time_series import TimeSeriesSplitter

N = 240
rng = np.random.default_rng(0)
y_cls = rng.integers(0, 2, N)
groups = np.repeat(np.arange(40), N // 40)
blocks = np.repeat(np.arange(12), N // 12)


def overlaps(splitter, y=None, g=None) -> bool:
    seen = Counter()
    for _, valid in splitter.split(N, y, g):
        seen.update(np.asarray(valid).tolist())
    return any(c > 1 for c in seen.values())


def configs():
    for n, shuffle in itertools.product([2, 3, 5], [True, False]):
        yield "kfold", KFoldSplitter(n_splits=n, shuffle=shuffle), None, None
        yield "stratified_kfold", StratifiedKFoldSplitter(n_splits=n, shuffle=shuffle), y_cls, None
    for n in [2, 3, 5]:
        yield "group_kfold", GroupKFoldSplitter(n_splits=n), None, groups
        yield "stratified_group_kfold", StratifiedGroupKFoldSplitter(n_splits=n), y_cls, groups
    for n, gap, mtr, mts in itertools.product([2, 3, 5], [0, 5], [None, 50], [None, 20]):
        yield "time_series", TimeSeriesSplitter(n, gap, mtr, mts), None, None
        yield "group_time_series", GroupTimeSeriesSplitter(n, gap, mtr, mts), None, groups
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for embargo in [None, 3]:
                yield (
                    "purged_time_series",
                    PurgedTimeSeriesSplitter(n, gap, embargo, mtr, mts),
                    None,
                    None,
                )
    for cutoffs, mode, window, n in itertools.product(
        [[4], [3, 6], [2, 5, 8], [3, 6, 9]], ["expanding", "sliding"], [None, 3], [2, 3]
    ):
        if mode == "sliding" and window is None:
            continue
        yield (
            "blocked_group_kfold",
            BlockedGroupKFoldSplitter(
                blocks, cutoffs, mode=mode, train_window=window, n_splits=n, random_state=0
            ),
            None,
            groups,
        )


class _Overlapping:
    """Negative control: two folds share row 1 (the shape of
    tests/test_training/test_oof_coverage.py::test_overlapping_valid_idx)."""

    def split(self, n_samples, y=None, groups=None):
        yield np.array([2, 3, 4]), np.array([0, 1])
        yield np.array([0, 3, 4]), np.array([1, 2])


def main() -> int:
    if not overlaps(_Overlapping()):
        print("negative control NOT caught: the overlap check is blind")
        return 2
    print("negative control caught: overlapping folds are detected")
    total: Counter[str] = Counter()
    bad: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    for name, splitter, y, g in configs():
        total[name] += 1
        try:
            if overlaps(splitter, y, g):
                bad[name] += 1
        except Exception as exc:  # report, never skip silently
            errors[name] += 1
            print(f"  error {name}: {type(exc).__name__}: {exc}")
    for name in sorted(total):
        print(f"{name}: {bad[name]}/{total[name]} configurations overlap, {errors[name]} errors")
    print(f"overlapping configurations: {sum(bad.values())}/{sum(total.values())}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
