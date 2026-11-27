"""MemoryEstimate -- the public result types of the memory estimates (H-0150).

``MemoryEstimate`` is what one fit asks the estimator library to allocate;
``TuneMemoryEstimate`` is the envelope over the next tune round. Both are
estimates of the **extra CPU memory training adds**, not a bound on the
process's resident set (H-0140, H-0150 policy 6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: Why ``num_leaves`` has its value: ``"derived"`` (``auto_num_leaves`` from
#: ``max_depth``), ``"explicit"`` (set in the parameters) or ``"default"``
#: (the library's own).
MemoryCause = Literal["derived", "explicit", "default"]


@dataclass(frozen=True)
class MemoryEstimate:
    """The approximate extra CPU memory of one fit, and why.

    Attributes:
        num_leaves: The ``num_leaves`` the estimator library receives.
        cause: Why ``num_leaves`` has that value.
        max_depth: The ``max_depth`` the library receives, or ``None``.
        estimate_bytes: The approximate extra CPU memory, in bytes.
        n_boosters: The trained models kept alive at once that the estimate
            counts (outer folds + 1 for a fit).
    """

    num_leaves: int
    cause: MemoryCause
    max_depth: int | None
    estimate_bytes: int
    n_boosters: int


@dataclass(frozen=True)
class TuneMemoryEstimate:
    """The memory envelope over the next tune round (H-0150).

    ``status == "complete"`` promises only this: while the study does not
    change, ``envelope.estimate_bytes`` bounds from above the estimate of every
    ordinary CPU trial sampled from the next round's effective ranges. It does
    not say which point the sampler draws, and it does not cover WAITING
    trials, values outside the ranges or non-CPU devices. An ``"incomplete"``
    envelope is only the maximum over the points that could be evaluated and is
    not an upper bound: it must not clear a whole-job admission check.

    Attributes:
        status: ``"complete"`` or ``"incomplete"``.
        envelope: The largest estimate over the evaluated candidate points, or
            ``None`` when the round runs no trial.
        unresolved: Names of dimensions whose bound could not be decided
            (``device_type`` when the device cannot be decided).
        waiting_trials: Every WAITING row in the target study, whichever round
            it belongs to, and also when the next round runs no trial (then
            it does not change ``status``).
        round: The number of the round estimated.
        n_candidates: How many candidate points were evaluated.
        failures: ``"<ExceptionType>: <message>"`` of each point whose
            evaluation raised.
    """

    status: Literal["complete", "incomplete"]
    envelope: MemoryEstimate | None
    unresolved: tuple[str, ...]
    waiting_trials: int
    round: int
    n_candidates: int
    failures: tuple[str, ...]
