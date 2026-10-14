"""trial_failure -- what a failed round reports about its trials (H-0128).

Shared by :class:`~lizyml.tuning.tuner.Tuner` and the ``Model.tune`` facade so
the two cannot count differently. Nothing here is stored on the study: a
reason is reported when this call saw one, never promised for every failure.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

#: Each part of a rendered exception is cut to this many characters.
MAX_TEXT = 500


def _cut(text: str) -> str:
    return text if len(text) <= MAX_TEXT else text[:MAX_TEXT]


def describe_exception(exc: BaseException) -> str:
    """``"<type name>: <message>"``, never raising (a hostile ``__str__`` or
    metaclass must not turn a trial failure into another exception)."""
    try:
        name = type.__getattribute__(type(exc), "__name__")
        if not isinstance(name, str):
            name = "<unknown type>"
    except Exception:  # noqa: BLE001 - the rendering must not raise
        name = "<unknown type>"
    return f"{_cut(name)}: {safe_message(exc)}"


def safe_message(exc: BaseException) -> str:
    """``str(exc)`` cut to :data:`MAX_TEXT`, or ``<unprintable message>``
    when ``__str__`` raises or returns a non-str."""
    try:
        message = str(exc)
        if not isinstance(message, str):
            message = "<unprintable message>"
    except Exception:  # noqa: BLE001 - the rendering must not raise
        message = "<unprintable message>"
    return _cut(message)


def round_trial_states(
    trials: Iterable[Any], start: int, *, interrupted: int = 0
) -> dict[str, int]:
    """Counts of the trials numbered ``start`` or above.

    ``interrupted`` is the number of trials this call moved from RUNNING to
    FAIL on reattach; they are counted in ``FAIL`` as well.
    """
    counts = {"trials": 0, "FAIL": 0, "PRUNED": 0, "interrupted": interrupted}
    for trial in trials:
        if trial.number < start:
            continue
        counts["trials"] += 1
        state = trial.state.name
        if state in ("FAIL", "PRUNED"):
            counts[state] += 1
    return counts


def states_text(states: dict[str, int], *, n_trials: int, timeout: float | None) -> str:
    """The counts as a phrase for an error message."""
    if states["trials"] == 0:
        return f"no trial ran (n_trials={n_trials}, timeout={timeout})"
    parts = [f"{states['trials']} trial(s): {states['FAIL']} FAIL"]
    if states["interrupted"]:
        parts.append(f"{states['interrupted']} of them interrupted")
    parts.append(f"{states['PRUNED']} PRUNED")
    return ", ".join(parts)
