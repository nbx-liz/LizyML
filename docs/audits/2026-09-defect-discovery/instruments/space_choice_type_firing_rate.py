"""Measure the population a stricter ``choices`` gate would newly refuse.

``_validate_categorical_choices`` admits ``(NoneType, bool, int, float, str)`` by
``isinstance``. Two numpy scalar types **subclass** a Python scalar and therefore
pass it -- ``np.float64`` and ``np.str_`` -- after which the exit assertion in the
adapter refuses them inside every trial and the user sees ``TUNING_FAILED``.

Judging the type by identity instead closes that. Narrowing an ``allow``
condition on a public Config surface needs measured evidence first (Change Gate,
Conditional-Activation Evidence): how many search spaces in the shipped suite
carry a choice whose type is one of those two? Each one is a config that works
today and would stop working.

Run from the repository root, with this directory on PYTHONPATH::

    PYTHONPATH=docs/audits/2026-09-defect-discovery/instruments \\
    SPACE_CHOICE_OUT=/path/to/report.txt \\
    uv run pytest tests -q --no-cov -p space_choice_type_firing_rate

It wraps the validator rather than reading configs from the tree, because the
suite builds most of its spaces in code. Every space that reaches ``parse_space``
is therefore in the population, including the default spaces the library
substitutes when the user writes none.

Measured at 22b11b3 on 2026-09-09: see the recorded result in HISTORY.md, H-0095,
the decision on the search space.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Any

COUNTS: Counter[str] = Counter()

#: The choices that pass ``isinstance`` and would be refused by identity.
WOULD_NEWLY_REFUSE: list[Any] = []

#: The choices already refused today, kept so the report shows the gate working
#: rather than only the hole in it.
ALREADY_REFUSED: list[Any] = []

PLAIN = (type(None), bool, int, float, str)


def _test_id() -> str:
    return os.environ.get("PYTEST_CURRENT_TEST", "?").split(" ")[0]


def _classify(name: str, choices: Any) -> None:
    COUNTS["categorical dimensions observed"] += 1
    for index, value in enumerate(choices):
        COUNTS["choices observed"] += 1
        exact = any(type(value) is plain for plain in PLAIN)
        loose = isinstance(value, PLAIN)
        if exact:
            COUNTS["choice: plain Python scalar"] += 1
        elif loose:
            # Admitted today, refused by an identity check: the population this
            # measurement exists for.
            COUNTS["choice: SUBCLASS of a plain scalar (newly refused)"] += 1
            WOULD_NEWLY_REFUSE.append(
                (_test_id(), name, index, type(value).__name__, repr(value)[:60])
            )
        else:
            COUNTS["choice: already refused today"] += 1
            ALREADY_REFUSED.append(
                (_test_id(), name, index, type(value).__name__, repr(value)[:60])
            )


def pytest_configure(config):  # noqa: ARG001
    from lizyml.tuning import search_space

    original = search_space._validate_categorical_choices

    def wrapped(name, choices):
        try:
            _classify(name, choices)
        except Exception:  # noqa: BLE001 -- measuring must not change behaviour
            COUNTS["classification raised"] += 1
        return original(name, choices)

    search_space._validate_categorical_choices = wrapped


def pytest_terminal_summary(terminalreporter, *args, **kwargs):  # noqa: ARG001
    lines = [f"{key}: {value}" for key, value in sorted(COUNTS.items())]
    lines.append("")
    lines.append(f"WOULD BE NEWLY REFUSED: {len(WOULD_NEWLY_REFUSE)}")
    for item in WOULD_NEWLY_REFUSE[:60]:
        lines.append(f"  {item}")
    lines.append("")
    lines.append(f"ALREADY REFUSED TODAY: {len(ALREADY_REFUSED)}")
    for item in ALREADY_REFUSED[:20]:
        lines.append(f"  {item}")
    text = "\n".join(lines)
    terminalreporter.write_line("=== SPACE CHOICE TYPE MEASUREMENT ===")
    terminalreporter.write_line(text)
    out = os.environ.get("SPACE_CHOICE_OUT")
    if out:
        with open(out, "w") as handle:
            handle.write(text + "\n")
