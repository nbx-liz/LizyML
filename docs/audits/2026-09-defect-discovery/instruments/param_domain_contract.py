"""Regenerate the H-0095 accepted-set block, and list the sinks it feeds.

H-0095 declares a set of parameter values and refuses everything else at the
surface. Twenty-four review rounds moved that set nineteen times, and the
proposal's own table was two rows out of date within a day of being written --
a table copied into prose goes stale while the code moves, which is DC3.

So the block that states the set is **generated from the module** and pasted
into `HISTORY.md` between the markers below. Running this with `--check`
compares the two and exits non-zero when they have drifted.

    uv run python docs/audits/2026-09-defect-discovery/instruments/param_domain_contract.py
    uv run python docs/audits/2026-09-defect-discovery/instruments/param_domain_contract.py --check

**The second half is a candidate list, not a closure.** The consumers of a
normalised value are the places one is handed to something that writes or
trains: `lgb.train`, `json.dump`, `joblib.dump`. This scan finds calls by name,
which is a heuristic over identifier text and not a type analysis -- a sink
reached through an alias, a wrapper, or a name none of `SINK_NAMES` matches is
invisible to it. The same claim was made twice about `parameter_merge_seams.py`
and falsified twice within a round of being made, so it is not made here:
**what the proposal asserts is the requirement list, each requirement executed
over the whole accepted population. This list is where those requirements came
from, not a proof that no other consumer exists.**
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import platform
import sys
from typing import Any

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from lizyml.core.exceptions import LizyMLError  # noqa: E402
from lizyml.core.param_domain import (  # noqa: E402
    NUMPY_SCALAR_TYPES,
    PATH_TYPES,
    PLAIN_SCALAR_TYPES,
    PLAIN_SEQUENCE_TYPES,
    REFUSED_SEQUENCE_TYPES,
    normalise_params,
)

HISTORY = REPO / "HISTORY.md"
BEGIN = "<!-- param-domain-contract:begin -->"
END = "<!-- param-domain-contract:end -->"

#: Call names that hand a value to something outside this process: the trainer,
#: and the two serialisers an artifact is written with. Listed rather than
#: described, so "the scan missed it" is checkable.
SINK_NAMES = ("train", "dump", "dumps")

#: Qualifiers that make one of the names above a sink rather than a homonym.
#:
#: `lgbm` is here because the first version of this scan did not have it, and
#: that version reported one `lgb.train` where the code has two: the calibrator
#: imports the same library as `lgbm` and the scan could not see its call. It
#: was found by comparing the output against a site already known from the exit
#: assertion, which is the whole argument for calling this a candidate list --
#: an alias nobody has thought of is invisible in exactly the same way.
SINK_QUALIFIERS = ("lgb", "lgbm", "lightgbm", "json", "joblib")


def _names(kinds: object) -> str:
    return ", ".join(sorted(kind.__name__ for kind in kinds))  # type: ignore[union-attr]


def _numpy_names(kinds: object) -> str:
    """Qualified, because numpy 2 calls its boolean scalar type ``bool``.

    An unqualified list would print ``bool`` for both the Python type in the
    scalar row and the numpy one here, and the two are accepted by different
    gates for different reasons.
    """
    return ", ".join(
        sorted(f"numpy.{kind.__name__}" for kind in kinds)  # type: ignore[union-attr]
    )


#: Values tried, in order, when constructing one instance of a numpy type.
PROBE_SEEDS = (1, 0, True, "x")


def _probe(kind: type, *, element: bool) -> bool:
    """Is a constructed value of ``kind`` accepted in that position?"""
    for seed in PROBE_SEEDS:
        try:
            value = kind(seed)
        except (TypeError, ValueError, OverflowError):
            continue
        break
    else:
        return False
    payload: Any = [value] if element else value
    try:
        normalise_params({"k": payload}, surface="probe")
    except LizyMLError:
        return False
    return True


def _refused_probes(*, element: bool) -> str:
    """The admitted types whose probe value this position refuses.

    Derived rather than noted, because the type set and the value set are not
    the same set and the difference moves. ``timedelta64`` is admitted as a type
    -- it is a ``numpy.integer``, so the dtype round trip keeps it -- and its
    values are refused by the ``format`` check. ``longdouble`` is a second case
    and an asymmetric one: ``.item()`` on it returns a ``longdouble`` rather
    than a Python float, so the scalar position refuses it, while the element
    position parses its text back and accepts. A line written by hand would
    have named the first and missed the second.
    """
    refused = [k for k in NUMPY_SCALAR_TYPES if not _probe(k, element=element)]
    return _numpy_names(refused) if refused else "none"


def _accepted_block() -> str:
    """The accepted set, read out of the module rather than described."""
    lines = [
        BEGIN,
        "```text",
        f"numpy               {np.__version__}",
        f"platform            {sys.platform} {platform.machine()}",
        "",
        "scalar position     " + _names(PLAIN_SCALAR_TYPES),
        "  converted         " + _names(PATH_TYPES) + " -> str",
        "  converted         numpy scalar -> .item(), checked by format()",
        "element position    " + _names(PLAIN_SCALAR_TYPES),
        "  converted         " + _names(PATH_TYPES) + " -> str",
        "  converted         numpy scalar -> the plain value printing as str(x)",
        "sequence            " + _names(PLAIN_SEQUENCE_TYPES) + ", 1-D ndarray",
        "  member            scalar, list (depth 2 only), dict",
        "mapping             dict with exact-str keys, values normalised",
        "refused sequence    " + _names(REFUSED_SEQUENCE_TYPES),
        "",
        "numpy scalar types (exact type; derived by np.dtype(k).type is k)",
        "  " + _numpy_names(NUMPY_SCALAR_TYPES),
        "",
        "  the type set is wider than the value set, and by position:",
        "  refused in scalar position   " + _refused_probes(element=False),
        "  refused in element position  " + _refused_probes(element=True),
        "  (one constructed value per type -- a measurement of these values,",
        "  not a proof about every value of the type)",
        "```",
        END,
    ]
    return "\n".join(lines)


def _is_sink(node: ast.Call) -> str | None:
    """The sink this call is, or ``None``."""
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr not in SINK_NAMES:
        return None
    owner = func.value
    if not isinstance(owner, ast.Name) or owner.id not in SINK_QUALIFIERS:
        return None
    return f"{owner.id}.{func.attr}"


def _sinks() -> list[tuple[str, int, str]]:
    found: list[tuple[str, int, str]] = []
    for path in sorted((REPO / "lizyml").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                sink = _is_sink(node)
                if sink is not None:
                    rel = path.relative_to(REPO)
                    found.append((str(rel), node.lineno, sink))
    return found


def _check() -> int:
    text = HISTORY.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        print(f"{HISTORY.name} carries no generated block; paste the one below.")
        return 1
    start = text.index(BEGIN)
    stop = text.index(END) + len(END)
    written = text[start:stop]
    generated = _accepted_block()
    if written == generated:
        print("HISTORY.md matches the module.")
        return 0
    print("HISTORY.md has drifted from the module. Written:\n")
    print(written)
    print("\nGenerated:\n")
    print(generated)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare the block in HISTORY.md against the module",
    )
    args = parser.parse_args()
    if args.check:
        return _check()

    print(_accepted_block())
    print("\nsink candidates (see the docstring: candidates, not a closure)\n")
    for path, line, sink in _sinks():
        print(f"  {path}:{line}  {sink}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
