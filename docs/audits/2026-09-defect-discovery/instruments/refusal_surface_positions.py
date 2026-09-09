"""Enumerate the positions the surface-naming rule binds, and their compliance.

H-0094 decision 3 declared that a refusal names the input the user has to change:
the same wrong parameter written in ``model.params``, in ``fit(params=)`` and in
``calibration.params`` must be reported against three different addresses. The
rule was implemented by threading a ``surface`` argument through the checks that
run at those entrances.

The rule binds more positions than the proposal named. #286 is one of them --
``_pop_by_identity`` refuses a duplicate spelling without saying where it came
from -- and it was found by a review round rather than by looking. This script
looks, so that the Proposal for PR 2b can state the whole population instead of
the one position somebody happened to notice.

Run from the repository root::

    uv run python docs/audits/2026-09-defect-discovery/instruments/refusal_surface_positions.py

What it derives, mechanically:

* every function in ``lizyml/`` that raises ``LizyMLError`` with
  ``ErrorCode.CONFIG_INVALID``;
* whether **the raise itself** names a surface.

The question is asked of the raise and not of the signature, and the first
version of this script asked the signature. That was a false negative of exactly
the class this run keeps hunting: ``check_param_names`` names the surface for
every unknown name, but it receives it inside ``named: Iterable[tuple[str, str]]``
rather than as a ``surface`` keyword, so a signature test reported a complying
position as a candidate. It was caught by reading one of the sites the script
flagged. A scan that reports "did not look" as "not compliant" is the mirror of
DC1 and is just as misleading to whatever consumes it.

What it does **not** derive, and why: whether a refusal that names no surface is
*reachable* from more than one surface. A refusal reachable from exactly one
input does not need to name it -- the address is implied, and most of these are
in that position (a plot function, the config reader, a splitter). Deciding it
needs the call graph and a judgement about which entrances exist, so this script
lists those sites and leaves the disposition to the Proposal rather than
guessing.
"""

from __future__ import annotations

import ast
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[4]
PACKAGE = REPO / "lizyml"


def _raises_config_invalid(node: ast.AST) -> list[ast.Raise]:
    """Return the CONFIG_INVALID raises lexically inside *node*."""
    found: list[ast.Raise] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Raise) or child.exc is None:
            continue
        text = ast.dump(child.exc)
        if "CONFIG_INVALID" in text and "LizyMLError" in text:
            found.append(child)
    return found


def _mentions_surface(raise_node: ast.Raise) -> bool:
    """Does this raise put the surface into the message or the context?"""
    for child in ast.walk(raise_node):
        if isinstance(child, ast.Name) and child.id == "surface":
            return True
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            if "surface" in child.value:
                return True
    return False


def main() -> int:
    naming: list[str] = []
    partial: list[str] = []
    silent: list[str] = []

    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            raises = _raises_config_invalid(node)
            if not raises:
                continue
            where = f"{path.relative_to(REPO)}:{node.lineno} {node.name}"
            named = [r for r in raises if _mentions_surface(r)]
            if len(named) == len(raises):
                naming.append(f"{where} ({len(raises)} raises)")
            elif named:
                quiet = [r.lineno for r in raises if not _mentions_surface(r)]
                partial.append(f"{where} -- silent at {quiet}")
            else:
                silent.append(f"{where} ({len(raises)} raises)")

    total = len(naming) + len(partial) + len(silent)
    print(f"CONFIG_INVALID raising functions: {total}\n")

    print(f"EVERY RAISE NAMES A SURFACE ({len(naming)})")
    for line in naming:
        print(f"  {line}")

    print(f"\nSOME RAISES NAME A SURFACE AND SOME DO NOT ({len(partial)})")
    for line in partial or ["none"]:
        print(f"  {line}")

    print(f"\nNO RAISE NAMES A SURFACE ({len(silent)})")
    print("  -- reachability from more than one entrance is NOT derived here.")
    print("     A refusal reachable from one entrance does not need to name it;")
    print("     each of these needs a disposition in the Proposal.")
    for line in silent:
        print(f"  {line}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
