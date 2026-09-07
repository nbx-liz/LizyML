"""Enumerate every place one parameter dict meets another inside ``lizyml/``.

The defect class this run met in most rounds is "two parameter dicts joined by
spelling, where the estimator resolves aliases". The population is finite, so it
is cheaper to enumerate it than to meet it once more. This scan is the
regeneration check for the table in HISTORY.md under H-0094 decision 8: run it
and compare, rather than trusting a table copied into prose (DC3).

    .venv/bin/python docs/audits/2026-09-defect-discovery/instruments/parameter_merge_seams.py

**What it looks for, and why the set is this wide.** Round 12's first version
declared four constructs -- ``{**a, **b}``, ``.update``, ``.setdefault`` and
``|`` -- and the rounds 11-12 monitor pointed out that two of the three defects
that round reported live in a fifth: plain ``d[key] = value``. Those two were
found by reading code adjacent to a declared construct, so the population was
closed by proximity rather than by the scan. Assignment into a subscript, and
keyword-splatting a parameter dict into a call, are included here for that
reason.

**What it cannot do**, stated rather than left to be discovered: the hint-word
filter is a heuristic over identifier text, not a type analysis. A parameter
dict held in a variable named none of the hint words is invisible to it. The
filter is listed in ``HINTS`` so the claim can be checked, and the output is a
candidate list for reading, not a verdict.
"""

from __future__ import annotations

import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4] / "lizyml"

#: Identifier substrings that mark an expression as parameter-shaped. Listed
#: rather than described, because "the scan missed it" must be checkable.
HINTS = (
    "param",
    "merged",
    "effective",
    "fixed",
    "override",
    "defaults",
    "space",
    "resolved",
    "smart",
)

#: Helpers that are themselves the answer, so a call to one is a resolved seam.
RESOLVERS = frozenset(
    {
        "overlay_params",
        "canonicalise_calibration_params",
        "check_duplicate_identities",
        "check_duplicate_space_dimensions",
    }
)


def _is_parameter_shaped(node: ast.AST) -> bool:
    return any(hint in ast.unparse(node).lower() for hint in HINTS)


class SeamFinder(ast.NodeVisitor):
    """Collect every expression where a mapping is joined, updated, or written."""

    def __init__(self, source: str) -> None:
        self.lines = source.splitlines()
        self.hits: set[tuple[int, str, str]] = set()

    def _record(self, node: ast.AST, kind: str) -> None:
        lineno = getattr(node, "lineno", 0)
        self.hits.add((lineno, kind, self.lines[lineno - 1].strip()))

    def visit_Dict(self, node: ast.Dict) -> None:
        if any(key is None for key in node.keys) and _is_parameter_shaped(node):
            self._record(node, "dict-unpack")
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.BitOr) and _is_parameter_shaped(node):
            self._record(node, "dict |")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        # `d[key] = value` -- the construct the first version of this scan did
        # not declare, and in which two of round 12's three defects lived.
        for target in node.targets:
            if isinstance(target, ast.Subscript) and _is_parameter_shaped(target):
                self._record(node, "subscript-assign")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in {"update", "setdefault"}:
            if _is_parameter_shaped(node):
                self._record(node, f".{func.attr}()")
        if isinstance(func, ast.Name) and func.id in RESOLVERS:
            self._record(node, func.id)
        if isinstance(func, ast.Name) and func.id == "dict" and node.keywords:
            if _is_parameter_shaped(node):
                self._record(node, "dict(a, **b)")
        # A parameter dict splatted into a call joins whatever the callee has.
        for keyword in node.keywords:
            if keyword.arg is None and _is_parameter_shaped(keyword.value):
                self._record(node, "call **kwargs")
        self.generic_visit(node)


def main() -> int:
    total = 0
    by_kind: dict[str, int] = {}
    for path in sorted(ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        finder = SeamFinder(source)
        finder.visit(ast.parse(source))
        rel = path.relative_to(ROOT.parent)
        for lineno, kind, text in sorted(finder.hits):
            print(f"{rel}:{lineno}  {kind}: {text}")
            by_kind[kind] = by_kind.get(kind, 0) + 1
            total += 1

    print()
    for kind, count in sorted(by_kind.items()):
        print(f"{kind:20} {count}")
    print(f"\n{total} candidate merge expressions over parameter-shaped values")
    print(
        "Each is classified by hand in HISTORY.md, H-0094 decision 8. This scan "
        "produces the candidates; it does not decide which are cross-source."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
