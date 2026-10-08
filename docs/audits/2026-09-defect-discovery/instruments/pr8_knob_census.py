"""#268 census: every defaulted or keyword-only ``__init__`` parameter of a public class.

Same population rule as ``plan_population_recheck.py`` (the plan's 74): a class
whose name does not start with ``_``, anywhere under ``lizyml/``; its
``__init__``'s defaulted positional and keyword-only parameters. Prints one
line per knob (``Class.param  file:line  default``) and the total, so the
population is enumerable rather than a count.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]


def census() -> list[tuple[str, str, str]]:
    knobs: list[tuple[str, str, str]] = []
    for path in sorted((REPO / "lizyml").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(REPO)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name.startswith("_"):
                continue
            for sub in node.body:
                if not isinstance(sub, ast.FunctionDef) or sub.name != "__init__":
                    continue
                a = sub.args
                positional = a.args[len(a.args) - len(a.defaults) :]
                for arg, default in zip(positional, a.defaults, strict=True):
                    knobs.append((f"{node.name}.{arg.arg}", f"{rel}:{arg.lineno}", ast.unparse(default)))
                for arg, kw_default in zip(a.kwonlyargs, a.kw_defaults, strict=True):
                    shown = ast.unparse(kw_default) if kw_default is not None else "<required>"
                    knobs.append((f"{node.name}.{arg.arg}", f"{rel}:{arg.lineno}", shown))
    return knobs


if __name__ == "__main__":
    rows = census()
    for name, where, default in rows:
        print(f"{name:48s} {where:55s} {default}")
    print(f"total: {len(rows)}")
