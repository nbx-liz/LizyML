"""#268: for each census knob, which construction sites in ``lizyml/`` bind it.

Static half of the reachability measurement. For every class in the census
(``pr8_knob_census.py``), finds ``ClassName(...)`` calls under ``lizyml/`` and
records, per knob, whether a site passes it (by keyword, or positionally by
index) and the source text of the value. A ``**kwargs`` splat is reported as
such -- the value then comes from a mapping the static scan cannot read, which
the executed probe (``pr8_reachability_probe.py``) settles.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

from pr8_knob_census import REPO, census


def _init_params(class_name: str) -> list[str]:
    for path in (REPO / "lizyml").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef) and sub.name == "__init__":
                        return [a.arg for a in sub.args.args[1:]] + [
                            a.arg for a in sub.args.kwonlyargs
                        ]
    return []


def main() -> None:
    knobs = census()
    classes = sorted({k.split(".")[0] for k, _, _ in knobs})
    sites: dict[str, list[tuple[str, ast.Call]]] = defaultdict(list)
    for path in sorted((REPO / "lizyml").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(REPO)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            if name in classes:
                sites[name].append((f"{rel}:{node.lineno}", node))
    for knob, where, default in knobs:
        cls, param = knob.split(".")
        params = _init_params(cls)
        index = params.index(param) if param in params else -1
        found: list[str] = []
        for loc, call in sites.get(cls, []):
            bound = None
            for kw in call.keywords:
                if kw.arg == param:
                    bound = ast.unparse(kw.value)
                elif kw.arg is None:
                    bound = bound or f"**{ast.unparse(kw.value)}"
            if bound is None and 0 <= index < len(call.args):
                bound = ast.unparse(call.args[index])
            found.append(f"{loc} -> {bound if bound is not None else 'DEFAULT'}")
        if not sites.get(cls):
            found.append("no direct construction site (dynamic / registry / user)")
        print(f"{knob}  [default {default}]")
        for line in found:
            print(f"    {line}")


if __name__ == "__main__":
    main()
