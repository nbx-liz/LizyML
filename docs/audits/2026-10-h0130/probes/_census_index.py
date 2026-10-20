"""Shared AST index for the H-0130 census probes.

Moved here from amendment 1's R6 test helpers when H-0130 amendment 2
withdrew R6 and deleted that module; only the probes use it.
``lizyml/_version.py`` is generated and untracked, so it is left out to keep
the probes' output independent of the checkout.
"""

from __future__ import annotations

import ast
import builtins
from collections.abc import Mapping
from pathlib import Path

PACKAGE = "lizyml"


def module_name(root: Path, path: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join([PACKAGE, *parts])


def _package(module: str, path: Path) -> str:
    return module if path.name == "__init__.py" else module.rsplit(".", 1)[0]


def _bindings(module: str, path: Path, tree: ast.Module) -> dict[str, str]:
    """Local name -> dotted lizyml-or-external path it is bound to."""
    names: dict[str, str] = {}
    package = _package(module, path)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[: len(base) - node.level + 1]
                source = ".".join(base + ([node.module] if node.module else []))
            else:
                source = node.module or ""
            for alias in node.names:
                names[alias.asname or alias.name] = f"{source}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    names[alias.asname] = alias.name
                else:
                    top = alias.name.split(".")[0]
                    names[top] = top
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            names[node.name] = f"{module}.{node.name}"
    return names


def _dotted(expr: ast.expr, names: Mapping[str, str]) -> str | None:
    parts: list[str] = []
    while isinstance(expr, ast.Attribute):
        parts.insert(0, expr.attr)
        expr = expr.value
    if not isinstance(expr, ast.Name):
        return None
    if expr.id in names:
        return ".".join([names[expr.id], *parts])
    if hasattr(builtins, expr.id):
        return ".".join(["builtins", expr.id, *parts])
    return None


class _Index:
    """Every module's tree, bindings and module-level classes."""

    def __init__(self, root: Path) -> None:
        self.paths: dict[str, Path] = {}
        self.trees: dict[str, ast.Module] = {}
        self.names: dict[str, dict[str, str]] = {}
        self.classes: set[str] = set()
        for path in sorted(root.rglob("*.py")):
            if path == root / "_version.py":
                continue
            module = module_name(root, path)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            self.paths[module] = path
            self.trees[module] = tree
            self.names[module] = _bindings(module, path, tree)
            self.classes.update(
                f"{module}.{n.name}" for n in tree.body if isinstance(n, ast.ClassDef)
            )

    def canonical(self, dotted: str) -> str:
        """Follow one re-export hop: ``pkg.Name`` imported into ``pkg``."""
        if dotted in self.classes:
            return dotted
        owner, _, name = dotted.rpartition(".")
        hop = self.names.get(owner, {}).get(name)
        return hop if hop is not None else dotted
