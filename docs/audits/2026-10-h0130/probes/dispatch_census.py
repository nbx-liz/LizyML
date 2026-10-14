"""H-0130 amendment 1 measurement: registry construction and type dispatch.

The closed population, over every module of ``lizyml/`` (Layers 0-4 reported,
Layers 1-3 are the ones the rule would bind):

- registry construction: a call ``<Name>.get(...)`` where ``<Name>`` resolves
  to a class of ``lizyml.core.registries``;
- type dispatch: ``isinstance(x, T)`` / ``issubclass(x, T)`` (``T`` a name, an
  attribute or a tuple of them), ``type(x) is T`` / ``is not`` / ``==`` /
  ``!=`` (either side), and ``case T(...)`` class patterns.

``T`` is resolved through the module's own imports and class definitions. A
``T`` that resolves to a ``lizyml`` class is classified by the module that
defines it: Layer 0, a declared interface module, or concrete. A ``T`` that
does not resolve is printed as UNRESOLVED, never skipped.

Usage: python3 -I dispatch_census.py [--repo PATH]
"""

from __future__ import annotations

import argparse
import ast
import builtins
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[4])
    repo = parser.parse_args().repo.resolve()
    sys.path.insert(0, str(repo / "tests" / "test_architecture"))
    import _layer_rules as lr  # noqa: PLC0415

    root = repo / "lizyml"
    table = lr.read_table(repo / "BLUEPRINT.md")
    modules: dict[str, Path] = {}
    for path in sorted(root.rglob("*.py")):
        modules[lr.module_name(root, path)] = path
    modules_set = set(modules) | {lr.version_module(repo / "pyproject.toml", root)}
    layer = {m: table.paths[o] for m in modules_set if (o := lr._owner(table, m))}
    interfaces = {m for names in table.interfaces.values() if names for m in names}

    # Where each lizyml class is defined (module-level ClassDef).
    defined: dict[tuple[str, str], str] = {}
    trees: dict[str, ast.Module] = {}
    for module, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        trees[module] = tree
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                defined[(module, node.name)] = module

    def bindings(module: str, tree: ast.Module) -> dict[str, tuple[str, str | None]]:
        """Local name -> (lizyml module, attribute or None for a module)."""
        names: dict[str, tuple[str, str | None]] = {}
        package = module if modules[module].name == "__init__.py" else module.rsplit(".", 1)[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package.split(".")
                    base = base[: len(base) - node.level + 1]
                    source = ".".join(base + ([node.module] if node.module else []))
                else:
                    source = node.module or ""
                for alias in node.names:
                    names[alias.asname or alias.name] = (source, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        names[alias.asname] = (alias.name, None)
                    else:
                        top = alias.name.split(".")[0]
                        names[top] = (top, None)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                names[node.name] = (module, node.name)
        return names

    def resolve(expr: ast.expr, names: dict[str, tuple[str, str | None]]) -> str | None:
        parts: list[str] = []
        while isinstance(expr, ast.Attribute):
            parts.insert(0, expr.attr)
            expr = expr.value
        if isinstance(expr, ast.Name) and expr.id not in names and hasattr(builtins, expr.id):
            return ".".join(["builtins", expr.id, *parts])
        if not isinstance(expr, ast.Name) or expr.id not in names:
            return None
        source, attr = names[expr.id]
        dotted = ".".join([source] + ([attr] if attr else []) + parts)
        return dotted

    def classify(dotted: str | None) -> str:
        if dotted is None:
            return "UNRESOLVED"
        if not dotted.startswith("lizyml"):
            return "external"
        owner, _, name = dotted.rpartition(".")
        # Re-exports: follow one hop through the owner's bindings.
        if (owner, name) not in defined and owner in trees:
            hop = bindings(owner, trees[owner]).get(name)
            if hop and hop[1]:
                owner, name = hop[0], hop[1]
        if (owner, name) not in defined:
            return f"lizyml-non-class:{dotted}"
        if layer.get(owner) == 0:
            return f"layer0:{owner}.{name}"
        if owner in interfaces:
            return f"interface:{owner}.{name}"
        return f"concrete:{owner}.{name}"

    sites: list[tuple[int, str, str, str, str]] = []
    for module, tree in sorted(trees.items()):
        names = bindings(module, tree)
        file = modules[module].relative_to(repo).as_posix()
        lay = layer.get(module)

        constants = {
            n.targets[0].id: n.value
            for n in tree.body
            if isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and isinstance(n.value, ast.Tuple)
        }

        def flatten(target: ast.expr) -> list[ast.expr]:
            if isinstance(target, ast.Tuple):
                return [e for elt in target.elts for e in flatten(elt)]
            if isinstance(target, ast.BinOp) and isinstance(target.op, ast.BitOr):
                return flatten(target.left) + flatten(target.right)
            if isinstance(target, ast.Name) and target.id in constants:
                return flatten(constants[target.id])
            if (
                isinstance(target, ast.Call)
                and isinstance(target.func, ast.Name)
                and target.func.id == "type"
                and len(target.args) == 1
                and isinstance(target.args[0], ast.Constant)
                and target.args[0].value is None
            ):
                return []  # type(None): a builtin type
            return [target]

        def report(node: ast.AST, form: str, target: ast.expr) -> None:
            for t in flatten(target):
                kind = classify(resolve(t, names))
                if kind != "external":
                    sites.append((lay, f"{file}:{node.lineno}", form, kind, module))

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ("isinstance", "issubclass") and len(node.args) == 2:
                    report(node, node.func.id, node.args[1])
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
            ):
                owner = resolve(node.func.value, names)
                if owner and owner.startswith("lizyml.core.registries."):
                    sites.append((lay, f"{file}:{node.lineno}", "registry.get", owner, module))
            if isinstance(node, ast.Compare) and len(node.ops) == 1:
                if isinstance(node.ops[0], (ast.Is, ast.IsNot, ast.Eq, ast.NotEq)):
                    left, right = node.left, node.comparators[0]
                    for a, b in ((left, right), (right, left)):
                        if (
                            isinstance(a, ast.Call)
                            and isinstance(a.func, ast.Name)
                            and a.func.id == "type"
                        ):
                            report(node, "type-compare", b)
            if isinstance(node, ast.MatchClass):
                report(node, "case-class", node.cls)

    print(f"modules: {len(trees)}; sites (lizyml or unresolved targets): {len(sites)}")
    counts: Counter[tuple[int | None, str]] = Counter()
    for lay, where, form, kind, module in sorted(sites, key=lambda s: (str(s[0]), s[1])):
        counts[(lay, kind.split(":")[0])] += 1
        print(f"L{lay} {where} {form} {kind}")
    print("\nby layer and kind:")
    for (lay, kind), n in sorted(counts.items(), key=lambda x: (str(x[0][0]), x[0][1])):
        print(f"  L{lay} {kind}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
