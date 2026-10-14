"""R6: concrete assembly and type dispatch outside the Facade (H-0130 amendment 1).

BLUEPRINT §2.1 gives concrete assembly and type dispatch to Layer 4, with two
exceptions: a provider's own factories and checks on its own adapter type, and
the two Layer 1 registries building registered types by name. The import rules
(R0-R5) cannot see either, so this module checks them over every module of
Layers 1-3 with a closed grammar (DC1):

- **Registry access.** An attribute of a ``lizyml.core.registries`` object other
  than ``register`` and ``keys`` (so ``get`` and any private name) is used only
  in ``metrics/registry.py`` and ``calibration/registry.py``; the registry class
  is instantiated only in ``core/registries.py`` (no new registry elsewhere).
- **Type dispatch**, in exactly these forms: the second argument of
  ``isinstance`` / ``issubclass``; ``type(x) is / is not / == / != T`` on
  either side; ``case T(...)``. ``T`` is a name, an attribute chain, a tuple of
  them, a ``|`` union of them, ``type(None)``, or a module-level constant tuple
  of them. Each target resolves through the module's imports and module-level
  classes (one re-export hop): a builtin or non-lizyml type, a Layer 0 class or
  a class of a declared interface module passes; any other lizyml class is
  concrete. Concrete and unresolvable targets count against an exact-count list
  keyed ``(file, target)``, so the list can neither go stale nor grow.

Bounds (declared, not checked): dispatch spelled another way (``type(x).__name__
== "..."``, ``hasattr`` duck typing, ``getattr(type(x), ...)``), a comparison of
two runtime types (``type(a) is type(b)``, which names no class), and code built
with ``eval`` / ``exec``.
"""

from __future__ import annotations

import ast
import builtins
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

PACKAGE = "lizyml"
REGISTRIES = "lizyml.core.registries"
REGISTRY_CLASS = "_Registry"
REGISTRY_PUBLIC = frozenset({"register", "keys"})
REGISTRY_MODULES = frozenset({"lizyml.metrics.registry", "lizyml.calibration.registry"})

DispatchKey = tuple[str, str]  # (file, resolved target or "unresolved:<expr>")


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
        from tests.test_architecture._layer_rules import module_name

        self.paths: dict[str, Path] = {}
        self.trees: dict[str, ast.Module] = {}
        self.names: dict[str, dict[str, str]] = {}
        self.classes: set[str] = set()
        for path in sorted(root.rglob("*.py")):
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


def _flatten(target: ast.expr, constants: Mapping[str, ast.expr]) -> list[ast.expr]:
    if isinstance(target, ast.Tuple):
        return [e for elt in target.elts for e in _flatten(elt, constants)]
    if isinstance(target, ast.BinOp) and isinstance(target.op, ast.BitOr):
        return _flatten(target.left, constants) + _flatten(target.right, constants)
    if isinstance(target, ast.Name) and target.id in constants:
        return _flatten(constants[target.id], constants)
    if (
        isinstance(target, ast.Call)
        and isinstance(target.func, ast.Name)
        and target.func.id == "type"
        and len(target.args) == 1
        and not target.keywords
        and isinstance(target.args[0], ast.Constant)
        and target.args[0].value is None
    ):
        return []
    return [target]


def _is_type_call(expr: ast.expr) -> bool:
    return (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id == "type"
        and len(expr.args) == 1
        and not expr.keywords
    )


def _targets(tree: ast.Module) -> list[ast.expr]:
    """Every dispatch target expression in *tree*, in the closed forms."""
    found: list[ast.expr] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("isinstance", "issubclass")
            and len(node.args) == 2
        ):
            found.append(node.args[1])
        elif isinstance(node, ast.Compare) and len(node.ops) == 1:
            if isinstance(node.ops[0], (ast.Is, ast.IsNot, ast.Eq, ast.NotEq)):
                left, right = node.left, node.comparators[0]
                for a, b in ((left, right), (right, left)):
                    # ``type(a) is type(b)`` compares two runtime types and
                    # names no class, except the literal ``type(None)``.
                    if _is_type_call(a) and (
                        not _is_type_call(b) or _flatten(b, {}) == []
                    ):
                        found.append(b)
        elif isinstance(node, ast.MatchClass):
            found.append(node.cls)
    return found


def dispatch_problems(
    root: Path,
    layer_of: Mapping[str, int],
    interfaces: frozenset[str],
    allowed: Mapping[DispatchKey, int],
) -> list[str]:
    """R6 problems over *root*; ``[]`` means the rule holds."""
    index = _Index(root)
    problems: list[str] = []
    found: Counter[DispatchKey] = Counter()
    for module, tree in index.trees.items():
        file = index.paths[module].relative_to(root.parent).as_posix()
        names = index.names[module]
        if module != REGISTRIES:
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    callee = _dotted(node.func, names)
                    if callee == f"{REGISTRIES}.{REGISTRY_CLASS}":
                        problems.append(
                            f"R6 {file}:{node.lineno} creates a registry outside "
                            "core/registries.py"
                        )
        if layer_of.get(module) not in (1, 2, 3):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                owner = _dotted(node.value, names)
                if (
                    owner is not None
                    and owner.startswith(f"{REGISTRIES}.")
                    and owner.count(".") == REGISTRIES.count(".") + 1
                    and node.attr not in REGISTRY_PUBLIC
                    and module not in REGISTRY_MODULES
                ):
                    problems.append(
                        f"R6 {file}:{node.lineno} uses {owner}.{node.attr} outside "
                        "metrics/registry.py and calibration/registry.py"
                    )
        constants = {
            n.targets[0].id: n.value
            for n in tree.body
            if isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and isinstance(n.value, ast.Tuple)
        }
        for target in _targets(tree):
            for leaf in _flatten(target, constants):
                dotted = _dotted(leaf, names)
                if dotted is None:
                    found[(file, f"unresolved:{ast.unparse(leaf)}")] += 1
                    continue
                if not dotted.startswith(f"{PACKAGE}.") and dotted != PACKAGE:
                    continue  # a builtin or another package's type
                cls = index.canonical(dotted)
                if cls not in index.classes:
                    found[(file, f"unresolved:{ast.unparse(leaf)}")] += 1
                    continue
                owner = cls.rpartition(".")[0]
                if layer_of.get(owner) == 0 or owner in interfaces:
                    continue
                found[(file, cls)] += 1
    for key, count in sorted(allowed.items()):
        if type(count) is not int or count < 1:
            problems.append(f"dispatch list: {key} -> {count!r} must be an int >= 1")
    for key in sorted(set(found) | set(allowed)):
        actual, expected = found.get(key, 0), allowed.get(key, 0)
        if actual != expected:
            problems.append(
                f"R6 {key[0]} -> {key[1]}: {actual} dispatch(es), "
                f"{expected} allowed by the dispatch list"
            )
    return problems
