"""BLUEPRINT §2.1's layer rules, checked over every ``lizyml`` import (H-0129).

The layers and their interface modules are read from the §2.1 table in
BLUEPRINT.md, never copied here, so the table and the check cannot drift (DC3).
Everything this reader cannot read is a problem, not a skip (DC1): a table row
outside the grammar, a ``.py`` file ``ast`` cannot parse, a relative import
that leaves the package, an unreadable ``pyproject.toml``.

What is checked, over every ``Import`` / ``ImportFrom`` statement anywhere in a
module (function bodies and ``if TYPE_CHECKING:`` included):

- R0 every module has exactly one layer (longest table path wins), and every
  table path names at least one module;
- R1 no import of a higher layer;
- R2 a Layer 1 module imports only Layer 0 and its own category;
- R3 a Layer 2 module imports only Layer 0, its own category and Layer 1
  interfaces; a Layer 3 module may also import Layer 2 interfaces;
- R4 no cycle between categories;
- R5 the names ``import_module`` and ``__import__`` appear only as the callee
  of a call that imports a non-``lizyml`` package by an absolute constant name.

Bounds (declared, not checked): what a parent package's ``__init__`` loads at
run time, imports built with ``eval`` / ``exec``, and import text inside
strings (the ``codegen`` templates).
"""

from __future__ import annotations

import ast
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10; pytest itself depends on tomli there
    import tomli as tomllib

PACKAGE = "lizyml"
HEADING = "## 2.1 5 層カテゴリアーキテクチャ"
HEADER = (
    "Layer",
    "名称",
    "依存先",
    "含まれるモジュール",
    "下の層から参照してよいモジュール（IF）",
)
ALL_IF = "全部"
NO_IF = "なし"
DYNAMIC_NAMES = frozenset({"import_module", "__import__"})
_TOKEN = re.compile(r"`([^`]+)`")
_PATH = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:/[A-Za-z_][A-Za-z0-9_]*)*(?:/|\.py)")
_SEPARATOR = re.compile(r"\|(?: *:?-{3,}:? *\|)+")
RULES = frozenset({"R1", "R2", "R3", "R4"})


@dataclass(frozen=True)
class Table:
    """The §2.1 table: ``paths[p] = layer`` and the declared interfaces."""

    paths: Mapping[str, int]
    packages: frozenset[str]  # the paths written as ``pkg/``
    interfaces: Mapping[int, frozenset[str] | None]  # None = every module


@dataclass(frozen=True)
class Edge:
    """One resolved import: the importing file, its module, the target."""

    file: str
    source: str
    target: str


Key = tuple[str, str, str]  # (importing file, target module, rule)


def _cells(line: str) -> list[str]:
    if not (line.startswith("|") and line.endswith("|")):
        raise ValueError(f"§2.1 table: not a table row: {line!r}")
    return [cell.strip() for cell in line[1:-1].split("|")]


def _tokens(cell: str, where: str) -> list[str]:
    """The backticked paths of *cell*, which must be exactly ``a`, `b`` form."""
    tokens = _TOKEN.findall(cell)
    if not tokens or cell != ", ".join(f"`{t}`" for t in tokens):
        raise ValueError(
            f"§2.1 table: {where} must be backticked paths joined by ', ': {cell!r}"
        )
    return tokens


def module_of(path: str) -> str:
    """``core/types/`` -> ``lizyml.core.types``; ``__init__.py`` -> ``lizyml``.

    Only canonical relative paths are read: identifiers joined by single
    slashes, ending in ``/`` (a package) or ``.py`` (a module).
    """
    if not _PATH.fullmatch(path):
        raise ValueError(f"§2.1 table: {path!r} is not a package/ or a module.py path")
    if path.endswith("/"):
        parts = path[:-1].split("/")
    else:
        parts = path[:-3].split("/")
        if parts[-1] == "__init__":
            parts = parts[:-1]
    return ".".join([PACKAGE, *parts])


def read_table(blueprint: Path) -> Table:
    lines = blueprint.read_text(encoding="utf-8").splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(HEADING)]
    if len(starts) != 1:
        raise ValueError(f"BLUEPRINT.md: {len(starts)} headings start with {HEADING!r}")
    start = starts[0]
    rows = []
    for line in lines[start + 1 :]:
        if line.startswith("|"):
            rows.append(line.strip())
        elif rows:
            break
        elif line.startswith("#"):
            raise ValueError("§2.1 table: no table under the heading")
    if len(rows) < 3 or tuple(_cells(rows[0])) != HEADER:
        raise ValueError(f"§2.1 table: header must be {HEADER}")
    if not _SEPARATOR.fullmatch(rows[1]) or len(_cells(rows[1])) != len(HEADER):
        raise ValueError(f"§2.1 table: malformed separator row {rows[1]!r}")
    paths: dict[str, int] = {}
    packages: set[str] = set()
    interfaces: dict[int, frozenset[str] | None] = {}
    for row in rows[2:]:
        cells = _cells(row)
        if len(cells) != len(HEADER) or not cells[0].isdigit():
            raise ValueError(f"§2.1 table: malformed row {row!r}")
        layer = int(cells[0])
        if layer in interfaces:
            raise ValueError(f"§2.1 table: layer {layer} listed twice")
        for token in _tokens(cells[3], f"layer {layer} modules"):
            module = module_of(token)
            if module in paths:
                raise ValueError(f"§2.1 table: {token!r} listed twice")
            paths[module] = layer
            if token.endswith("/"):
                packages.add(module)
        if cells[4] == ALL_IF:
            interfaces[layer] = None
        elif cells[4] == NO_IF:
            interfaces[layer] = frozenset()
        else:
            tokens = _tokens(cells[4], f"layer {layer} interfaces")
            if any(not t.endswith(".py") for t in tokens):
                raise ValueError(
                    f"§2.1 table: layer {layer} interfaces must be modules"
                )
            interfaces[layer] = frozenset(module_of(t) for t in tokens)
    if sorted(interfaces) != [0, 1, 2, 3, 4]:
        raise ValueError(f"§2.1 table: layers {sorted(interfaces)}, expected 0-4")
    return Table(paths=paths, packages=frozenset(packages), interfaces=interfaces)


def version_module(pyproject: Path, root: Path) -> str:
    """The generated module hatch-vcs writes (``version-file``), checked every run."""
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    try:
        value = data["tool"]["hatch"]["build"]["hooks"]["vcs"]["version-file"]
    except (KeyError, TypeError):
        raise ValueError(
            "pyproject.toml: tool.hatch.build.hooks.vcs.version-file is missing"
        ) from None
    if not isinstance(value, str):
        raise ValueError(
            f"pyproject.toml: version-file must be a string, not {value!r}"
        )
    rel = Path(value)
    if rel.parts[:1] != (root.name,) or rel.suffix != ".py" or len(rel.parts) < 2:
        raise ValueError(
            f"pyproject.toml: version-file {value!r} is not a .py under {root.name}/"
        )
    return module_of("/".join(rel.parts[1:]))


def module_name(root: Path, path: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join([PACKAGE, *parts])


def _is_module(root: Path, module: str) -> bool:
    base = root.joinpath(*module.split(".")[1:])
    return base.with_suffix(".py").is_file() or (base / "__init__.py").is_file()


def resolve(root: Path, source: str, is_package: bool, node: ast.AST) -> list[str]:
    """Every ``lizyml`` module one import statement loads, once each."""
    if isinstance(node, ast.Import):
        names = [a.name for a in node.names if a.name.split(".")[0] == PACKAGE]
        return list(dict.fromkeys(names))
    assert isinstance(node, ast.ImportFrom)
    if node.level:
        package = source.split(".") if is_package else source.split(".")[:-1]
        if node.level > len(package):
            raise ValueError(f"{source}: relative import leaves the package")
        base_parts = package[: len(package) - (node.level - 1)]
        base = ".".join(base_parts + ([node.module] if node.module else []))
    else:
        base = node.module or ""
    if base.split(".")[0] != PACKAGE:
        return []
    targets: list[str] = []
    for alias in node.names:
        sub = f"{base}.{alias.name}"
        target = sub if alias.name != "*" and _is_module(root, sub) else base
        if target not in targets:
            targets.append(target)
    return targets


def _dynamic_call_allowed(call: ast.Call, name: str) -> bool:
    """An absolute import of a non-``lizyml`` package by a constant name."""
    if not call.args:
        return False
    # ``*args`` / ``**kwargs`` can carry a relative ``level`` or package unseen.
    if any(isinstance(a, ast.Starred) for a in call.args) or any(
        kw.arg is None for kw in call.keywords
    ):
        return False
    first = call.args[0]
    if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
        return False
    target = first.value
    if not target or target.startswith(".") or target.split(".")[0] == PACKAGE:
        return False
    if name == "__import__":
        levels = [kw.value for kw in call.keywords if kw.arg == "level"]
        if len(call.args) > 4:
            levels.append(call.args[4])
        return all(isinstance(v, ast.Constant) and v.value == 0 for v in levels)
    return True


def dynamic_import_problems(tree: ast.AST, file: str) -> list[str]:
    """R5: every appearance of the two names that is not an allowed call."""
    allowed: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        else:
            continue
        if name in DYNAMIC_NAMES and _dynamic_call_allowed(node, name):
            allowed.add(id(func))
    problems: list[str] = []
    for node in ast.walk(tree):
        hit = (
            (isinstance(node, ast.Name) and node.id in DYNAMIC_NAMES)
            or (isinstance(node, ast.Attribute) and node.attr in DYNAMIC_NAMES)
            or (isinstance(node, ast.Constant) and node.value in DYNAMIC_NAMES)
            or (
                isinstance(node, ast.ImportFrom)
                and any(a.name in DYNAMIC_NAMES for a in node.names)
            )
        )
        if hit and id(node) not in allowed:
            line = getattr(node, "lineno", 0)
            problems.append(f"R5 {file}:{line} dynamic import outside the rule")
    return problems


def scan(root: Path) -> tuple[list[Edge], set[str], list[str]]:
    """Every resolved import edge, every module, and the R5 problems."""
    edges: list[Edge] = []
    modules: set[str] = set()
    problems: list[str] = []
    for path in sorted(root.rglob("*.py")):
        source = module_name(root, path)
        modules.add(source)
        file = path.relative_to(root.parent).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=file)
        problems.extend(dynamic_import_problems(tree, file))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for target in resolve(root, source, path.name == "__init__.py", node):
                    edges.append(Edge(file=file, source=source, target=target))
    return edges, modules, problems


def _covers(table: Table, path: str, module: str) -> bool:
    """A ``pkg/`` row covers the package and everything in it; a ``mod.py``
    row covers that module only (``__init__.py`` is the root, not all of it)."""
    return module == path or (path in table.packages and module.startswith(path + "."))


def _owner(table: Table, module: str) -> str | None:
    """The longest table path that covers *module*."""
    best = None
    for path in table.paths:
        if _covers(table, path, module) and (best is None or len(path) > len(best)):
            best = path
    return best


def category(module: str) -> str:
    parts = module.split(".")
    if len(parts) >= 3 and parts[1] == "core":
        return ".".join(parts[:3])
    return ".".join(parts[:2])


def _cycle_categories(edges: Iterable[tuple[str, str]]) -> dict[str, int]:
    """Category -> cycle number, for categories on a cycle (a strongly
    connected component of size > 1)."""
    graph: dict[str, set[str]] = {}
    for a, b in edges:
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set())
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    found: dict[str, int] = {}
    counter = 0

    def visit(v: str) -> None:
        nonlocal counter
        index[v] = low[v] = counter
        counter += 1
        stack.append(v)
        on_stack.add(v)
        for w in sorted(graph[v]):
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            component = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                component.append(w)
                if w == v:
                    break
            if len(component) > 1:
                number = len(set(found.values()))
                found.update(dict.fromkeys(component, number))

    for v in sorted(graph):
        if v not in index:
            visit(v)
    return found


def check(
    root: Path, blueprint: Path, pyproject: Path, exceptions: Mapping[Key, int]
) -> list[str]:
    """Every problem with *root*'s imports; ``[]`` means the rules hold."""
    table = read_table(blueprint)
    generated = version_module(pyproject, root)
    edges, modules, problems = scan(root)
    modules.add(generated)

    layer_of: dict[str, int] = {}
    for module in sorted(modules):
        owner = _owner(table, module)
        if owner is None:
            problems.append(f"R0 {module} has no row in the §2.1 table")
        else:
            layer_of[module] = table.paths[owner]
    for path in sorted(table.paths):
        if not any(_covers(table, path, m) for m in modules):
            problems.append(f"R0 the §2.1 table names {path}, which no module is")
    if _owner(table, generated) is None:
        problems.append(f"R0 the generated {generated} has no row in the §2.1 table")
    for layer, names in table.interfaces.items():
        for name in sorted(names or ()):
            if name not in modules or layer_of.get(name) != layer:
                problems.append(f"R0 interface {name} is not a layer-{layer} module")

    def interface(module: str, layer: int) -> bool:
        names = table.interfaces[layer]
        return names is None or module in names

    found: Counter[Key] = Counter()
    for edge in edges:
        src, dst = layer_of.get(edge.source), layer_of.get(edge.target)
        if src is None or dst is None:
            continue  # reported by R0
        same = category(edge.source) == category(edge.target)
        if dst > src:
            found[(edge.file, edge.target, "R1")] += 1
        elif src == 1 and not same and dst != 0:
            found[(edge.file, edge.target, "R2")] += 1
        elif src in (2, 3) and not same and dst != 0:
            ok = (dst == 1 and interface(edge.target, 1)) or (
                src == 3 and dst == 2 and interface(edge.target, 2)
            )
            if not ok:
                found[(edge.file, edge.target, "R3")] += 1

    cyclic = _cycle_categories(
        (category(e.source), category(e.target))
        for e in edges
        if category(e.source) != category(e.target)
    )
    for edge in edges:
        a, b = category(edge.source), category(edge.target)
        if a != b and a in cyclic and cyclic.get(b) == cyclic[a]:
            found[(edge.file, edge.target, "R4")] += 1

    for key, count in sorted(exceptions.items()):
        if key[2] not in RULES or type(count) is not int or count < 1:
            problems.append(
                f"exception list: {key} -> {count!r} must name R1-R4 with a count >= 1"
            )
    for key in sorted(set(found) | set(exceptions)):
        actual, expected = found.get(key, 0), exceptions.get(key, 0)
        if actual != expected:
            file, target, rule = key
            problems.append(
                f"{rule} {file} -> {target}: {actual} import(s), "
                f"{expected} allowed by the exception list"
            )
    return problems
