"""H-0130 options paper: registry-name occurrences and constructor calls in Layers 1-3.

Usage (from the repository root): uv run python docs/audits/2026-10-h0130/probes/construct_census.py .
"""

import ast
import sys
from collections import Counter
from pathlib import Path

repo = Path(sys.argv[1])
sys.path.insert(0, str(repo))
from tests.test_architecture._dispatch_rules import _dotted, _Index  # noqa: E402
from tests.test_architecture._layer_rules import _owner, category, read_table  # noqa: E402

root = repo / "lizyml"
table = read_table(repo / "BLUEPRINT.md")
index = _Index(root)
layer = {m: table.paths[o] for m in index.trees if (o := _owner(table, m))}
interfaces = {m for names in table.interfaces.values() if names for m in names}

print("== registry names (MetricRegistry / CalibratorRegistry / _Registry) by module")
names = {"MetricRegistry", "CalibratorRegistry", "_Registry"}
uses: Counter[tuple[str, str]] = Counter()
for module, tree in index.trees.items():
    for node in ast.walk(tree):
        ident = None
        if isinstance(node, ast.Name) and node.id in names:
            ident = f"{node.id} ({type(node.ctx).__name__})"
        elif isinstance(node, ast.Attribute) and node.attr in names:
            ident = f".{node.attr}"
        elif isinstance(node, ast.alias) and node.name in names:
            ident = f"import {node.name}"
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in names:
            ident = f"{node.value.id}.{node.attr}"
        if ident:
            uses[(f"L{layer.get(module)} {module}", ident)] += 1
for (m, ident), n in sorted(uses.items()):
    print(f"  {m}: {ident} x{n}")

print("\n== constructor calls of lizyml classes in Layers 1-3, by target kind")
kinds: Counter[str] = Counter()
examples: dict[str, list[str]] = {}
for module, tree in index.trees.items():
    if layer.get(module) not in (1, 2, 3):
        continue
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        dotted = _dotted(node.func, index.names[module])
        if dotted is None or not dotted.startswith("lizyml."):
            continue
        cls = index.canonical(dotted)
        if cls not in index.classes:
            continue
        owner = cls.rpartition(".")[0]
        if layer.get(owner) == 0:
            kind = "layer0"
        elif owner in interfaces:
            kind = "interface module"
        elif category(owner) == category(module):
            kind = "same-category concrete"
        else:
            kind = "other-category concrete"
        kinds[kind] += 1
        examples.setdefault(kind, []).append(f"{module}:{node.lineno} -> {cls}")
for kind, n in sorted(kinds.items()):
    print(f"  {kind}: {n}")
    for e in examples[kind][:6]:
        print(f"    e.g. {e}")
