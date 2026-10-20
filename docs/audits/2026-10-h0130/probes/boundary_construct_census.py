"""H-0130 amendment 2: which constructor calls in Layers 1-3 build an implementation of a Layer 1 boundary IF?

Usage (repository root): uv run python docs/audits/2026-10-h0130/probes/boundary_construct_census.py .
"""

import ast
import importlib
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _census_index import _dotted, _Index  # noqa: E402
from tests.test_architecture._layer_rules import _owner, read_table  # noqa: E402

BASES = {
    "lizyml.splitters.base": "BaseSplitter",
    "lizyml.features.pipeline_base": "BaseFeaturePipeline",
    "lizyml.estimators.base": "BaseEstimatorAdapter",
    "lizyml.metrics.base": "BaseMetric",
    "lizyml.calibration.base": "BaseCalibratorAdapter",
}
bases = tuple(getattr(importlib.import_module(m), n) for m, n in BASES.items())

root = repo / "lizyml"
table = read_table(repo / "BLUEPRINT.md")
index = _Index(root)
layer = {m: table.paths[o] for m in index.trees if (o := _owner(table, m))}

total = 0
hits = []
for module, tree in sorted(index.trees.items()):
    if layer.get(module) not in (1, 2, 3):
        continue
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        dotted = _dotted(node.func, index.names[module])
        if dotted is None or not dotted.startswith("lizyml."):
            continue
        cls_name = index.canonical(dotted)
        if cls_name not in index.classes:
            continue
        total += 1
        mod, _, name = cls_name.rpartition(".")
        obj = getattr(importlib.import_module(mod), name)
        if isinstance(obj, type) and issubclass(obj, bases) and obj not in bases:
            hits.append(f"L{layer[module]} {module}:{node.lineno} -> {cls_name}")
print(f"constructor calls of lizyml classes in Layers 1-3: {total}")
print(f"of which build a Layer 1 boundary-IF implementation: {len(hits)}")
for h in hits:
    print("  " + h)
