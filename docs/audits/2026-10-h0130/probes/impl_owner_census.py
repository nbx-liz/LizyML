"""H-0130 options paper: who imports the modules that implement Layer 1 interfaces.

An "implementation module" defines a module-level class whose base resolves to
one of the Layer 1 interface classes (BaseSplitter, BaseFeaturePipeline,
BaseEstimatorAdapter, BaseMetric, BaseCalibratorAdapter), directly or through
another class of lizyml. For each such module, every importer (any layer) is
listed with its layer, so an "owner" table (option 4) can be judged against
the real import graph.

Usage (from the repository root): uv run python docs/audits/2026-10-h0130/probes/impl_owner_census.py
"""

from __future__ import annotations

import ast
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from tests.test_architecture._dispatch_rules import _dotted, _Index  # noqa: E402
from tests.test_architecture._layer_rules import _owner, read_table, scan  # noqa: E402

INTERFACES = {
    "lizyml.splitters.base.BaseSplitter",
    "lizyml.features.pipeline_base.BaseFeaturePipeline",
    "lizyml.estimators.base.BaseEstimatorAdapter",
    "lizyml.metrics.base.BaseMetric",
    "lizyml.calibration.base.BaseCalibratorAdapter",
}


def main() -> int:
    root = REPO / "lizyml"
    table = read_table(REPO / "BLUEPRINT.md")
    index = _Index(root)
    layer = {m: table.paths[o] for m in index.trees if (o := _owner(table, m))}

    bases: dict[str, list[str]] = {}
    for module, tree in index.trees.items():
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                resolved = []
                for base in node.bases:
                    dotted = _dotted(base, index.names[module])
                    if dotted is not None:
                        resolved.append(index.canonical(dotted))
                bases[f"{module}.{node.name}"] = resolved

    def implements(cls: str, seen: frozenset[str] = frozenset()) -> str | None:
        for base in bases.get(cls, []):
            if base in INTERFACES:
                return base
            if base not in seen:
                found = implements(base, seen | {cls})
                if found:
                    return found
        return None

    impl_modules: dict[str, set[str]] = defaultdict(set)
    for cls in bases:
        interface = implements(cls)
        if interface and cls.rpartition(".")[0] != interface.rpartition(".")[0]:
            impl_modules[cls.rpartition(".")[0]].add(interface.rsplit(".", 1)[1])

    edges, _, _ = scan(root)
    importers: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        if edge.target in impl_modules and edge.source != edge.target:
            importers[edge.target].add(f"L{layer.get(edge.source)} {edge.source}")

    print(f"implementation modules: {len(impl_modules)}")
    for module in sorted(impl_modules):
        kinds = ", ".join(sorted(impl_modules[module]))
        print(f"\n{module} (L{layer.get(module)}; implements {kinds})")
        for importer in sorted(importers.get(module, ())):
            print(f"  <- {importer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
