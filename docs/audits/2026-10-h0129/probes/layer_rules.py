"""H-0129 measurement: every lizyml import against BLUEPRINT §2.1's layers.

Extends ``docs/audits/2026-10-h0128/probes/layer_imports.py`` with the two
forms H-0128's proposal review found it missed:

- relative imports (``from ..estimators.lgbm import category_json``) are
  resolved against the importing module's package;
- imported names are resolved to a module when one exists on disk
  (``from lizyml.estimators import lgbm`` targets ``lizyml.estimators.lgbm``).

Imports under ``if TYPE_CHECKING:`` and inside functions count too.

Layers (option B, decided 2026-10-14): Layer 0 = core/exceptions,
core/logging, core/registries, core/types/*, core/param_domain, core/specs/*;
Layer 1-3 = BLUEPRINT §2.1's table; Layer 4 = the rest of core/. A module in no
table (lizyml/__init__, _extras, _version, codegen/) is reported as
unclassified, never silently skipped.

Usage: python layer_rules.py [--layer0-without-b] [--root PATH]
``--layer0-without-b`` measures with the layering before option B
(param_domain and specs counted as Layer 4). ``--root`` points at another
``lizyml`` tree (used by the should-fail mutations).
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from pathlib import Path

LAYER0_CORE = {"exceptions", "logging", "registries", "types"}
LAYER0_B = {"param_domain", "specs"}
LAYER1 = {"config", "data", "splitters", "features", "estimators", "metrics", "calibration"}
LAYER2 = {"training", "evaluation", "tuning"}
LAYER3 = {"explain", "plots", "persistence"}


def build_layer(option_b: bool):
    layer0 = LAYER0_CORE | (LAYER0_B if option_b else set())

    def layer(module: str) -> int | None:
        parts = module.split(".")
        if parts[0] != "lizyml" or len(parts) < 2:
            return None
        top = parts[1]
        if top == "core":
            if len(parts) >= 3 and parts[2] in layer0:
                return 0
            return 4 if len(parts) >= 3 else None
        if top in LAYER1:
            return 1
        if top in LAYER2:
            return 2
        if top in LAYER3:
            return 3
        return None

    return layer


def category(module: str) -> str:
    parts = module.split(".")
    if len(parts) < 2:
        return module
    if parts[1] == "core" and len(parts) >= 3:
        return f"core.{parts[2]}"
    return parts[1]


def module_name(root: Path, path: Path) -> str:
    rel = path.relative_to(root.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def is_module(root: Path, module: str) -> bool:
    rel = Path(*module.split(".")[1:])
    base = root / rel
    return base.with_suffix(".py").is_file() or (base / "__init__.py").is_file()


def resolve(root: Path, src_mod: str, is_pkg: bool, node: ast.AST) -> list[str]:
    """Every lizyml module an import statement binds or loads."""
    if isinstance(node, ast.Import):
        return list(dict.fromkeys(a.name for a in node.names if a.name.split(".")[0] == "lizyml"))
    assert isinstance(node, ast.ImportFrom)
    if node.level:
        package = src_mod.split(".") if is_pkg else src_mod.split(".")[:-1]
        if node.level - 1 > len(package) - 1:
            raise SystemExit(f"relative import escapes the package: {src_mod} level {node.level}")
        base_parts = package[: len(package) - (node.level - 1)]
        base = ".".join(base_parts + ([node.module] if node.module else []))
    else:
        base = node.module or ""
    if base.split(".")[0] != "lizyml":
        return []
    targets: list[str] = []
    for alias in node.names:
        sub = f"{base}.{alias.name}"
        target = sub if alias.name != "*" and is_module(root, sub) else base
        if target not in targets:  # one entry per module per statement
            targets.append(target)
    return targets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer0-without-b", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4] / "lizyml")
    args = parser.parse_args()
    root: Path = args.root
    layer = build_layer(option_b=not args.layer0_without_b)

    upward: list[str] = []
    leaf_to_leaf: list[str] = []
    concrete_estimator: list[str] = []
    composition_to_leaf: dict[str, Counter[str]] = defaultdict(Counter)
    unclassified_src: Counter[str] = Counter()
    unclassified_dst: Counter[str] = Counter()
    cat_edges: set[tuple[str, str]] = set()
    checked = 0

    for path in sorted(root.rglob("*.py")):
        src_mod = module_name(root, path)
        src_layer = layer(src_mod)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        is_pkg = path.name == "__init__.py"
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for target in resolve(root, src_mod, is_pkg, node):
                checked += 1
                where = f"{path.relative_to(root.parent)}:{node.lineno}"
                dst_layer = layer(target)
                if src_layer is None:
                    unclassified_src[category(src_mod)] += 1
                    continue
                if dst_layer is None:
                    unclassified_dst[f"{category(target)} <- layer {src_layer}"] += 1
                    continue
                src_cat, dst_cat = category(src_mod), category(target)
                if src_cat != dst_cat:
                    cat_edges.add((src_cat, dst_cat))
                if dst_layer > src_layer:
                    upward.append(f"{where} layer {src_layer} -> layer {dst_layer}: {target}")
                if src_layer == 1 and dst_layer == 1 and src_cat != dst_cat:
                    leaf_to_leaf.append(f"{where} {src_cat} -> {target}")
                if src_layer in (2, 3) and dst_layer == 1:
                    composition_to_leaf[f"{src_cat} -> {dst_cat}"][target] += 1
                    if target.startswith("lizyml.estimators") and target not in {
                        "lizyml.estimators",
                        "lizyml.estimators.base",
                        "lizyml.estimators.provider",
                    }:
                        concrete_estimator.append(f"{where} {src_cat} -> {target}")

    # Category-level cycles (strongly connected components of size > 1).
    graph: dict[str, set[str]] = defaultdict(set)
    for a, b in cat_edges:
        graph[a].add(b)
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    sccs: list[list[str]] = []
    counter = [0]

    def strong(v: str) -> None:
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in graph[v]:
            if w not in index:
                strong(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1:
                sccs.append(sorted(comp))

    for v in sorted(set(graph) | {b for _, b in cat_edges}):
        if v not in index:
            strong(v)

    mode = "before option B" if args.layer0_without_b else "option B"
    print(f"layering: {mode}; imports resolved: {checked}")
    print(f"\n[A] layer 2/3 -> concrete estimator module: {len(concrete_estimator)}")
    for line in concrete_estimator:
        print("  " + line)
    print(f"\n[B] import pointing to a higher layer: {len(upward)}")
    for line in upward:
        print("  " + line)
    print(f"\n[C] layer 1 -> another layer-1 category: {len(leaf_to_leaf)}")
    for line in leaf_to_leaf:
        print("  " + line)
    print(f"\n[D] category cycles: {len(sccs)}")
    for comp in sccs:
        print("  " + " <-> ".join(comp))
    print("\n[E] layer 2/3 -> layer 1 targets (what 'the IF' would have to cover):")
    for edge in sorted(composition_to_leaf):
        targets = composition_to_leaf[edge]
        print(f"  {edge}: " + ", ".join(f"{t} x{n}" for t, n in sorted(targets.items())))
    print("\n[F] unclassified importers (not in any table): " + ", ".join(
        f"{k} x{n}" for k, n in sorted(unclassified_src.items())))
    print("[F] unclassified targets: " + ", ".join(
        f"{k} x{n}" for k, n in sorted(unclassified_dst.items())))
    return 1 if concrete_estimator or upward else 0


if __name__ == "__main__":
    raise SystemExit(main())
