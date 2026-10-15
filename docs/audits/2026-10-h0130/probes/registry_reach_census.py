"""H-0130 amendment 2 measurement: who uses the two global registries.

The first draft of amendment 2 proposed a name-based registry rule (option O2)
with this grammar. Its reviews found it open: an aliased import through an
interface module's re-export (``from lizyml.metrics.registry import
MetricRegistry as M``) is counted by none of the forms. The revised amendment
adds no registry check. This probe stays as a record that the tree matches the
roles §2.1's second exception describes (implementations register, the registry
helpers look up); it is not a check and does not close that rule.

Over every ``lizyml`` module outside Layer 4 it counts these occurrences, and
nothing else:

- F1 ``import lizyml.core.registries`` (any alias), or ``from lizyml.core import
  registries`` (absolute or relative);
- F2 a name imported from ``lizyml.core.registries`` (absolute or relative);
- F3 an ``ast.Name`` whose id is a registry identifier;
- F4 an ``ast.Attribute`` whose attr is a registry identifier;
- F5 a ``str`` constant exactly equal to a registry identifier or to
  ``lizyml.core.registries`` (a docstring that merely mentions a name is not
  equal to it, and comments are not in the AST).

The registry identifiers are ``MetricRegistry``, ``CalibratorRegistry`` and
``_Registry``. Each occurrence is classified against the proposed roles:

- ``core/registries.py`` defines them: any form.
- ``metrics/registry.py`` and ``calibration/registry.py`` look up and construct:
  any form.
- any other module of ``metrics/``: F2 of ``MetricRegistry`` only, and F3 of
  ``MetricRegistry`` only as the value of an attribute ``.register``.
  ``calibration/`` likewise with ``CalibratorRegistry``.
- every other module outside Layer 4: nothing.

Layer 4 (``core/`` apart from Layer 0, and ``lizyml/__init__.py``) is not
counted. ``lizyml/_version.py`` is generated and untracked; it is left out of
the module count so the count does not depend on the checkout.

Usage (repository root): python3 -I docs/audits/2026-10-h0130/probes/registry_reach_census.py [--repo PATH]
Exit status 1 if any occurrence is not admitted.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import Counter
from pathlib import Path

IDENTS = frozenset({"MetricRegistry", "CalibratorRegistry", "_Registry"})
MODULE = "lizyml.core.registries"
LOOKUP = {"lizyml.core.registries", "lizyml.metrics.registry", "lizyml.calibration.registry"}
OWN = {"lizyml.metrics": "MetricRegistry", "lizyml.calibration": "CalibratorRegistry"}


def _base(module: str, is_package: bool, node: ast.ImportFrom) -> str:
    if not node.level:
        return node.module or ""
    package = module.split(".") if is_package else module.split(".")[:-1]
    parts = package[: len(package) - (node.level - 1)]
    return ".".join(parts + ([node.module] if node.module else []))


def occurrences(module: str, is_package: bool, tree: ast.Module) -> list[tuple[str, str, int, bool]]:
    """(form, identifier, line, admitted) for every counted occurrence."""
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    top = ".".join(module.split(".")[:2])
    own = OWN.get(top)
    lookup = module in LOOKUP
    found: list[tuple[str, str, int, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == MODULE:
                    found.append(("F1", alias.name, node.lineno, lookup))
        elif isinstance(node, ast.ImportFrom):
            base = _base(module, is_package, node)
            for alias in node.names:
                if base == "lizyml.core" and alias.name == "registries":
                    found.append(("F1", "registries", node.lineno, lookup))
                elif base == MODULE:
                    ok = lookup or (own is not None and alias.name == own and alias.asname is None)
                    found.append(("F2", alias.name, node.lineno, ok))
        elif isinstance(node, ast.Name) and node.id in IDENTS:
            parent = parents.get(node)
            register = (
                isinstance(parent, ast.Attribute)
                and parent.value is node
                and parent.attr == "register"
            )
            ok = lookup or (own is not None and node.id == own and register)
            found.append(("F3", node.id, node.lineno, ok))
        elif isinstance(node, ast.Attribute) and node.attr in IDENTS:
            found.append(("F4", node.attr, node.lineno, lookup))
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and (node.value in IDENTS or node.value == MODULE)
        ):
            found.append(("F5", node.value, node.lineno, lookup))
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[4]))
    repo = Path(parser.parse_args().repo).resolve()
    sys.path.insert(0, str(repo))
    from tests.test_architecture._layer_rules import _owner, module_name, read_table

    root = repo / "lizyml"
    table = read_table(repo / "BLUEPRINT.md")
    modules = 0
    by_role: Counter[str] = Counter()
    refused: list[str] = []
    rows: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if path == root / "_version.py":
            continue
        module = module_name(root, path)
        modules += 1
        owner = _owner(table, module)
        layer = table.paths[owner] if owner else None
        if layer == 4:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for form, ident, line, ok in occurrences(module, path.name == "__init__.py", tree):
            rel = path.relative_to(repo).as_posix()
            role = (
                "lookup"
                if module in LOOKUP
                else ("register" if ".".join(module.split(".")[:2]) in OWN else "other")
            )
            by_role[f"{role} {'admitted' if ok else 'REFUSED'}"] += 1
            rows.append(f"  L{layer} {rel}:{line} {form} {ident} {'ok' if ok else 'REFUSED'}")
            if not ok:
                refused.append(rows[-1])
    total = sum(by_role.values())
    print(f"modules (without _version.py): {modules}")
    print(f"registry-identifier occurrences outside Layer 4: {total}")
    for key, n in sorted(by_role.items()):
        print(f"  {key}: {n}")
    print("occurrences:")
    print("\n".join(rows))
    print(f"refused: {len(refused)}")
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
