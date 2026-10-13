"""H-0128 policy 8c: lizyml imports against BLUEPRINT §2.1's layers.

Enforced rule (exit status): a Layer 2 (training, evaluation, tuning) or
Layer 3 (explain, plots, persistence) module imports from ``lizyml.estimators``
only its interfaces (``estimators/base.py``, ``estimators/provider.py``).
Imports under ``if TYPE_CHECKING:`` count too.

Reported, not enforced: other imports that point to a higher layer, with
Layer 0 = core/exceptions, core/logging, core/registries (ARCHITECTURE.md),
core/types/*; Layer 4 = the rest of core/. ``core/param_domain`` and
``core/specs`` are in neither document's table, so those rows need a layering
decision before they can be rules.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4] / "lizyml"
LAYER1 = {"config", "data", "splitters", "features", "estimators", "metrics", "calibration"}
LAYER2 = {"training", "evaluation", "tuning"}
LAYER3 = {"explain", "plots", "persistence"}
INTERFACES = {"lizyml.estimators.base", "lizyml.estimators.provider", "lizyml.estimators"}


def layer(module: str) -> int | None:
    parts = module.split(".")
    if parts[0] != "lizyml" or len(parts) < 2:
        return None
    top = parts[1]
    if top == "core":
        if len(parts) >= 3 and parts[2] in ("exceptions", "logging", "registries", "types"):
            return 0
        return 4
    if top in LAYER1:
        return 1
    if top in LAYER2:
        return 2
    if top in LAYER3:
        return 3
    return None  # lizyml/__init__, codegen, ... (not in the table)


def module_name(path: Path) -> str:
    rel = path.relative_to(ROOT.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


violations: list[str] = []
reported: list[str] = []
checked = 0
for path in sorted(ROOT.rglob("*.py")):
    src_mod = module_name(path)
    src_layer = layer(src_mod)
    if src_layer is None:
        continue
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        targets: list[str] = []
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            targets = [node.module]
        elif isinstance(node, ast.Import):
            targets = [a.name for a in node.names]
        for target in targets:
            dst_layer = layer(target)
            if dst_layer is None:
                continue
            checked += 1
            where = f"{path.relative_to(ROOT.parent)}:{node.lineno}"
            if dst_layer > src_layer:
                reported.append(f"{where} layer {src_layer} imports layer {dst_layer}: {target}")
            if (
                src_layer in (2, 3)
                and target.startswith("lizyml.estimators")
                and target not in INTERFACES
            ):
                violations.append(f"{where} layer {src_layer} imports a concrete estimator: {target}")

print("enforced rule (layers 2-3 import only estimator interfaces):")
for line in violations:
    print("  " + line)
print("reported (an import pointing to a higher layer):")
for line in reported:
    print("  " + line)
print(f"imports checked: {checked}; enforced violations: {len(violations)}; reported: {len(reported)}")
sys.exit(1 if violations else 0)
