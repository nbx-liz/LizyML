"""H-0130 measurement: how each of #362's seventeen exceptions would be closed.

Reads the tree with the H-0129 checker (``tests/test_architecture/_layer_rules.py``)
and BLUEPRINT §2.1's table as they are at the measured head. Reports:

1. the number of resolved ``lizyml`` imports;
2. every Layer 1 import of another Layer 1 category, grouped by category pair
   (the population an allowance table would admit), and the reverse pairs;
3. the imports the proposed allowances (``estimators`` -> ``features``,
   ``metrics``) would admit, and the imports the proposed interface additions
   (``metrics/registry.py``, ``training/results.py``) would admit;
4. the check's current problems, which must be exactly the seventeen
   exceptions, each classified by the mechanism that closes it.

Nothing is skipped: a key that no mechanism closes is printed as UNCLOSED and
the exit status is 1.

Usage: python3 -I census_h0130.py [--repo PATH]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ALLOW = {("lizyml.estimators", "lizyml.features"), ("lizyml.estimators", "lizyml.metrics")}
NEW_INTERFACES = {"lizyml.metrics.registry", "lizyml.training.results"}

# The seven settled removals, by (file, target, rule) -> closing change.
MOVED = {
    ("lizyml/__init__.py", "lizyml.core.model", "R4"): "core/model.py imports lizyml._version",
    ("lizyml/core/model.py", "lizyml", "R4"): "core/model.py imports lizyml._version",
    ("lizyml/evaluation/confusion.py", "lizyml.training.oof_assembly", "R3"): (
        "compute_oof_valid_mask moves to core/types/"
    ),
    ("lizyml/evaluation/evaluator.py", "lizyml.training.oof_assembly", "R3"): (
        "compute_oof_valid_mask moves to core/types/"
    ),
    ("lizyml/estimators/lgbm/provider.py", "lizyml.config.schema", "R2"): (
        "SMART_RATIO_RANGES / RatioRange move to core/types/"
    ),
    ("lizyml/explain/shap_explainer.py", "lizyml.features.pipelines_native", "R3"): (
        "pipeline_factory becomes required"
    ),
    ("lizyml/plots/calibration.py", "lizyml.calibration.cross_fit", "R3"): (
        "isinstance on calibration/base.py's marker (option 3)"
    ),
    ("lizyml/persistence/exporter.py", "lizyml.training.refit_trainer", "R3"): (
        "annotation by training/results.py's Protocol (option 3)"
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[4])
    repo = parser.parse_args().repo.resolve()
    sys.path.insert(0, str(repo / "tests" / "test_architecture"))
    import _layer_rules as lr  # noqa: PLC0415 - the path is set just above

    root = repo / "lizyml"
    table = lr.read_table(repo / "BLUEPRINT.md")
    edges, modules, _ = lr.scan(root)
    modules.add(lr.version_module(repo / "pyproject.toml", root))
    layer = {m: table.paths[o] for m in modules if (o := lr._owner(table, m)) is not None}
    print(f"resolved lizyml imports: {len(edges)}")

    pairs: Counter[tuple[str, str]] = Counter()
    admitted: list[str] = []
    for e in edges:
        a, b = lr.category(e.source), lr.category(e.target)
        if layer.get(e.source) == 1 and layer.get(e.target) == 1 and a != b:
            pairs[(a, b)] += 1
            if (a, b) in ALLOW:
                admitted.append(f"  {e.file} -> {e.target}")
    print("\nLayer 1 -> another Layer 1 category, by pair:")
    for (a, b), n in sorted(pairs.items()):
        print(f"  {a} -> {b}: {n}" + ("  [allowed]" if (a, b) in ALLOW else ""))
    reverse = sum(n for (a, b), n in pairs.items() if (b, a) in ALLOW)
    print(f"reverse of an allowed pair: {reverse}")
    print(f"\nadmitted by the allowances: {len(admitted)}")
    print("\n".join(admitted))

    by_interface = [
        f"  {e.file} -> {e.target}"
        for e in edges
        if e.target in NEW_INTERFACES and layer.get(e.source) in (2, 3)
        and lr.category(e.source) != lr.category(e.target)
    ]
    print(f"\nLayer 2/3 imports admitted by the new interfaces: {len(by_interface)}")
    print("\n".join(by_interface))

    problems = lr.check(root, repo / "BLUEPRINT.md", repo / "pyproject.toml", {})
    print(f"\ncheck() problems with an empty exception list: {len(problems)}")
    unclosed = 0
    for p in problems:
        rule, rest = p.split(" ", 1)
        file, target = rest.split(" -> ")[0], rest.split(" -> ")[1].split(":")[0]
        key = (file, target, rule)
        src = lr.module_name(root, repo / file)
        if key in MOVED:
            how = MOVED[key]
        elif rule == "R2" and (lr.category(src), lr.category(target)) in ALLOW:
            how = "allowance " + lr.category(src) + " -> " + lr.category(target)
        elif rule == "R3" and target in NEW_INTERFACES:
            how = "new interface " + target
        else:
            how = "UNCLOSED"
            unclosed += 1
        print(f"  {p}\n      closed by: {how}")
    print(f"\nunclosed: {unclosed}")
    return 1 if unclosed else 0


if __name__ == "__main__":
    raise SystemExit(main())
