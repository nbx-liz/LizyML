"""H-0131 census: the rows a declarative registry schema would hold today.

Reads the two shipped registries (``MetricRegistry``, ``CalibratorRegistry``)
after their helper modules have imported every implementation, and checks each
row against the bootstrap validation H-0131 proposes:

1. the key is registered once (counted from the ``register("key")`` decorator
   calls in the source, since the runtime dict keeps only the last write);
2. the class is defined in a module of its own category package;
3. the class is reachable as ``module.qualname`` (an owned definition, not an
   import alias);
4. the class subclasses the category's boundary interface;
5. for metrics, the task table ``_TASK_METRICS`` names exactly the registered
   keys.

It also lists every reference to the two registry symbols inside ``lizyml/``
and ``tests/``, which the implementation must migrate.

Run from the repository root: ``.venv/bin/python docs/audits/2026-11-i366/registry_census.py``.
"""

from __future__ import annotations

import ast
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import lizyml.calibration.registry  # noqa: E402,F401  (registers calibrators)
import lizyml.metrics.registry as metric_helpers  # noqa: E402
from lizyml.calibration.base import BaseCalibratorAdapter  # noqa: E402
from lizyml.core.registries import CalibratorRegistry, MetricRegistry  # noqa: E402
from lizyml.metrics.base import BaseMetric  # noqa: E402

SYMBOLS = ("MetricRegistry", "CalibratorRegistry", "_Registry")


def decorator_keys(package: Path, symbol: str) -> Counter[str]:
    """Count ``@<symbol>.register("<key>")`` decorators under *package*."""
    counts: Counter[str] = Counter()
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for dec in node.decorator_list:
                if (
                    isinstance(dec, ast.Call)
                    and isinstance(dec.func, ast.Attribute)
                    and dec.func.attr == "register"
                    and isinstance(dec.func.value, ast.Name)
                    and dec.func.value.id == symbol
                    and dec.args
                    and isinstance(dec.args[0], ast.Constant)
                ):
                    counts[str(dec.args[0].value)] += 1
    return counts


def check(name: str, registry: object, package: str, base: type) -> int:
    store: dict[str, type] = registry._store  # type: ignore[attr-defined]  # noqa: SLF001
    counts = decorator_keys(ROOT / "lizyml" / package, name)
    bad = 0
    print(f"## {name}: {len(store)} rows")
    for key in sorted(store):
        cls = store[key]
        module = sys.modules[cls.__module__]
        problems = []
        if counts[key] != 1:
            problems.append(f"registered {counts[key]} times in source")
        if not cls.__module__.startswith(f"lizyml.{package}."):
            problems.append(f"defined outside lizyml.{package}")
        if getattr(module, cls.__qualname__, None) is not cls:
            problems.append("not reachable as module.qualname")
        if not issubclass(cls, base):
            problems.append(f"not a {base.__name__}")
        bad += bool(problems)
        status = "; ".join(problems) if problems else "ok"
        print(f"  {key:16s} {cls.__module__}.{cls.__qualname__}: {status}")
    extra = sorted(set(counts) - set(store))
    if extra:
        bad += len(extra)
        print(f"  decorator keys missing at run time: {extra}")
    return bad


def references() -> None:
    print("## references to the registry symbols")
    for top in ("lizyml", "tests"):
        for path in sorted((ROOT / top).rglob("*.py")):
            rel = path.relative_to(ROOT)
            tree = ast.parse(path.read_text(), filename=str(path))
            hits = sorted(
                {
                    node.lineno
                    for node in ast.walk(tree)
                    if (isinstance(node, ast.Name) and node.id in SYMBOLS)
                    or (isinstance(node, ast.alias) and node.name in SYMBOLS)
                    or (isinstance(node, ast.Attribute) and node.attr in SYMBOLS)
                }
            )
            if hits:
                print(f"  {rel}: lines {hits}")


def main() -> int:
    bad = check("MetricRegistry", MetricRegistry, "metrics", BaseMetric)
    bad += check("CalibratorRegistry", CalibratorRegistry, "calibration", BaseCalibratorAdapter)
    task_keys = set().union(*metric_helpers._TASK_METRICS.values())  # noqa: SLF001
    metric_keys = set(MetricRegistry._store)  # noqa: SLF001
    print(f"## _TASK_METRICS names {len(task_keys)} keys")
    if task_keys != metric_keys:
        bad += 1
        print(f"  only in _TASK_METRICS: {sorted(task_keys - metric_keys)}")
        print(f"  only registered: {sorted(metric_keys - task_keys)}")
    else:
        print("  equal to the registered metric keys")
    references()
    print(f"## rows the proposed bootstrap validation would refuse: {bad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
