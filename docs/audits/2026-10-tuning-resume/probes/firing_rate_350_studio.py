"""H-0124 (#350) firing rate over LizyStudio's search-space catalog defaults.

LizyStudio writes its UI stepper increment (`step_map`) into the space as
`step` for every Range dimension, log or not (frontend
`SearchSpaceTable.tsx`, `step: stepMap?.[key]`). For each catalog entry with a
default numeric range and a `step_map` entry, this checks H-0124's rule: `low`
and `high` are multiples of `step` from origin 0 (decided with Decimal(str)),
so a log dimension's `low` is at least `step`. It also reports which accepted
entries would take the log rounding path.

Run (LizyStudio checkout path as the argument):
  env PYTHONPATH=<LizyStudio>/src .venv/bin/python \
    docs/audits/2026-10-tuning-resume/probes/firing_rate_350_studio.py \
    <LizyStudio>/src/lizystudio/backends/lizyml_ui_schema.py
"""

from __future__ import annotations

import importlib.util
import sys
from decimal import Decimal

path = sys.argv[1]
spec = importlib.util.spec_from_file_location("ui_schema", path)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
catalog = mod._SEARCH_SPACE_CATALOG

# step_map is a literal inside the ui-schema builder; read it from the source.
src = open(path, encoding="utf-8").read()
block = src[src.index('"step_map": {') :]
block = block[: block.index("},") + 1]
step_map: dict[str, str] = {}
for line in block.splitlines()[1:]:
    line = line.strip().rstrip(",")
    if ":" not in line or line.startswith("#"):
        continue
    k, v = line.split(":", 1)
    step_map[k.strip().strip('"')] = v.strip()


def on_grid(x: object, s: str) -> bool:
    return (Decimal(str(x)) / Decimal(s)) % 1 == 0


items = catalog.items() if isinstance(catalog, dict) else [(e.get("key"), e) for e in catalog]
n = accepted = rounding = 0
for key, entry in items:
    rng = entry.get("default_range") if isinstance(entry, dict) else None
    if not rng or "low" not in rng:
        continue
    if key not in step_map:
        print(f"{key}: no step_map entry (no step sent)")
        continue
    s = step_map[key]
    n += 1
    reasons = []
    if not on_grid(rng["low"], s):
        reasons.append(f"low {rng['low']} not a multiple of {s}")
    if not on_grid(rng["high"], s):
        reasons.append(f"high {rng['high']} not a multiple of {s}")
    if rng.get("log") and Decimal(str(rng["low"])) < Decimal(s):
        reasons.append(f"log low {rng['low']} < step {s}")
    if reasons:
        print(f"{key}: REFUSED ({'; '.join(reasons)}) range={rng}")
        continue
    accepted += 1
    if rng.get("log"):
        rounding += 1
    print(f"{key}: accepted range={rng} step={s}" + (" (log rounding)" if rng.get("log") else ""))
print(f"accepted {accepted}/{n} catalog defaults that carry a step; log rounding on {rounding}/{n}")
