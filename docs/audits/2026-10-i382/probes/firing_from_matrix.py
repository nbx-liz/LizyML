"""H-0141 firing rate from a ``name_matrix.py`` output: among the cases that fit,
how many projects would get aliases (an admitted non-string name) and how many
names would be refused (a type H-0141 does not admit).

    python docs/audits/2026-10-i382/probes/firing_from_matrix.py <name_matrix_*.out>
"""

from __future__ import annotations

import ast
import sys

ADMITTED_NON_STR = {"int", "float", "bool", "None", "numpy.float64"}
STRINGS = {"str", "numpy.str_"}

fitted = aliased = refused = 0
refused_today: dict[str, int] = {}
for line in open(sys.argv[1], encoding="utf-8"):
    parts = line.split(None, 2)
    if len(parts) < 3 or not parts[2].startswith("{"):
        continue
    name, _kind, raw = parts
    result = ast.literal_eval(raw.strip())
    if result.get("fit") != "ok":
        continue
    fitted += 1
    if name in ADMITTED_NON_STR:
        aliased += 1
    elif name not in STRINGS:
        refused += 1
        outcome = result.get("export_code", "?")
        refused_today[outcome] = refused_today.get(outcome, 0) + 1
print(f"fitted {fitted}; aliased {aliased}/{fitted}; refused {refused}/{fitted}")
print(f"  export_code today for the refused: {refused_today}")
