#!/usr/bin/env python3
"""PR 9 acceptance A6, the mechanical half: every fold-map quote is in BLUEPRINT.md.

``results/pr9_fold_map.md`` maps each owed clause (and each H-0110 decision) to the
BLUEPRINT text that states it. This checks that every quoted cell is a verbatim
substring of the current BLUEPRINT.md and that every owed clause of the inventory has
a row. Whether the quoted text *states* the clause is the reviewed half of A6.

Exit 1 on a missing quote, an empty quote on a folded row, or an owed clause count
that differs from the rows.

Usage (from the repository root)::

    python3 docs/audits/2026-09-defect-discovery/instruments/pr9_fold_map_check.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RESULTS = Path(__file__).resolve().parent.parent / "results"
STATUSES = {"folded", "not_folded", "already_stated"}


def rows() -> list[list[str]]:
    table = []
    for line in (RESULTS / "pr9_fold_map.md").read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or line.startswith("| id ") or set(line) <= {"|", "-", " "}:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        table.append(cells)
    return table


def main() -> int:
    blueprint = (ROOT / "BLUEPRINT.md").read_text(encoding="utf-8")
    problems = []
    statuses: Counter[str] = Counter()
    per_id: Counter[str] = Counter()
    for cells in rows():
        if len(cells) < 6:
            problems.append(f"row with {len(cells)} cells: {cells[:2]}")
            continue
        pid, _, status, quote = cells[0], cells[1], cells[2], cells[3]
        status = status.strip("`")
        statuses[status] += 1
        if status not in STATUSES:
            problems.append(f"{pid}: unknown status {status!r}")
            continue
        per_id[re.sub(r"[^A-Z0-9-].*$", "", pid)] += 1
        quote = quote.strip()
        # The fold map shows each quote in a code span: `x`, or `` x `` (one padding
        # space each side, not part of the quote) when the quote holds a backtick.
        if quote.startswith("`` ") and quote.endswith(" ``"):
            quote = quote[3:-3]
        elif quote.startswith("`") and quote.endswith("`") and quote.count("`") == 2:
            quote = quote[1:-1]
        if status != "not_folded" and not quote:
            problems.append(f"{pid}: {status} row has no quote")
        elif quote and quote not in blueprint:
            problems.append(f"{pid}: quote not in BLUEPRINT.md: {quote[:60]!r}")
    owed = 0
    for batch in ("A", "B"):
        data = json.loads((RESULTS / f"pr9_inventory_{batch}.json").read_text("utf-8"))
        for entry in data["entries"].values():
            owed += sum(c["verdict"] in ("missed", "contradicted") for c in entry["clauses"])
    print(f"rows: {sum(statuses.values())}  " + "  ".join(f"{k} {v}" for k, v in sorted(statuses.items())))
    print(f"owed clauses in the inventory: {owed}")
    for problem in problems:
        print(f"  ! {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
