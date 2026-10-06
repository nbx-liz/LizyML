#!/usr/bin/env python3
"""PR 9 acceptance A6, the mechanical half: the fold map is complete and its quotes exist.

``results/pr9_fold_map.md`` maps each owed clause (and each H-0110 decision) to the
BLUEPRINT text that states it. This checks, closed in both directions:

- the map's row ids equal the expected ids exactly: every ``missed`` /
  ``contradicted`` clause of ``results/pr9_inventory_{A,B}.json`` as
  ``<proposal>#<index>`` (``H-0078#4`` is mapped as its two halves ``#4a`` / ``#4b``),
  plus the declared decision rows ``DECISION_ROWS``. A missing, extra or duplicated
  id fails (acceptance review round 1 deleted ``H-0083#0`` and this stayed green);
- every row has a known status, and every row but a ``not_folded`` one quotes text
  that is a verbatim substring of the current BLUEPRINT.md.

Whether the quoted text *states* the clause is the reviewed half of A6.

Usage (from the repository root)::

    python3 docs/audits/2026-09-defect-discovery/instruments/pr9_fold_map_check.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RESULTS = Path(__file__).resolve().parent.parent / "results"
STATUSES = frozenset({"folded", "not_folded", "already_stated"})
OWED = ("missed", "contradicted")
# One owed clause the map splits in two, because half of it is not in force (#318).
SPLITS = {"H-0078#4": ("H-0078#4a", "H-0078#4b")}
# H-0110 decision 6 (C1-C5, scout C's flags) and decisions 4 / 8 (D1-D3); the
# lettered rows are the second and third halves of one decision.
DECISION_ROWS = frozenset(
    {
        "C1 H-0057", "C2 H-0085", "C3 H-0087", "C3b H-0087", "C4 H-0104",
        "C4b H-0104", "C5 H-0106", "C5b H-0106", "D1 H-0110", "D2 H-0110",
        "D2b H-0110", "D2c H-0110", "D3 H-0110",
    }
)


def parse_rows(text: str) -> list[list[str]]:
    """The cells of every table row of the map (header and rule lines excluded)."""
    table = []
    for line in text.splitlines():
        if not line.startswith("| ") or line.startswith("| id ") or set(line) <= {"|", "-", " "}:
            continue
        table.append([c.strip() for c in line.strip().strip("|").split("|")])
    return table


def expected_ids(inventories: list[dict]) -> set[str]:
    ids: set[str] = set()
    for data in inventories:
        for pid, entry in data["entries"].items():
            for index, clause in enumerate(entry["clauses"]):
                if clause["verdict"] in OWED:
                    key = f"{pid}#{index}"
                    ids.update(SPLITS.get(key, (key,)))
    return ids | DECISION_ROWS


def unquote(cell: str) -> str:
    """The quote inside a cell's code span (`x`, or `` x `` when it holds a backtick)."""
    cell = cell.strip()
    if cell.startswith("`` ") and cell.endswith(" ``"):
        return cell[3:-3]
    if cell.startswith("`") and cell.endswith("`") and cell.count("`") == 2:
        return cell[1:-1]
    return cell


def problems(rows: list[list[str]], expected: set[str], blueprint: str) -> list[str]:
    found: list[str] = []
    seen: Counter[str] = Counter()
    for cells in rows:
        if len(cells) < 6:
            found.append(f"row with {len(cells)} cells: {cells[:2]}")
            continue
        rid, status, quote = cells[0], cells[2].strip("`"), unquote(cells[3])
        seen[rid] += 1
        if status not in STATUSES:
            found.append(f"{rid}: unknown status {status!r}")
        elif status != "not_folded" and (not quote or quote == "-"):
            found.append(f"{rid}: {status} row has no quote")
        elif status != "not_folded" and quote not in blueprint:
            found.append(f"{rid}: quote not in BLUEPRINT.md: {quote[:60]!r}")
    found += [f"{rid}: mapped {n} times" for rid, n in sorted(seen.items()) if n > 1]
    found += [f"{rid}: owed but not mapped" for rid in sorted(expected - set(seen))]
    found += [f"{rid}: mapped but not owed" for rid in sorted(set(seen) - expected)]
    return found


def main() -> int:
    inventories = [
        json.loads((RESULTS / f"pr9_inventory_{b}.json").read_text(encoding="utf-8"))
        for b in ("A", "B")
    ]
    rows = parse_rows((RESULTS / "pr9_fold_map.md").read_text(encoding="utf-8"))
    expected = expected_ids(inventories)
    found = problems(rows, expected, (ROOT / "BLUEPRINT.md").read_text(encoding="utf-8"))
    statuses = Counter(cells[2].strip("`") for cells in rows if len(cells) >= 6)
    print(f"rows: {len(rows)}  expected ids: {len(expected)}  "
          + "  ".join(f"{k} {v}" for k, v in sorted(statuses.items())))
    for problem in found:
        print(f"  ! {problem}")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
