"""PR 8c measurement: the issue set plan section 3 assigns to Phase 3.

Reads the section 3 table of `phase3-plan.md` and prints every `#N` in its
Fixes and Refs columns, with the PR row that names it. The completion manifest
must cover exactly this set; the shipped tool re-derives it the same way.

    .venv/bin/python .../pr8c_plan_issue_set.py
"""

from __future__ import annotations

import pathlib
import re

PLAN = pathlib.Path(__file__).resolve().parents[1] / "phase3-plan.md"


def main() -> int:
    lines = PLAN.read_text(encoding="utf-8").splitlines()
    start = lines.index("## 3. The sequence")
    header = next(i for i in range(start, len(lines)) if lines[i].startswith("| PR |"))
    cols = [c.strip() for c in lines[header].strip("|").split("|")]
    fixes, refs = cols.index("Fixes"), cols.index("Refs")
    found: dict[int, list[str]] = {}
    for ln in lines[header + 2:]:
        if not ln.startswith("|"):
            break
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) != len(cols):
            raise SystemExit(f"row has {len(cells)} cells, header has {len(cols)}: {ln!r}")
        pr = cells[0].strip("*")
        for kind, idx in (("fixes", fixes), ("refs", refs)):
            for n in re.findall(r"(?<![\d/])#(\d+)(?!\d)", cells[idx]):
                found.setdefault(int(n), []).append(f"PR {pr} {kind}")
    for n in sorted(found):
        print(f"#{n}: {', '.join(found[n])}")
    print(f"total {len(found)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
