#!/usr/bin/env python3
"""PR 9 acceptance: every owing entry has an anchor the fold introduced.

An owing entry is one the clause audit found with a ``missed`` or ``contradicted``
clause (results/pr9_inventory_{A,B}.json), plus the entries scout C flagged as
decided but unstated (``FLAGGED``). For each, at least one of its anchors in
``docs/proposal_dispositions.toml`` must be absent from BLUEPRINT.md at the base
ref and present now. Otherwise the coverage test would pass with or without the
fold, and could not show that it happened.

Exit 1 when an owing entry has no such anchor; each is named.

Usage (from the repository root)::

    .venv/bin/python docs/audits/2026-09-defect-discovery/instruments/pr9_discriminating_anchors.py \
        [--base 13fb9d7]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RESULTS = Path(__file__).resolve().parent.parent / "results"
sys.path.insert(0, str(ROOT))

from tests.test_docs.test_proposal_blueprint_coverage import (  # noqa: E402
    DISPOSITIONS,
    has_token,
    tomllib,
)

OWED = ("missed", "contradicted")
FLAGGED = ("H-0057", "H-0085", "H-0087", "H-0104", "H-0106")


def owing() -> list[str]:
    ids = set(FLAGGED)
    for batch in ("A", "B"):
        data = json.loads((RESULTS / f"pr9_inventory_{batch}.json").read_text("utf-8"))
        for pid, entry in data["entries"].items():
            if any(c["verdict"] in OWED for c in entry["clauses"]):
                ids.add(pid)
    return sorted(ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="13fb9d7")
    args = parser.parse_args()
    before = subprocess.run(["git", "show", f"{args.base}:BLUEPRINT.md"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout
    now = (ROOT / "BLUEPRINT.md").read_text(encoding="utf-8")
    with DISPOSITIONS.open("rb") as handle:
        rows = tomllib.load(handle)["proposals"]
    failures = []
    for pid in owing():
        anchors = rows.get(pid, {}).get("anchors", [])
        new = [a for a in anchors if not has_token(a, before) and has_token(a, now)]
        status = "ok " if new else "NO "
        if not new:
            failures.append(pid)
        print(f"{status} {pid}  introduced: {' | '.join(new) or '-'}")
    print()
    print(f"owing entries: {len(owing())}  with an introduced anchor: "
          f"{len(owing()) - len(failures)}  without: {len(failures)} {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
