#!/usr/bin/env python3
"""PR 9: per owing entry, the tokens a BLUEPRINT fold could introduce as a discriminating anchor.

An anchor already present in BLUEPRINT before the fold cannot show that the fold
happened: the check would pass with or without it. A discriminating anchor is a
token that (a) occurs by full token in the proposal's HISTORY entry, (b) is
absent from BLUEPRINT at the base ref, and (c) the fold writes. (a) and (b) are
measured here from the backticked spans within ``WINDOW`` lines of each owed
clause's ``history_line``; (c) is the edit's obligation, checked afterwards by
the coverage test.

Usage (from the repository root)::

    .venv/bin/python docs/audits/2026-09-defect-discovery/instruments/pr9_fold_tokens.py \
        [--ref 13fb9d7]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RESULTS = Path(__file__).resolve().parent.parent / "results"
sys.path.insert(0, str(ROOT))

from tests.test_docs._history_grammar import entry_texts  # noqa: E402
from tests.test_docs.test_proposal_blueprint_coverage import has_token  # noqa: E402

OWED = ("missed", "contradicted")
WINDOW = 4
_BACKTICK = re.compile(r"`([^`\n]{2,80})`")
_PROPOSAL_ID = re.compile(r"H-\d{4}")


def show(ref: str, name: str) -> str:
    return subprocess.run(["git", "show", f"{ref}:{name}"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", default="13fb9d7")
    args = parser.parse_args()
    history = show(args.ref, "HISTORY.md")
    blueprint = show(args.ref, "BLUEPRINT.md")
    lines = history.splitlines()
    entries = entry_texts(history)
    none_found = []
    for batch in ("A", "B"):
        data = json.loads((RESULTS / f"pr9_inventory_{batch}.json").read_text("utf-8"))
        for pid in sorted(data["entries"]):
            owed = [c for c in data["entries"][pid]["clauses"] if c["verdict"] in OWED]
            if not owed:
                continue
            found: dict[str, None] = {}
            proposed = {c.get("proposed_anchor") for c in owed}
            for token in proposed:
                if (token and has_token(token, entries[pid])
                        and not has_token(token, blueprint)):
                    found.setdefault(token, None)
            for clause in owed:
                n = clause.get("history_line") or 0
                for line in lines[max(n - 1 - WINDOW, 0): n + WINDOW]:
                    for span in _BACKTICK.findall(line):
                        span = span.strip()
                        if (span and not _PROPOSAL_ID.search(span)
                                and has_token(span, entries[pid])
                                and not has_token(span, blueprint)):
                            found.setdefault(span, None)
            if not found:
                none_found.append(pid)
            print(f"{pid}  {len(owed):>2} owed  {' | '.join(list(found)[:10])}")
    print()
    print(f"owing entries with no discriminating candidate: {len(none_found)} {none_found}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
