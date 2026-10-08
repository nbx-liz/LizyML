#!/usr/bin/env python3
"""PR 9: candidate BLUEPRINT anchors per proposal, derived mechanically.

A candidate is a backticked span in the proposal's own HISTORY entry that also
occurs in BLUEPRINT.md by full token (the matcher of
``tests/test_docs/test_proposal_blueprint_coverage.py``). Proposal ids are
excluded: id presence is not content (#271).

This only proposes. Which candidates actually state the proposal's decision is
a reviewed judgement recorded in ``docs/proposal_dispositions.toml``; a
candidate that merely shares a word with BLUEPRINT is not an anchor.

Usage (from the repository root)::

    python3 docs/audits/2026-09-defect-discovery/instruments/pr9_anchor_candidates.py [--ref REF]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from tests.test_docs._history_grammar import entry_texts  # noqa: E402
from tests.test_docs.test_proposal_blueprint_coverage import has_token  # noqa: E402

_BACKTICK = re.compile(r"`([^`\n]{2,80})`")
_PROPOSAL_ID = re.compile(r"H-\d{4}")


def read(name: str, ref: str | None) -> str:
    if ref is None:
        return (ROOT / name).read_text(encoding="utf-8")
    return subprocess.run(
        ["git", "show", f"{ref}:{name}"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout


def candidates(entry: str, blueprint: str) -> list[str]:
    seen: dict[str, None] = {}
    for span in _BACKTICK.findall(entry):
        span = span.strip()
        if span and not _PROPOSAL_ID.search(span) and has_token(span, blueprint):
            seen.setdefault(span, None)
    return list(seen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", default=None)
    args = parser.parse_args()
    entries = entry_texts(read("HISTORY.md", args.ref))
    blueprint = read("BLUEPRINT.md", args.ref)
    zero = []
    for proposal_id in sorted(entries):
        found = candidates(entries[proposal_id], blueprint)
        if not found:
            zero.append(proposal_id)
        print(f"{proposal_id}  {len(found):>3}  {' | '.join(found[:12])}")
    print()
    print(f"entries: {len(entries)}  with no candidate: {len(zero)} {zero}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
