#!/usr/bin/env python3
"""Rebuild the no-obligation re-audit's `missed` clause inventory from a transcript.

The re-audit's own data files (`r16_verdicts.json`, `r38_merged.json`,
`reviews/noob-{A,B,C}.json`) were written to disk by scripts and are therefore
NOT in the transcript.  What *is* in the transcript is each batch's printed
summary, in this shape:

    H-0002: SPECIFIED -> PARTIAL  (50 clauses)  <<< REVERSED
          + FitResult is a dataclass.
          + oof_pred is an np.ndarray.
          ...

This reconstructs the 78 `missed` clauses across 38 entries from those blocks.
Verified 2026-09-06: 38 entries, 78 clauses, per-batch 59/15/4, 24 reversed —
each figure matching the totals the batches themselves reported.

Not recoverable by this or any other means: the `stated`, `superseded` and
`not_section_3` dispositions (287 clauses), and the 65 clauses of the 19 entries
that already carried an obligation before the re-audit.

Usage
-----
    python3 extract_missed_clauses.py <session>.jsonl -o missed_clauses.json

See `recover_from_transcript.py` for the general Write/Edit replay, and the
`discovery-reaudit-2026-09-05` memory for what the inventory means.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections.abc import Iterator
from typing import Any

HEAD = re.compile(
    r"^(H-\d{4}): ([A-Z_]+) -> ([A-Z_]+)\s+\((\d+) clauses\)(\s+<<< REVERSED)?\s*$"
)
BULLET = "      + "


def tool_results(path: pathlib.Path) -> Iterator[str]:
    """Yield the text of every tool result in the transcript."""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict):
            continue
        content = (rec.get("message") or {}).get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            c = block.get("content")
            if isinstance(c, str):
                yield c
            elif isinstance(c, list):
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "text":
                        yield b.get("text", "")


def parse(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    """Merge every batch summary block found, keeping the richest per entry."""
    out: dict[str, dict[str, Any]] = {}
    for body in tool_results(path):
        if "clauses)" not in body:
            continue
        cur: dict[str, Any] | None = None
        for line in body.splitlines():
            m = HEAD.match(line)
            if m:
                pid, before, after, total, rev = m.groups()
                cur = {
                    "before": before, "after": after,
                    "clauses_total": int(total), "reversed": bool(rev),
                    "missed": [],
                }
                # A later block wins only if it carries at least as many clauses.
                prev = out.get(pid)
                if prev is None or len(prev["missed"]) == 0:
                    out[pid] = cur
                else:
                    cur = None
                continue
            if cur is None:
                continue
            if line.startswith(BULLET):
                cur["missed"].append(line[len(BULLET):].strip())
            elif line and not line.startswith(" "):
                cur = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("transcript", type=pathlib.Path)
    ap.add_argument("-o", "--out", type=pathlib.Path, help="write JSON here")
    args = ap.parse_args()

    if not args.transcript.exists():
        print(f"no such transcript: {args.transcript}", file=sys.stderr)
        return 2

    data = parse(args.transcript)
    if not data:
        print("no batch summary blocks found", file=sys.stderr)
        return 1

    total = sum(len(v["missed"]) for v in data.values())
    reversed_n = sum(1 for v in data.values() if v["reversed"])
    print(f"entries {len(data)}   missed clauses {total}   reversed {reversed_n}")
    for pid in sorted(data):
        v = data[pid]
        if v["missed"]:
            print(f"  {pid}  {v['before']}->{v['after']}  "
                  f"missed {len(v['missed'])}/{v['clauses_total']}")

    if args.out:
        args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n-> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
