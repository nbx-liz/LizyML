"""Summarize negative_controls_h0133.full.log: each control and the first test
that failed under it (the evidence it was caught by the guard it is for).

Usage: python3 summarize_controls.py <full.log>
"""

from __future__ import annotations

import sys

lines = open(sys.argv[1], encoding="utf-8").read().splitlines()
failed: list[str] = []
for line in lines:
    if line.startswith("FAILED ") or line.startswith("ERROR "):
        failed.append(line.split(" ")[1])
    elif line.startswith("CONTROL "):
        first = failed[0] if failed else "none"
        print(f"{line}\n    first failing test: {first}")
        failed = []
    elif line.startswith("BASELINE") or line.endswith("controls caught"):
        print(line)
