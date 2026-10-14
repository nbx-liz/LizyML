"""H-0130 measurement: every documented sentence that states a layer rule.

Searches every Git-tracked Markdown file except ``HISTORY.md``,
``CHANGELOG.md`` and ``docs/audits/`` (dated records, not specifications) for
the wordings of BLUEPRINT §2.1's rules: downward-only dependencies, Layer 1
independence, interface-only references, Facade-only assembly, and the #362
exception list. H-0130's proposal review round 2 found sentences that a list
written from memory had missed, so the population is this search, not a
recollection. The proposal gives every hit a disposition.

Usage: python3 -I doc_statements_h0130.py [--repo PATH]
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

PATTERN = re.compile(
    "|".join(
        [
            r"下方向のみ",
            r"下の Layer にのみ",
            r"互いに依存しない",
            r"Foundation のみ",
            r"Foundation の型",
            r"Layer 4（Facade）のみ",
            r"Layer 4 のみ",
            r"Facade 独占",
            r"[Dd]ownward-only",
            r"上位層→下位層",
            r"Leaf → Leaf",
            r"Leaf カテゴリ",
            r"IF のみ",
            r"abstract interfaces only",
            r"具象クラスの組み立て",
            r"具象 import",
            r"[Oo]nly the Facade",
            r"唯一",
            r"#362",
            r"例外 17",
            r"Layer 1 → Layer 0",
        ]
    )
)
EXCLUDED = ("HISTORY.md", "CHANGELOG.md")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[4])
    repo = parser.parse_args().repo.resolve()
    files = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "*.md"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    hits = 0
    for name in sorted(files):
        if name in EXCLUDED or name.startswith("docs/audits/"):
            continue
        for number, line in enumerate((repo / name).read_text(encoding="utf-8").splitlines(), 1):
            if PATTERN.search(line):
                hits += 1
                print(f"{name}:{number}: {line.strip()[:160]}")
    print(f"\nhits: {hits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
