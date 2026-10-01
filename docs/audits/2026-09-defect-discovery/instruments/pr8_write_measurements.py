"""Regenerate ``results/pr8_measurements.txt`` from the three PR 8 instruments."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE.parent / "results" / "pr8_measurements.txt"
ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4"}


def run(script: str, cwd: Path) -> str:
    return subprocess.run(
        [sys.executable, str(HERE / script)], cwd=cwd, env=ENV,
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip


def literal_or_default_sites(text: str) -> str:
    out: list[str] = []
    for block in re.split(r"\n(?=\S)", text.strip()):
        lines = block.splitlines()
        if lines[0].startswith("LizyMLError"):
            continue
        hits = [ln for ln in lines[1:] if re.search(r"-> (DEFAULT|True|False)$", ln)]
        if hits:
            out.append(lines[0])
            out.extend(hits)
    return "\n".join(out)


def main() -> None:
    parts = [
        "# PR 8 measurements at develop 91a698b (2026-10-01). Regenerate with "
        "instruments/pr8_write_measurements.py",
        "## pr8_knob_census.py",
        run("pr8_knob_census.py", REPO),
        "## pr8_reachability_probe.py (executed: a non-default value set at the Config "
        "path or public argument; constructor arguments spied)",
        run("pr8_reachability_probe.py", REPO),
        "(IsotonicCalibrator.params MISMATCH: the configured items arrive and the "
        "calibrator adds 'seed', so they are contained rather than equal.)",
        "## pr8_construction_sites.py: sites that pass a literal or leave the default",
        literal_or_default_sites(run("pr8_construction_sites.py", HERE)),
    ]
    OUT.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
