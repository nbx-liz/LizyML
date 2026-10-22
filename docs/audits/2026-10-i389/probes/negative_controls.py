"""H-0144 acceptance 6: in same-minor mode, a corrupted selected oracle fails.

A scratch tree holds the oracle module and a copy of the 3.0.1 oracle and
overlay; each control corrupts one answer, runs the matching test on pandas
3.0.6 in the latest-deps mode, and must report it failed (not skipped).

    python docs/audits/2026-10-i389/probes/negative_controls.py <scratch dir>
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SRC = Path("/tmp/claude-1000/wt-389")
PY = "/tmp/claude-1000/venvlatest/bin/python"
KEEP = [
    "tests/__init__.py",
    "tests/conftest.py",
    "tests/test_core/__init__.py",
    "tests/test_core/test_category_values.py",
    "tests/test_features/__init__.py",
    "tests/test_features/conftest.py",
    "tests/test_features/_h0138_oracle.py",
    "tests/test_features/_h0138_answers.py",
    "tests/test_features/test_h0138_faithful_oracle.py",
    "tests/test_features/data/h0138_oracle_pandas-3.0.1.json.gz",
    "tests/test_features/data/h0139_overlay_pandas-3.0.1.json.gz",
]
ORACLE = "tests/test_features/data/h0138_oracle_pandas-3.0.1.json.gz"


def load(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        out: dict[str, Any] = json.load(fh)
    return out


def corrupt(oracle: dict[str, Any], overlay: dict[str, Any], what: str) -> str:
    """Corrupt one answer in place; return the test id it must fail."""
    changed = {c["id"] for c in overlay["changes"]}
    if what == "temporal":
        case = next(
            c
            for c in oracle["temporal_cases"]
            if c["id"] not in changed and "codes_forward" in c["outcome"]
        )
        case["outcome"]["codes_forward"] = [9] + case["outcome"]["codes_forward"][1:]
        return f"test_a_temporal_column_behaves_as_frozen_or_as_h0139_decided[{case['id']}]"
    case = next(
        c
        for c in oracle["cases"]
        if not c["id"].startswith("o-") and c["encoder"]["mode"] is not None
    )
    enc = case["encoder"]
    if what == "categories":
        enc["categories"][0] = {"type": "builtins.str", "value": "corrupted"}
    elif what == "mode":
        enc["mode"] = {"type": "builtins.str", "value": "corrupted"}
    elif what == "codes":
        enc["codes_forward"] = [9] + enc["codes_forward"][1:]
    elif what == "stand_in":
        enc["stand_in"] = [] if enc["stand_in"] else ["c"]
    return f"test_a_faithful_column_answers_as_today[{case['id']}]"


def main() -> None:
    root = Path(sys.argv[1])
    shutil.rmtree(root, ignore_errors=True)
    for rel in KEEP:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SRC / rel, root / rel)
    (root / "lizyml").symlink_to(SRC / "lizyml")
    shutil.copy2(SRC / "pyproject.toml", root / "pyproject.toml")
    pristine = load(SRC / ORACLE)
    overlay = load(SRC / "tests/test_features/data/h0139_overlay_pandas-3.0.1.json.gz")
    env = {
        **os.environ,
        "PYTHONPATH": str(root),
        "GITHUB_ACTIONS": "true",
        "GITHUB_JOB": "quality-latest-deps",
        "LIZYML_H0138_ORACLE_MODE": "latest-same-minor",
    }
    failures = 0
    for what in ("categories", "mode", "codes", "stand_in", "temporal"):
        oracle = json.loads(json.dumps(pristine))
        test_id = corrupt(oracle, overlay, what)
        (root / ORACLE).write_bytes(gzip.compress(json.dumps(oracle).encode(), mtime=0))
        node = f"tests/test_features/test_h0138_faithful_oracle.py::{test_id}"
        out = subprocess.run(
            [PY, "-m", "pytest", "-p", "no:cacheprovider", "-q", "--no-header",
             "-o", "addopts=", node],  # fmt: skip
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip().splitlines()[-1]
        ok = "1 failed" in out and "skipped" not in out
        failures += not ok
        print(f"{what:10s} {test_id}: {out} -> {'FAILS (good)' if ok else 'NOT CAUGHT'}")
    sys.exit(failures)


if __name__ == "__main__":
    main()
