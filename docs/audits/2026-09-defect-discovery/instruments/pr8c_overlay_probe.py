"""PR 8c measurement: can a row red only by a missing API show a behavioural RED?

Design review round 1 showed that "collection fails on a missing lizyml name"
admits an unrelated import. This probe asks whether the three such rows (#264,
#288, #277) fail behaviourally when the missing name is supplied WITHOUT the fix:
it copies into the before tree only the after-tree lizyml files that do not exist
there (new modules), never a modified one, then runs the row's tests. A name added
to an existing module cannot be supplied this way, and the probe says so.

    .venv/bin/python .../pr8c_overlay_probe.py <scratch-with-worktrees>
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import sys

REPO = pathlib.Path.cwd()
PY = str(pathlib.Path(".venv/bin/python").absolute())
SUPPORT = ["tests/_train_spy.py", "tests/_ast_scan.py"]
ROWS = {
    "264/288": ("ccae32b", ["tests/test_core/test_fit_params_override.py"]),
    "277": ("5ac725e", [
        "tests/test_core/test_calibration_params_reach.py",
        "tests/test_calibration/test_calibration_param_contract.py",
        "tests/test_calibration/test_platt_mle.py",
        "tests/test_codegen/test_calibration_params_codegen.py",
    ]),
}


def new_lizyml_files(before: pathlib.Path) -> list[str]:
    out = []
    for p in sorted((REPO / "lizyml").rglob("*.py")):
        rel = p.relative_to(REPO).as_posix()
        if rel.endswith("_version.py"):
            continue
        if not (before / rel).exists():
            out.append(rel)
    return out


def main(scratch: str) -> int:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for row, (sha, tests) in ROWS.items():
        tree = pathlib.Path(scratch) / sha
        added = new_lizyml_files(tree)
        staged = [*tests, *SUPPORT, *added]
        backups = {t: (tree / t).read_bytes() for t in staged if (tree / t).exists()}
        for t in staged:
            (tree / t).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / t, tree / t)
        try:
            p = subprocess.run([PY, "-m", "pytest", *tests, "-q", "--no-cov", "-p",
                                "no:cacheprovider"], cwd=tree, capture_output=True,
                               text=True, env=env)
        finally:
            for t in staged:
                if t in backups:
                    (tree / t).write_bytes(backups[t])
                else:
                    (tree / t).unlink()
        out = p.stdout + p.stderr
        tail = out.strip().splitlines()[-1]
        missing = sorted(set(re.findall(r"^E\s+(?:ImportError|ModuleNotFoundError): (.+)$",
                                        out, re.M)))
        print(f"#{row} before={sha} new lizyml files staged: {len(added)}")
        print(f"    {tail}")
        for m in missing:
            print(f"    still missing: {m}")
        dirty = subprocess.run(["git", "status", "--short"], cwd=tree, capture_output=True,
                               text=True).stdout.strip()
        print(f"    tree clean after restore: {not dirty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
