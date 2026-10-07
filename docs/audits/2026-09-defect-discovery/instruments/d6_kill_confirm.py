"""R6 — confirm or downgrade every D6 hollow candidate by disabling producers.

Emits one node-id file per producer set, runs pytest once per set with
`kill_producers` armed, and reads the verdict off the result:

  the test STILL PASSES with its producers disabled  -> CONFIRMED HOLLOW
  the test now fails or errors                       -> DOWNGRADED (it did use one)

That is the plan's mutation check, applied per population instead of per test.
No file under /home/rem/repos/LizyML is modified: the kill patches attributes on
the imported modules for the duration of the run.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys


def _out_dir() -> pathlib.Path:
    """The D6 work directory, from LIZYML_D6_OUT. Refuses to guess one."""
    value = os.environ.get("LIZYML_D6_OUT")
    if not value:
        raise SystemExit("set LIZYML_D6_OUT to the D6 work directory")
    path = pathlib.Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path

RES = _out_dir()
WORK = _out_dir() / "r6"
WORK.mkdir(parents=True, exist_ok=True)
PY = sys.executable
ROOT = pathlib.Path(__file__).resolve().parents[4]

SET_OF = {
    ("D1a.boundary_operations",): "train",
    ("D3.public_api",): "api",
    ("D3.public_api", "D3b.metric_operations"): "metric",
    ("D4.stage_entry_points", "D5.splitter_operations"): "split",
}

rows = [json.loads(l) for l in (RES / "d6_rows.jsonl").read_text(encoding="utf-8").splitlines()]
cands = [r for r in rows if r["verdict"] == "CANDIDATE-HOLLOW"]

groups: dict[str, list[str]] = {}
for r in cands:
    mode = SET_OF.get(tuple(r["producers"]), "all")
    groups.setdefault(mode, []).append(r["test"])

print("candidates by kill mode")
for m, v in sorted(groups.items()):
    print(f"  {len(v):4d}  {m}")
print(f"  {sum(len(v) for v in groups.values()):4d}  TOTAL")
print()


def nodeid(key: str) -> str:
    """`path::Class::name` -> a pytest node id, with the class part kept."""
    parts = key.split("::")
    if len(parts) == 3 and parts[1] != "None":
        return f"{parts[0]}::{parts[1]}::{parts[2]}"
    return f"{parts[0]}::{parts[-1]}"


results: dict[str, dict] = {}
for mode, tests in sorted(groups.items()):
    ids = sorted({nodeid(t) for t in tests})
    f = WORK / f"{mode}.txt"
    f.write_text("\n".join(ids) + "\n", encoding="utf-8")
    # -rA prints a PASSED / FAILED / ERROR line per item, so d6_tally can
    # require a positive PASSED for "confirmed" instead of "not seen failing".
    cmd = [PY, "-m", "pytest", "-p", "no:cacheprovider", "--no-cov", "-q", "-rA",
           "-p", "kill_producers", "--continue-on-collection-errors",
           *ids]
    env = {"LIZYML_KILL": mode,
           "PYTHONPATH": str(pathlib.Path(__file__).resolve().parent)}
    print(f"running {len(ids)} node ids with LIZYML_KILL={mode} ...")
    proc = subprocess.run(  # noqa: S603
        cmd, cwd=str(ROOT), capture_output=True, text=True,
        env={**dict(__import__("os").environ), **env}, timeout=3600,
    )
    out = proc.stdout + proc.stderr
    (WORK / f"{mode}.out.txt").write_text(out, encoding="utf-8")
    tail = [l for l in out.splitlines() if re.match(r"^=* ?\d+ (failed|passed|error)", l)]
    results[mode] = {"node_ids": len(ids), "returncode": proc.returncode,
                     "summary": tail[-1] if tail else out.strip()[-200:]}
    print(f"  -> {results[mode]['summary']}")

(WORK / "summary.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
print()
print("A test that still passes with its producers disabled never ran them.")
