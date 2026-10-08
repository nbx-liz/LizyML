"""PR 8c measurement: what proposition 2 sees when each row's tests run at the before SHA.

Plan section 8 proposition 2 copies the after-tree's test files into a worktree at
`5712f41` and requires at least one reported FAILED. This instrument stages each
row the same way, plus the `tests/` helper modules the row names, and classifies
what pytest reports: failed / passed / errors, and for errors whether they are
collection errors and what they raise. It does not decide anything; it measures
which rule the real rows can satisfy.

Inputs: a JSON file mapping issue -> {"tests": [...], "support": [...],
"before": <sha, optional>}, a default before SHA, the after tree, and a scratch
directory for worktrees. Each worktree gets `lizyml/_version.py` copied in,
since that file is generated and untracked and `import lizyml` fails without
it. Run from the repository root:

    .venv/bin/python .../pr8c_red_before.py <rows.json> <default-before> <after-tree> <scratch>
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

# absolute(), not resolve(): resolving follows the venv symlink to the base
# interpreter, which has no pytest and no lizyml dependencies.
PY = str(pathlib.Path(".venv/bin/python").absolute())
# tests/ helper modules the shipped tests import that do not exist at 5712f41,
# plus the package marker `tests/test_docs` lacked. Staged for every row, because
# without them a row errors at collection for a reason unrelated to the defect.
SUPPORT = [
    "tests/_train_spy.py",
    "tests/_ast_scan.py",
    "tests/test_data/_hostile_arrays.py",
    "tests/test_config/_knob_registry.py",
    "tests/test_persistence/_load_census.py",
    "tests/test_docs/__init__.py",
]
SUMMARY = re.compile(r"(\d+) (failed|passed|errors?|skipped|xfailed|xpassed|deselected)")


def run(tree: pathlib.Path, files: list[str]) -> tuple[int, str]:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run(
        [PY, "-m", "pytest", *files, "-q", "-rfE", "--no-cov", "-p", "no:cacheprovider",
         "-p", "no:randomly"],
        cwd=tree, capture_output=True, text=True, env=env,
    )
    return p.returncode, p.stdout + p.stderr


def classify(out: str) -> dict:
    tail = out.strip().splitlines()[-1] if out.strip() else ""
    counts = {kind.rstrip("s") if kind.startswith("error") else kind: int(n)
              for n, kind in SUMMARY.findall(tail)}
    collection = "ERROR collecting" in out or "errors during collection" in out
    raised = sorted(set(re.findall(r"^E\s+(\w+(?:Error|Exception))\b", out, re.M)))
    return {"summary": tail, "counts": counts, "collection_error": collection,
            "raised": raised}


def tree_at(repo: pathlib.Path, sha: str, scratch: pathlib.Path) -> pathlib.Path:
    """A detached worktree at `sha`, with the generated `_version.py` copied in."""
    dest = scratch / sha
    if not dest.exists():
        subprocess.run(["git", "worktree", "add", "--detach", str(dest), sha],
                       cwd=repo, check=True, capture_output=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=dest, check=True,
                          capture_output=True, text=True).stdout.strip()
    full = subprocess.run(["git", "rev-parse", sha], cwd=repo, check=True,
                          capture_output=True, text=True).stdout.strip()
    if head != full:
        raise SystemExit(f"{dest} is at {head[:7]}, not {sha}")
    shutil.copy2(repo / "lizyml" / "_version.py", dest / "lizyml" / "_version.py")
    return dest


def main(rows_path: str, default_before: str, after: str, scratch: str) -> int:
    rows = json.loads(pathlib.Path(rows_path).read_text(encoding="utf-8"))
    wa = pathlib.Path(after)
    for issue, row in sorted(rows.items(), key=lambda kv: int(kv[0])):
        sha = row.get("before", default_before)
        wb = tree_at(wa, sha, pathlib.Path(scratch))
        staged = list(dict.fromkeys([*row["tests"], *row.get("support", []), *SUPPORT]))
        existed = {t: (wb / t).exists() for t in staged}
        backups = {}
        for t in staged:
            dst = wb / t
            if dst.exists():
                backups[t] = dst.read_bytes()
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(wa / t, dst)
        try:
            rc, out = run(wb, row["tests"])
        finally:
            for t in staged:
                if t in backups:
                    (wb / t).write_bytes(backups[t])
                else:
                    (wb / t).unlink(missing_ok=True)
        c = classify(out)
        if not c["counts"]:
            raise SystemExit(f"#{issue}: pytest produced no summary line: {c['summary']!r}")
        print(f"#{issue} before={sha} rc={rc} {c['summary']}")
        print(f"    collection_error={c['collection_error']} raised={c['raised']}")
        print(f"    files existed before: {existed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:5]))
