"""PR 8c measurement after design review round 2.

1. Reintroduction mutations (finding 2). For #264 and #288, whose tests cannot be
   red in any before tree, apply a declared mutation that puts the defect back
   into a worktree at the after SHA, run the row's tests, and restore. The
   mutation must match exactly once.
2. New-file references (finding 4). For each regression row, list the package
   files the before tree lacks and any before-tree `lizyml/` source line that
   imports one of them by its module name -- the case where staging a new file
   could change before-tree behaviour.
3. Closure comments (finding 3). For each closed row, the URL and first line of
   the last comment at or before closedAt, which the manifest will pin.

    .venv/bin/python .../pr8c_round2_probe.py <after-sha> <scratch>
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

REPO = pathlib.Path.cwd()
PY = str(pathlib.Path(".venv/bin/python").absolute())
ROWS = json.loads((REPO / "docs/audits/2026-09-defect-discovery/results/"
                   "pr8c_rows_parent.json").read_text(encoding="utf-8"))
MUTATIONS = {
    "264": ("lizyml/core/model.py",
            "provider, override=params, validate_values=True",
            "provider, override=None, validate_values=True"),
    "288": ("lizyml/core/_model_factories.py",
            "    if not overlay:\n        return dict(base)\n    canonical = ",
            "    return {**base, **overlay}\n    canonical = "),
}
Q = ("query($owner:String!,$name:String!,$n:Int!){repository(owner:$owner,name:$name)"
     "{issue(number:$n){state closedAt comments(last:50){nodes{createdAt url body}}}}}")


def tree_at(sha: str, scratch: pathlib.Path) -> pathlib.Path:
    dest = scratch / sha
    if not dest.exists():
        subprocess.run(["git", "worktree", "add", "--detach", str(dest), sha], cwd=REPO,
                       check=True, capture_output=True)
    shutil.copy2(REPO / "lizyml" / "_version.py", dest / "lizyml" / "_version.py")
    return dest


def pytest(tree: pathlib.Path, tests: list[str]) -> str:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    p = subprocess.run([PY, "-m", "pytest", *tests, "-q", "--no-cov", "-p",
                        "no:cacheprovider"], cwd=tree, capture_output=True, text=True,
                       env=env)
    return (p.stdout + p.stderr).strip().splitlines()[-1]


def mutations(after: pathlib.Path) -> None:
    print("== reintroduction mutations at the after tree")
    for issue, (rel, old, new) in MUTATIONS.items():
        path = after / rel
        original = path.read_text(encoding="utf-8")
        count = original.count(old)
        if count != 1:
            print(f"#{issue}: mutation matches {count} times, not applied")
            continue
        try:
            path.write_text(original.replace(old, new), encoding="utf-8")
            print(f"#{issue}: {rel} mutated -> {pytest(after, ROWS[issue]['tests'])}")
        finally:
            path.write_text(original, encoding="utf-8")
        print(f"    unmutated -> {pytest(after, ROWS[issue]['tests'])}")


def references(after: pathlib.Path, scratch: pathlib.Path) -> None:
    print("== before-tree imports of package files only the after tree has")
    for issue, row in sorted(ROWS.items(), key=lambda kv: int(kv[0])):
        before = tree_at(row["before"], scratch)
        new = sorted(p.relative_to(after).as_posix() for p in (after / "lizyml").rglob("*.py")
                     if "__pycache__" not in p.parts and p.name != "_version.py"
                     and not (before / p.relative_to(after)).exists())
        hits = []
        for rel in new:
            stem = pathlib.PurePosixPath(rel).stem
            mod = rel[:-3].replace("/", ".")
            pat = re.compile(rf"^\s*(from\s+\S*\b{re.escape(stem)}\b|import\s+.*\b{re.escape(stem)}\b"
                             rf"|from\s+\S+\s+import\s+.*\b{re.escape(stem)}\b)", re.M)
            for src in (before / "lizyml").rglob("*.py"):
                text = src.read_text(encoding="utf-8")
                if pat.search(text) or mod in text:
                    hits.append(f"{src.relative_to(before)} -> {rel}")
        print(f"#{issue} before={row['before']}: {len(new)} new files, "
              f"{len(hits)} before-tree references {hits[:3]}")


def closure() -> None:
    print("== closure comments to pin")
    for issue in sorted(ROWS, key=int):
        out = subprocess.run(["gh", "api", "graphql", "-f", f"query={Q}", "-F",
                              "owner=nbx-liz", "-F", "name=LizyML", "-F", f"n={issue}"],
                             capture_output=True, text=True, check=True).stdout
        data = json.loads(out)["data"]["repository"]["issue"]
        if data["state"] != "CLOSED":
            print(f"#{issue}: {data['state']}")
            continue
        before = [c for c in data["comments"]["nodes"] if c["createdAt"] <= data["closedAt"]]
        last = before[-1]
        first = last["body"].strip().splitlines()[0][:110]
        print(f"#{issue}: {last['url']}\n    {first}")


def main(after_sha: str, scratch: str) -> int:
    s = pathlib.Path(scratch)
    after = tree_at(after_sha, s)
    mutations(after)
    references(after, s)
    closure()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
