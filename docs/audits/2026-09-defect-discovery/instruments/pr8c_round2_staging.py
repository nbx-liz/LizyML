"""PR 8c measurement after design review round 2: what staging would put into each before tree.

For every manifest row whose proposition 2 runs in a before tree (a regression row
without a `red_mutation`), and for the mutation rows as well so the table is
complete, at the before tree phase3_gap would build:

1. `new_module_references`: before-tree modules that import, by resolved module
   name, a package file only the after tree has. Non-empty means phase3_gap refuses
   to stage (round 2 finding 4). The string-based check in pr8c_round2_probe.py
   false-positived on `lizyml/config/version.py`; this is the AST check that ships.
2. Helper identity: each tests/ helper and package marker staged with the row's
   tests (`files_to_stage` minus the row's tests and the new package files), and
   `tests/conftest.py`, which is not staged: identical to the before tree's copy,
   different, or absent there.

    .venv/bin/python .../pr8c_round2_staging.py <after-ref> <scratch-dir>
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import phase3_gap as g  # noqa: E402


def main(after_ref: str, scratch: str) -> int:
    repo = pathlib.Path.cwd()
    runner = g.Runner(repo, repo / ".venv/bin/python", pathlib.Path(scratch))
    pathlib.Path(scratch).mkdir(parents=True, exist_ok=True)
    after = runner.worktree(runner.rev(after_ref))
    data = g.load_manifest()
    for key, row in sorted(data["issues"].items(), key=lambda kv: int(kv[0])):
        if row["disposition"] != "regression" or not row["github_prs"]:
            continue
        before = runner.worktree(runner.first_parent(g._merges(row, runner)[0][1]))
        new = g.new_package_files(before, after)
        refs = g.new_module_references(before, new)
        how = "mutation" if row.get("red_mutation") else "before tree"
        print(f"#{key} ({how}) before={before.name[:7]}: {len(new)} new package files, "
              f"references {refs}")
        helpers = [f for f in g.files_to_stage(before, after, row["tests"])
                   if f not in row["tests"] and f not in new] + ["tests/conftest.py"]
        differ = []
        for f in helpers:
            b = before / f
            if not b.exists():
                differ.append(f"{f} absent")
            elif b.read_bytes() != (after / f).read_bytes():
                differ.append(f"{f} differs")
        print(f"    {len(helpers)} helper files; not identical: {differ or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
