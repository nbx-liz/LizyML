"""PR 8c measurement after round 6 (option C): p2 without staging any package file.

For every regression row that runs p2 in a before tree, run its after-tree tests
there with only the tests and tests/ helpers staged (phase3_gap.files_to_stage),
and report the outcome counts and whether collection failed. For #277, whose tests
cannot run in its before tree, run the declared red_mutation at the after tree and
list the nodes it fails.

    .venv/bin/python .../pr8c_round6_probe.py <after-ref> <scratch-dir>
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import phase3_gap as g  # noqa: E402


def main(after_ref: str, scratch: str) -> int:
    repo = pathlib.Path.cwd()
    runner = g.Runner(repo, repo / ".venv/bin/python", pathlib.Path(scratch))
    after = runner.worktree(runner.rev(after_ref))
    rows = g.load_manifest()["issues"]
    print("== p2 in the before tree, no package file staged")
    for key, row in sorted(rows.items(), key=lambda kv: int(kv[0])):
        if row["disposition"] != "regression" or not row["github_prs"]:
            continue
        before = runner.worktree(runner.first_parent(g._merges(row, runner)[0][1]))
        with g.staged(before, after, g.files_to_stage(after, row["tests"])):
            _, out, cases = runner.run_tests(before, row["tests"])
        errors = "collection error" if "error" in g.outcomes(cases) else "collects"
        how = "has red_mutation" if row.get("red_mutation") else "before tree"
        print(f"#{key} ({how}) before={before.name[:7]}: {dict(g.outcomes(cases))}, {errors}")
    print("== #277 red_mutation at the after tree")
    row = rows["277"]
    _, _, plain = runner.run_tests(after, row["tests"])
    with g.mutated(after, row["red_mutation"]):
        _, _, cases = runner.run_tests(after, row["tests"])
    failed = sorted(c.node for c in cases if c.outcome == "failed")
    print(f"unmutated {dict(g.outcomes(plain))}; mutated {dict(g.outcomes(cases))}")
    for node in failed:
        print(f"    failed: {node}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
