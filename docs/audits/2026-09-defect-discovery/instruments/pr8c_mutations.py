"""PR 8c acceptance criterion 10: mutate the real manifest or tree, and see the row fall.

Each mutation changes one shipped manifest row (in memory) or one file of the after
worktree (restored afterwards), then evaluates that row alone with phase3_gap. A
mutation passes this check when the row is no longer COMPLETE or
COMPLETE-RED-BY-MUTATION. The unmutated verdicts are in results/pr8c_trial2.txt.

  (a) node row: population off by one              -> INCOMPLETE (p4)
  (b) note row: declared count disagrees with derived -> INCOMPLETE (p5)
  (c) github_prs replaced by an unrelated PR        -> INCOMPLETE (p6)
  (d) a population node skipped in the after tree   -> INCOMPLETE (p3)
  (e) one #262 xfail declaration moved to another node -> INCOMPLETE (p3)
  (f) #264's red_mutation removed                   -> INCOMPLETE (p2)
  (g) #264's fix_text is text #278 did not add      -> INCOMPLETE (p2)
  (h) closure_comment is another issue's comment    -> INCOMPLETE (p6)
  (i) `_version.py` not copied into the worktrees   -> UNKNOWN or INCOMPLETE, exit 1

    .venv/bin/python .../pr8c_mutations.py <after-ref> <scratch-dir>
"""

from __future__ import annotations

import copy
import io
import pathlib
import shutil
import sys
from collections.abc import Callable
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import phase3_gap as g  # noqa: E402

PASSING = {"COMPLETE", "COMPLETE-RED-BY-MUTATION"}


class NoVersionRunner(g.Runner):
    """A Runner whose worktrees lack the generated `lizyml/_version.py` (mutation i)."""

    def worktree(self, sha: str) -> pathlib.Path:
        dest = self.scratch / sha
        if not dest.exists():
            rc, out = self.sh(["git", "worktree", "add", "--detach", str(dest), sha], self.repo)
            if rc != 0:
                raise g.ManifestError(f"git worktree add {sha} failed: {out.strip()[:200]}")
        (dest / "lizyml" / "_version.py").unlink(missing_ok=True)
        return dest


def run_row(runner: g.Runner, after_ref: str, num: str, row: dict[str, Any]) -> tuple[str, int, str]:
    g.validate_row(num, row)
    results = g.evaluate({"issues": {num: row}}, runner, after_ref)
    buf = io.StringIO()
    code = g.report(results, out=buf)
    return results[int(num)]["verdict"], code, buf.getvalue()


def main(after_ref: str, scratch: str) -> int:
    repo = pathlib.Path.cwd()
    python = repo / ".venv/bin/python"
    runner = g.Runner(repo, python, pathlib.Path(scratch))
    pathlib.Path(scratch).mkdir(parents=True, exist_ok=True)
    rows = g.load_manifest()["issues"]
    after = runner.worktree(runner.rev(after_ref))

    def edit(num: str, change: Callable[[dict[str, Any]], None]) -> tuple[str, dict[str, Any]]:
        row = copy.deepcopy(rows[num])
        change(row)
        return num, row

    def xfail_moved(row: dict[str, Any]) -> None:
        declared = {e["test"] for e in row["expected_nonpass"]}
        other = next(i for i in runner.collect(after, row["tests"]) if i not in declared)
        row["expected_nonpass"][0]["test"] = other

    cases: list[tuple[str, str, dict[str, Any], dict[str, str] | None]] = [
        ("a", *edit("258", lambda r: r.update(population=r["population"] + 1)), None),
        ("b", *edit("259", lambda r: r.update(population=r["population"] + 1)), None),
        ("c", *edit("258", lambda r: r.update(github_prs=[305])), None),
        ("d", "258", copy.deepcopy(rows["258"]), {
            "file": "tests/test_tuning/test_direction_reconciliation.py",
            "old": "def test_inferred_direction_selects_correct_extremum(",
            "new": "@pytest.mark.skip(reason='pr8c mutation d')\n"
                   "def test_inferred_direction_selects_correct_extremum("}),
        ("e", *edit("262", xfail_moved), None),
        ("f", *edit("264", lambda r: r.pop("red_mutation")), None),
        ("g", *edit("264", lambda r: r["red_mutation"].update(
            fix_text="params, validate_values")), None),
        ("h", *edit("258", lambda r: r.update(closure_comment=rows["259"]["closure_comment"])),
         None),
    ]
    failures = 0
    for label, num, row, tree_edit in cases:
        try:
            if tree_edit:
                with g.mutated(after, tree_edit):
                    verdict, code, text = run_row(runner, after_ref, num, row)
            else:
                verdict, code, text = run_row(runner, after_ref, num, row)
        except g.ManifestError as exc:
            verdict, code, text = "REFUSED", 1, f"    ! {exc}\n"
        held = verdict not in PASSING
        failures += not held
        print(f"({label}) #{num}: {verdict}, exit {code} -> {'caught' if held else 'MISSED'}")
        print("".join(ln + "\n" for ln in text.splitlines() if ln.startswith("    !")), end="")

    bare = pathlib.Path(scratch) / "no-version"
    shutil.rmtree(bare, ignore_errors=True)
    bare.mkdir(parents=True)
    nov = NoVersionRunner(repo, python, bare)
    try:
        verdict, code, text = run_row(nov, after_ref, "258", copy.deepcopy(rows["258"]))
    finally:
        for wt in bare.iterdir():
            runner.sh(["git", "worktree", "remove", "--force", str(wt)], repo)
    held = verdict not in PASSING and code == 1
    failures += not held
    print(f"(i) #258: {verdict}, exit {code} -> {'caught' if held else 'MISSED'}")
    print("".join(ln + "\n" for ln in text.splitlines() if ln.startswith("    !")), end="")
    print(f"\n{9 - failures} of 9 mutations caught")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
