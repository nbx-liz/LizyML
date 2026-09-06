"""Measure Phase 3 completion from what actually runs, not from a step list.

Six propositions per issue, every one executed:

  1. the named tests exist at the after-SHA;
  2. they FAIL at the before-SHA -- by copying the after-tree's test files into
     the before worktree and running them there, which is the only way to ask
     the question for a test that did not exist before. "Absent before, so it
     must have been red" is an assumption, and an assumption in a completion
     gate is DC1;
  3. they PASS at the after-SHA;
  4. the collected node count EQUALS the manifest's declared population, not
     `>=`;
  5. the population is derived from the code: `derived_from` is an executable
     expression, evaluated inside the after worktree, whose value must equal
     the declared population. A prose description proves nothing;
  6. the issue is closed by the PR that added the tests -- read from GitHub,
     not assumed. An issue whose state cannot be read is UNKNOWN.

Anything the tool cannot evaluate is UNKNOWN and counts AGAINST completion, and
`main` exits non-zero on any INCOMPLETE or UNKNOWN. The Phase 1 equivalent of
this script shipped with several steps hard-coded complete and printed 100%.

This is a run tool, not a CI test: it creates git worktrees and runs pytest in
each. Its unit tests (`tests/test_docs/test_phase3_gap.py`) do run in CI and
cover the manifest grammar, the node-id counting and the verdict arithmetic --
the parts that can be wrong quietly.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys

MANIFEST = pathlib.Path(__file__).with_name("phase3_manifest.json")
NODE_ID = re.compile(r"^(?P<file>[^:]+\.py)::(?P<rest>\S+)$")
DISPOSITIONS = {"regression", "decision-only", "partial"}


class ManifestError(Exception):
    """The manifest is malformed, or an input cannot be parsed. Never a warning."""


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def load_manifest(path: pathlib.Path = MANIFEST, *, require_github_pr: bool = True) -> dict:
    """Parse and validate the manifest.

    `require_github_pr` is on when the tool runs, because proposition 6 cannot
    be evaluated without the real numbers. It is off for planning-time grammar
    checks: before any PR opens, every `github_pr` is legitimately null, and a
    validator that refused the manifest then would make the shipped file
    permanently invalid.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if "issues" not in data or not isinstance(data["issues"], dict):
        raise ManifestError("manifest has no `issues` mapping")
    required = {"plan_pr", "github_pr", "tests", "population", "derived_from",
                "disposition"}
    for num, row in data["issues"].items():
        if not num.isdigit():
            raise ManifestError(f"issue key is not a number: {num!r}")
        missing = required - set(row)
        if missing:
            raise ManifestError(f"issue {num} is missing {sorted(missing)}")
        # `plan_pr` is this plan's ordinal (0-9). `github_pr` is the number
        # GitHub assigns when the PR is opened, filled in by that PR itself.
        # An earlier version had one field holding the ordinal and compared it
        # against `closedByPullRequestsReferences`, which GitHub numbers from 1
        # -- so PR 0's two issues could never satisfy proposition 6 however
        # perfectly they were repaired. A declaration no real input can meet is
        # DC7, and it was introduced by the fix for the previous round's DC1.
        if row["disposition"] != "partial":
            gh = row["github_pr"]
            if gh is None:
                raise ManifestError(
                    f"issue {num} has no github_pr yet; PR {row['plan_pr']} must "
                    f"write its own number into the archived manifest when it opens"
                )
            if not isinstance(gh, int) or gh < 1:
                raise ManifestError(
                    f"issue {num} has github_pr={gh!r}, which is not a GitHub PR "
                    f"number (they start at 1)"
                )
        if row["disposition"] not in DISPOSITIONS:
            raise ManifestError(
                f"issue {num} has disposition {row['disposition']!r}, "
                f"not one of {sorted(DISPOSITIONS)}"
            )
        if row["disposition"] != "regression" and not row.get("justification"):
            raise ManifestError(
                f"issue {num} is {row['disposition']} with no justification"
            )
        if not isinstance(row["population"], int) or row["population"] < 1:
            raise ManifestError(f"issue {num} has a non-positive population")
        if not row["tests"]:
            raise ManifestError(f"issue {num} names no test")
        if not str(row["derived_from"]).strip():
            raise ManifestError(f"issue {num} names no derivation for its population")
        # Most populations are a parametrisation, and `population_test` names
        # the node whose cases are it. Two are not -- a conformance suite and a
        # partial repair -- and those must say so in `population_note` rather
        # than leaving the check quietly unapplied.
        if not row.get("population_test") and not row.get("population_note"):
            raise ManifestError(
                f"issue {num} has no population_test and no population_note "
                f"explaining why its population is not a parametrisation"
            )
    return data


def parse_node_ids(stdout: str) -> list[str]:
    """Collected node ids from `pytest --collect-only -q`, with a closed grammar.

    A line containing `::` that is not a node id is an error, and a duplicate id
    is an error -- a collapsed or dynamically generated parametrisation would
    otherwise let a count come out right by accident.
    """
    ids: list[str] = []
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line or line.startswith(("=", "-", "no tests ran")):
            continue
        if line[0].isdigit() or line.startswith(("warnings summary", "ERROR")):
            continue
        if "::" not in line:
            continue
        if not NODE_ID.match(line):
            raise ManifestError(f"unparseable collected line: {line!r}")
        ids.append(line)
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ManifestError(f"duplicate collected node ids: {dupes}")
    return ids


# --------------------------------------------------------------------------
# Runners -- injectable so the unit tests can drive `evaluate` without git
# --------------------------------------------------------------------------


class Runner:
    """Everything that touches the outside world, in one replaceable object."""

    def __init__(self, python: str) -> None:
        self.python = python

    def sh(self, cmd: list[str], cwd: pathlib.Path) -> tuple[int, str]:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        return p.returncode, p.stdout + p.stderr

    def worktree(self, repo: pathlib.Path, sha: str, dest: pathlib.Path) -> pathlib.Path:
        if not dest.exists():
            rc, out = self.sh(["git", "worktree", "add", "--detach", str(dest), sha], repo)
            if rc != 0:
                raise ManifestError(f"git worktree add failed for {sha}: {out.strip()[:200]}")
        rc, out = self.sh(["git", "rev-parse", "HEAD"], dest)
        if rc != 0:
            raise ManifestError(f"cannot read HEAD of {dest}: {out.strip()[:200]}")
        got = out.strip().splitlines()[-1]
        if not got.startswith(sha) and not sha.startswith(got[:7]):
            raise ManifestError(
                f"worktree {dest} is at {got[:12]}, not the requested {sha}. "
                f"A reused scratch directory would otherwise be measured against "
                f"the wrong commit."
            )
        return dest

    def pytest(self, tree: pathlib.Path, args: list[str]) -> tuple[int, str]:
        return self.sh([self.python, "-m", "pytest", *args,
                        "--no-cov", "-p", "no:randomly"], tree)

    def evaluate_expression(self, tree: pathlib.Path, snippet: str) -> int:
        """Run `derived_from` inside the worktree; it must print one integer.

        The snippet carries its own imports and is executed against the
        after-SHA's source, so the declared population is read out of the code
        rather than described in prose.
        """
        rc, out = self.sh([self.python, "-c", snippet], tree)
        if rc != 0:
            raise ManifestError(f"derivation failed: {snippet[:80]!r} -> {out.strip()[:200]}")
        lines = [ln for ln in out.strip().splitlines() if ln.strip()]
        if not lines:
            raise ManifestError(f"derivation {snippet[:80]!r} printed nothing")
        try:
            return int(lines[-1].strip())
        except ValueError as exc:
            raise ManifestError(
                f"derivation {snippet[:80]!r} printed {lines[-1]!r}, not an int"
            ) from exc

    def issue_state(self, repo: pathlib.Path, num: str) -> dict:
        rc, out = self.sh(
            ["gh", "issue", "view", num, "--json",
             "state,closedByPullRequestsReferences"], repo)
        if rc != 0:
            raise ManifestError(f"cannot read issue #{num} state: {out.strip()[:160]}")
        return json.loads(out)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def evaluate(repo: pathlib.Path, before: str, after: str, scratch: pathlib.Path,
             runner: Runner, manifest: dict | None = None) -> dict:
    data = manifest or load_manifest()
    wt_before = runner.worktree(repo, before, scratch / "before")
    wt_after = runner.worktree(repo, after, scratch / "after")

    results: dict[str, dict] = {}
    for num, row in sorted(data["issues"].items(), key=lambda kv: int(kv[0])):
        r: dict[str, object] = {"disposition": row["disposition"]}
        tests = row["tests"]
        try:
            r["p1_exists_after"] = all((wt_after / t).exists() for t in tests)
            if not r["p1_exists_after"]:
                r["verdict"] = "INCOMPLETE"
                results[num] = r
                continue

            if row["disposition"] == "regression":
                # Copy the after-tree's tests into the before worktree and run
                # them there. A test that did not exist before cannot be judged
                # by its absence.
                staged: list[str] = []
                for t in tests:
                    dst = wt_before / t
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(wt_after / t, dst)
                    staged.append(t)
                rc, out = runner.pytest(wt_before, [*staged, "-rf", "-q"])
                for t in staged:
                    (wt_before / t).unlink(missing_ok=True)
                # A non-zero exit is not enough: a test importing an API the
                # before-SHA does not have exits non-zero on a *collection*
                # error, which would score "red for the right reason" when the
                # file never ran. Require at least one reported test failure.
                r["p2_failed_nodes"] = sum(
                    1 for ln in out.splitlines() if ln.startswith("FAILED ")
                )
                r["p2_collection_error"] = "error" in out.lower().split("short test summary")[0][-400:]
                r["p2_fails_before"] = rc != 0 and r["p2_failed_nodes"] > 0
            else:
                r["p2_fails_before"] = "n/a"

            rc, _ = runner.pytest(wt_after, [*tests, "-q"])
            r["p3_passes_after"] = rc == 0

            rc, out = runner.pytest(wt_after, [*tests, "--collect-only", "-q"])
            ids = parse_node_ids(out)
            prefix = row.get("population_test")
            if prefix:
                matching = [i for i in ids if i == prefix or i.startswith(prefix + "[")]
                r["p4_collected"] = len(matching)
                r["p4_declared"] = row["population"]
                r["p4_matches"] = len(matching) == row["population"]
            else:
                r["p4_collected"] = f"n/a ({row['population_note']})"
                r["p4_declared"] = row["population"]
                r["p4_matches"] = len(ids) > 0

            derived = runner.evaluate_expression(wt_after, row["derived_from"])
            r["p5_derived"] = derived
            r["p5_matches"] = derived == row["population"]

            state = runner.issue_state(repo, num)
            closed_by = [
                p.get("number") for p in state.get("closedByPullRequestsReferences", [])
            ]
            r["p6_state"] = state.get("state")
            r["p6_closed_by"] = closed_by
            r["p6_owner"] = row["github_pr"]
            if row["disposition"] == "partial":
                # A partial repair must stay open. Its github_pr is null.
                r["p6_ok"] = state.get("state") == "OPEN"
                r["repaired"] = f"{row.get('repaired')}/{row['population']}"
            else:
                # Closed is not enough: it must be closed by the PR the
                # manifest names. An earlier version computed `closed_by` and
                # then ignored it, so an issue closed by an unrelated PR --
                # or by hand -- satisfied proposition 6. Reading a value and
                # not comparing it is DC1 with extra steps.
                r["p6_ok"] = (
                    state.get("state") == "CLOSED" and row["github_pr"] in closed_by
                )
        except ManifestError as exc:
            r["verdict"] = "UNKNOWN"
            r["error"] = str(exc)
            results[num] = r
            continue

        checks = [r["p1_exists_after"], r["p3_passes_after"], r["p4_matches"],
                  r["p5_matches"], r["p6_ok"]]
        if row["disposition"] == "regression":
            checks.append(r["p2_fails_before"])
        if row["disposition"] == "partial":
            r["verdict"] = "PARTIAL" if all(checks) else "INCOMPLETE"
        else:
            r["verdict"] = "COMPLETE" if all(checks) else "INCOMPLETE"
        results[num] = r
    return results


def summarise(results: dict) -> tuple[int, int, int, int]:
    complete = sum(1 for r in results.values() if r["verdict"] == "COMPLETE")
    partial = sum(1 for r in results.values() if r["verdict"] == "PARTIAL")
    incomplete = sum(1 for r in results.values() if r["verdict"] == "INCOMPLETE")
    unknown = sum(1 for r in results.values() if r["verdict"] == "UNKNOWN")
    return complete, partial, incomplete, unknown


def report(results: dict) -> int:
    for num, r in results.items():
        line = f"#{num}: {r['verdict']}"
        if r.get("repaired"):
            line += f" ({r['repaired']} repaired)"
        if r.get("error"):
            line += f"  <- {r['error']}"
        print(line)
        if r["verdict"] not in {"COMPLETE", "PARTIAL"}:
            for k in sorted(r):
                if k.startswith("p"):
                    print(f"    {k} = {r[k]}")
    complete, partial, incomplete, unknown = summarise(results)
    print()
    print(f"complete {complete}/{len(results)}   partial {partial}   "
          f"incomplete {incomplete}   unknown {unknown}")
    print("INCOMPLETE and UNKNOWN both count against completion.")
    return 0 if (incomplete == 0 and unknown == 0) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/home/rem/repos/LizyML")
    ap.add_argument("--before", default=None, help="defaults to the manifest's head_before")
    ap.add_argument("--after", required=True)
    ap.add_argument("--scratch", default="/tmp/lizyml-phase3-gap")
    ap.add_argument("--python", default="/home/rem/repos/LizyML/.venv/bin/python")
    args = ap.parse_args(argv)

    data = load_manifest()
    scratch = pathlib.Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    results = evaluate(pathlib.Path(args.repo), args.before or data["head_before"],
                       args.after, scratch, Runner(args.python), data)
    return report(results)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
