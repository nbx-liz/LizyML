"""Measure Phase 3 completion from what actually runs, not from a step list.

Plan section 8, as revised by Revision 7 (PR 8c). One row per issue the plan's
section 3 table assigns to Phase 3; the manifest must cover exactly that set.
Every proposition is executed:

  p1  the row's tests exist at the after SHA;
  p2  red before the fix, shown one of two ways, and only by a FAILED node:
      - before tree: the first parent of the earliest fixing PR's merge commit,
        with the row's after-tree tests and the tests/ helper modules staged into
        it. No package file is ever staged, so the system under test is exactly
        the before commit;
      - reintroduction mutation (`red_mutation`): for a row whose tests cannot run
        in any before tree, a declared edit that puts the defect back into the
        after tree -- or, for an issue that reported a coverage gap rather than a
        defect (#288), the faulty behaviour the missing tests would not have
        caught. Its `fix_text` must be text the fixing PR added and text the
        edit removes. Such a row is COMPLETE-RED-BY-MUTATION, never COMPLETE;
  p3  at the after SHA JUnit reports exactly the collected node ids (compared as a
      multiset, not a count), and every one passed, except the non-passing nodes
      the row declares in `expected_nonpass`, each bound to a test, an outcome, a
      reason and an exact count;
  p4  `population_test` collects exactly the declared (or derived) population;
      a `population_note` row needs its tests to collect, and its count is
      reported as declared, not measured;
  p5  `derived_from`, when present, is executed with `python -c` inside the after
      worktree and must print the same number;
  p6  the issue is closed (COMPLETED, or NOT_PLANNED for a not-planned row); every
      fixing PR is MERGED, its merge commit is an ancestor of the after SHA, its
      body cites the issue; and the pinned `closure_comment` exists on the issue,
      was written no earlier than every fixing merge and no later than the close,
      and names every fixing PR by `#N` or by its exact title. That the pinned
      comment affirms the fix is a reviewed declaration of the manifest, not
      something this tool can read. A partial row's issue must be OPEN.

Why p6 does not read `closedByPullRequestsReferences`: merges go to `develop`,
which is not the default branch, so GitHub never closes an issue from a PR here.
Every issue in this run was closed by hand and that field is empty for all of
them (results/pr8c_measurements.txt item 2) -- reading it made the archived
proposition unsatisfiable (DC7).

Anything the tool cannot evaluate is UNKNOWN. INCOMPLETE and UNKNOWN both count
against completion: `main` exits 1 when either occurs. This is a run tool, not a
CI test -- it creates git worktrees and runs pytest in each. Its unit tests are
`tests/test_docs/test_phase3_gap.py`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterator
from typing import Any, NamedTuple

HERE = pathlib.Path(__file__).resolve().parent
MANIFEST = HERE / "phase3_manifest.json"
PLAN = HERE.parent / "phase3-plan.md"
OWNER, NAME = "nbx-liz", "LizyML"

DISPOSITIONS = {"regression", "decision-only", "partial", "not-planned"}
NONPASS = {"skipped", "xfailed"}
VERDICTS = ("COMPLETE", "COMPLETE-RED-BY-MUTATION", "PARTIAL", "NOT-PLANNED",
            "INCOMPLETE", "UNKNOWN")
NODE_ID = re.compile(r"^(?P<file>[^:\s]+\.py)::(?P<rest>\S.*)$")


class ManifestError(Exception):
    """The manifest or an input cannot be read as declared. Never a warning."""


class Case(NamedTuple):
    """One JUnit test case: its node id, outcome and message."""

    node: str
    outcome: str
    message: str


# --------------------------------------------------------------------------
# Text rules
# --------------------------------------------------------------------------


def cites(text: str, number: int) -> bool:
    """`#number`, bounded: #26 never matches #263, and `owner/repo#263` is not #263 (DC2)."""
    return re.search(rf"(?<![\w/])#{number}(?!\d)", text) is not None


def names_pr(comment: str, number: int, title: str) -> bool:
    """The comment names the PR by `#number` or by its exact title.

    A trailing ` (#N)` -- what a squash merge appends -- is dropped from the title
    first. A title shorter than 12 characters is not accepted as a name, because a
    short one ("fix", "docs: x") would match unrelated text.
    """
    if cites(comment, number):
        return True
    bare = re.sub(r"\s+\(#\d+\)$", "", title).strip()
    return len(bare) >= 12 and bare in comment


def plan_issue_set(plan_text: str) -> set[int]:
    """Every `#N` in the Fixes and Refs columns of the plan's section 3 table."""
    lines = plan_text.splitlines()
    try:
        start = lines.index("## 3. The sequence")
    except ValueError as exc:
        raise ManifestError("plan has no '## 3. The sequence' section") from exc
    header = next((i for i in range(start, len(lines)) if lines[i].startswith("| PR |")),
                  None)
    if header is None:
        raise ManifestError("plan section 3 has no '| PR |' table")
    cols = [c.strip() for c in lines[header].strip().strip("|").split("|")]
    if "Fixes" not in cols or "Refs" not in cols:
        raise ManifestError(f"plan section 3 table lacks Fixes or Refs: {cols}")
    fixes, refs = cols.index("Fixes"), cols.index("Refs")
    found: set[int] = set()
    for ln in lines[header + 2:]:
        if not ln.startswith("|"):
            break
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) != len(cols):
            raise ManifestError(f"plan row has {len(cells)} cells, header {len(cols)}: {ln!r}")
        for idx in (fixes, refs):
            found.update(int(n) for n in re.findall(r"(?<![\w/])#(\d+)(?!\d)", cells[idx]))
    if not found:
        raise ManifestError("plan section 3 table names no issue")
    return found


def parse_node_ids(stdout: str) -> list[str]:
    """Collected node ids from `pytest --collect-only -q`, with a closed grammar.

    A line containing `::` that is not a node id is an error, and so is a
    duplicate id -- a collapsed parametrisation would otherwise let a count come
    out right by accident. Parameter ids may contain spaces and `::`.
    """
    ids: list[str] = []
    for raw in stdout.splitlines():
        line = raw.rstrip()
        if "::" not in line:
            continue
        if not NODE_ID.match(line):
            raise ManifestError(f"unparseable collected line: {line!r}")
        ids.append(line)
    dupes = sorted(k for k, v in Counter(ids).items() if v > 1)
    if dupes:
        raise ManifestError(f"duplicate collected node ids: {dupes}")
    return ids


def _node_from_junit(classname: str, name: str, tree: pathlib.Path) -> str:
    """Rebuild a pytest node id from JUnit's dotted classname and test name."""
    parts = classname.split(".")
    for cut in range(len(parts), 0, -1):
        path = "/".join(parts[:cut]) + ".py"
        if (tree / path).exists():
            return "::".join([path, *parts[cut:], name])
    raise ManifestError(f"cannot map JUnit classname {classname!r} to a file")


def _collection_error_node(name: str, tree: pathlib.Path) -> str:
    """The file a JUnit collection error names (`classname=""`, dotted module in `name`)."""
    path = name.replace(".", "/") + ".py"
    if not (tree / path).exists():
        raise ManifestError(f"cannot map JUnit collection error {name!r} to a file")
    return path


def parse_junit(xml_text: str, tree: pathlib.Path) -> list[Case]:
    """Every JUnit case with its outcome: passed / failed / error / skipped / xfailed.

    A collection error is reported with an empty classname and the module's dotted
    name; it becomes an `error` case on that file (design review round 3, finding 4).
    """
    cases: list[Case] = []
    for case in ET.fromstring(xml_text).iter("testcase"):
        tags = {child.tag: child for child in case}
        if not case.get("classname") and "error" in tags:
            cases.append(Case(_collection_error_node(case.get("name", ""), tree), "error",
                              tags["error"].get("message", "")))
            continue
        node = _node_from_junit(case.get("classname", ""), case.get("name", ""), tree)
        if "failure" in tags:
            cases.append(Case(node, "failed", tags["failure"].get("message", "")))
        elif "error" in tags:
            cases.append(Case(node, "error", tags["error"].get("message", "")))
        elif "skipped" in tags:
            kind = tags["skipped"].get("type", "")
            cases.append(Case(node, "xfailed" if "xfail" in kind else "skipped",
                              tags["skipped"].get("message", "")))
        else:
            cases.append(Case(node, "passed", ""))
    return cases


def outcomes(cases: list[Case]) -> Counter[str]:
    return Counter(c.outcome for c in cases)


def unexplained(cases: list[Case], expected: list[dict[str, Any]]) -> list[str]:
    """Non-passing cases no `expected_nonpass` entry accounts for, and entry miscounts.

    An entry binds a test (a node id, or a function id matching its parametrised
    cases), an outcome and a reason fragment of the message; its count is exact.
    Binding by identity, not by count, is what stops a known failure from
    standing in for a new one (design review round 2, finding 5).
    """
    problems: list[str] = []
    used = Counter[int]()
    for c in cases:
        if c.outcome == "passed":
            continue
        hits = [i for i, e in enumerate(expected)
                if e["outcome"] == c.outcome and e["reason"] in c.message
                and (c.node == e["test"] or c.node.startswith(e["test"] + "["))]
        if len(hits) != 1:
            problems.append(f"{c.outcome} {c.node}: {c.message[:80]!r} matches {len(hits)} "
                            f"declared entries")
            continue
        used[hits[0]] += 1
    for i, e in enumerate(expected):
        if used[i] != e["count"]:
            problems.append(f"declared {e['count']} {e['outcome']} for {e['test']}, "
                            f"saw {used[i]}")
    return problems


def unreconciled(cases: list[Case], collected: list[str]) -> list[str]:
    """Collected nodes JUnit did not report, and reported nodes nobody collected.

    Compared as multisets of node ids, not as counts: a substituted node or a
    duplicate report keeps the count right (design review round 3, finding 2).
    """
    reported, wanted = Counter(c.node for c in cases), Counter(collected)
    missing, extra = sorted((wanted - reported).elements()), sorted((reported - wanted).elements())
    problems = []
    if missing:
        problems.append(f"{len(missing)} collected nodes not reported: {missing[:3]}")
    if extra:
        problems.append(f"{len(extra)} reported nodes not collected: {extra[:3]}")
    return problems


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_row(num: str, row: dict[str, Any]) -> None:
    """Refuse a row that does not state what each proposition needs."""
    if not num.isdigit():
        raise ManifestError(f"issue key is not a number: {num!r}")
    # Every membership test below is preceded by a type check: an unhashable value
    # must be a ManifestError, not a TypeError (design review round 3, finding 5).
    disp = row.get("disposition")
    if not isinstance(disp, str) or disp not in DISPOSITIONS:
        raise ManifestError(f"issue {num}: disposition {disp!r} not in {sorted(DISPOSITIONS)}")
    prs = row.get("github_prs")
    if not isinstance(prs, list) or not all(_positive_int(p) for p in prs):
        raise ManifestError(f"issue {num}: github_prs must be a list of PR numbers")
    if not prs and disp not in {"regression", "not-planned"}:
        raise ManifestError(f"issue {num}: a {disp} row must name the PRs that touched it")
    if disp != "regression" and not _text(row.get("justification")):
        raise ManifestError(f"issue {num}: a {disp} row needs a justification")
    comment = row.get("closure_comment")
    if disp != "partial" and prs and not _positive_int(comment):
        raise ManifestError(f"issue {num}: closure_comment must pin a comment id")
    tests = row.get("tests")
    if not isinstance(tests, list) or not all(_text(t) for t in tests):
        raise ManifestError(f"issue {num}: tests must be a list of paths")
    if disp == "not-planned":
        if tests:
            raise ManifestError(f"issue {num}: a not-planned row names no tests")
        return
    if not tests:
        raise ManifestError(f"issue {num}: names no test")
    has_test = _text(row.get("population_test"))
    has_note = _text(row.get("population_note"))
    if has_test == has_note:
        raise ManifestError(
            f"issue {num}: exactly one of population_test and population_note is required")
    pop = row.get("population")
    if pop is not None and not _positive_int(pop):
        raise ManifestError(f"issue {num}: population must be a positive integer")
    derived = row.get("derived_from")
    if derived is not None and not _text(derived):
        raise ManifestError(f"issue {num}: derived_from must be a non-empty snippet")
    if has_test and pop is None and derived is None and prs:
        raise ManifestError(
            f"issue {num}: a population_test needs a population or a derived_from")
    expected = row.get("expected_nonpass", [])
    if not isinstance(expected, list):
        raise ManifestError(f"issue {num}: expected_nonpass must be a list")
    for e in expected:
        if (not isinstance(e, dict) or set(e) != {"test", "outcome", "reason", "count"}
                or not isinstance(e["outcome"], str) or e["outcome"] not in NONPASS
                or not _text(e["test"])
                or not _text(e["reason"]) or not _positive_int(e["count"])):
            raise ManifestError(f"issue {num}: malformed expected_nonpass entry {e!r}")
    mutation = row.get("red_mutation")
    if mutation is not None:
        if disp != "regression":
            raise ManifestError(f"issue {num}: red_mutation is for a regression row only")
        if (not isinstance(mutation, dict)
                or set(mutation) != {"file", "old", "new", "fix_text", "why"}
                or not all(_text(mutation[k]) for k in ("file", "old", "fix_text", "why"))
                or not isinstance(mutation["new"], str)):
            raise ManifestError(f"issue {num}: malformed red_mutation {mutation!r}")
        if mutation["fix_text"] not in mutation["old"] or mutation["fix_text"] in mutation["new"]:
            raise ManifestError(
                f"issue {num}: red_mutation must remove its fix_text (in old, not in new)")


def load_manifest(path: pathlib.Path = MANIFEST,
                  plan_path: pathlib.Path = PLAN) -> dict[str, Any]:
    """Parse and validate the manifest, and require it to cover the plan's issue set."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestError("manifest is not a JSON object")
    issues = data.get("issues")
    if not isinstance(issues, dict) or not issues:
        raise ManifestError("manifest has no `issues` mapping")
    for num, row in issues.items():
        if not isinstance(row, dict):
            raise ManifestError(f"issue {num}: row is not an object")
        validate_row(num, row)
    planned = plan_issue_set(plan_path.read_text(encoding="utf-8"))
    declared = {int(n) for n in issues}
    if declared != planned:
        raise ManifestError(
            f"manifest rows differ from plan section 3: missing {sorted(planned - declared)}, "
            f"extra {sorted(declared - planned)}")
    return data


# --------------------------------------------------------------------------
# Runner -- everything that touches the outside world, replaceable in tests
# --------------------------------------------------------------------------


class Runner:
    def __init__(self, repo: pathlib.Path, python: pathlib.Path, scratch: pathlib.Path) -> None:
        self.repo = repo
        # absolute(), never resolve(): resolving follows the venv symlink to the
        # base interpreter, which has no pytest (measurement item 3).
        self.python = str(python.absolute())
        self.scratch = scratch

    def env(self) -> dict[str, str]:
        return dict(os.environ, PYTHONDONTWRITEBYTECODE="1")

    def sh(self, cmd: list[str], cwd: pathlib.Path) -> tuple[int, str]:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=self.env())
        return p.returncode, p.stdout + p.stderr

    def git(self, *args: str) -> str:
        rc, out = self.sh(["git", *args], self.repo)
        if rc != 0:
            raise ManifestError(f"git {' '.join(args)} failed: {out.strip()[:200]}")
        return out

    def rev(self, ref: str) -> str:
        return self.git("rev-parse", "--verify", f"{ref}^{{commit}}").strip()

    def first_parent(self, sha: str) -> str:
        return self.rev(f"{sha}^1")

    def is_ancestor(self, sha: str, head: str) -> bool:
        rc, out = self.sh(["git", "merge-base", "--is-ancestor", sha, head], self.repo)
        if rc not in (0, 1):
            raise ManifestError(f"git merge-base failed: {out.strip()[:200]}")
        return rc == 0

    def added_lines(self, merge: str, path: str) -> list[str]:
        """The `+` lines the merge commit added to `path`, against its first parent."""
        diff = self.git("diff", f"{merge}^1", merge, "--", path)
        return [ln[1:] for ln in diff.splitlines()
                if ln.startswith("+") and not ln.startswith("+++")]

    def worktree(self, sha: str) -> pathlib.Path:
        """A detached worktree at `sha`, with the generated `_version.py` copied in.

        `lizyml/_version.py` is generated by hatch-vcs and ignored, so a fresh
        worktree lacks it and `import lizyml` fails there (measurement item 3).
        """
        dest = self.scratch / sha
        if not dest.exists():
            rc, out = self.sh(["git", "worktree", "add", "--detach", str(dest), sha], self.repo)
            if rc != 0:
                raise ManifestError(f"git worktree add {sha} failed: {out.strip()[:200]}")
        rc, out = self.sh(["git", "rev-parse", "HEAD"], dest)
        if rc != 0 or out.strip() != sha:
            raise ManifestError(
                f"worktree {dest} is at {out.strip()[:12]!r}, not {sha}; a reused scratch "
                f"directory would be measured against the wrong commit")
        version = self.repo / "lizyml" / "_version.py"
        if not version.exists():
            raise ManifestError(f"{version} is missing; install the package first")
        shutil.copy2(version, dest / "lizyml" / "_version.py")
        return dest

    def pytest(self, tree: pathlib.Path, args: list[str]) -> tuple[int, str]:
        # `-m pytest` with cwd=tree imports the tree's package: the editable
        # install's path entry comes after cwd (measurement item 3).
        return self.sh([self.python, "-m", "pytest", *args, "--no-cov", "-p",
                        "no:cacheprovider", "-p", "no:randomly"], tree)

    def run_tests(self, tree: pathlib.Path, tests: list[str]) -> tuple[int, str, list[Case]]:
        with tempfile.TemporaryDirectory() as d:
            report = pathlib.Path(d) / "junit.xml"
            rc, out = self.pytest(tree, [*tests, "-q", f"--junitxml={report}"])
            if not report.exists():
                raise ManifestError(f"pytest wrote no report: {out.strip()[-200:]}")
            return rc, out, parse_junit(report.read_text(encoding="utf-8"), tree)

    def collect(self, tree: pathlib.Path, tests: list[str]) -> list[str]:
        rc, out = self.pytest(tree, [*tests, "--collect-only", "-q"])
        if rc != 0:
            raise ManifestError(f"collection failed (exit {rc}): {out.strip()[-200:]}")
        return parse_node_ids(out)

    def python_c(self, tree: pathlib.Path, snippet: str) -> tuple[int, str]:
        # Always `-c`, never a script file: a file puts its own directory first
        # on sys.path and imports the main checkout (measurement item 3).
        return self.sh([self.python, "-c", snippet], tree)

    def derive(self, tree: pathlib.Path, snippet: str) -> int:
        rc, out = self.python_c(tree, snippet)
        if rc != 0:
            raise ManifestError(f"derivation failed: {out.strip()[-200:]}")
        lines = [ln for ln in out.strip().splitlines() if ln.strip()]
        try:
            return int(lines[-1])
        except (IndexError, ValueError) as exc:
            raise ManifestError(f"derivation printed {out.strip()[-80:]!r}, not an int") from exc

    def graphql(self, query: str, number: int) -> dict[str, Any]:
        rc, out = self.sh(["gh", "api", "graphql", "-f", f"query={query}", "-F",
                           f"owner={OWNER}", "-F", f"name={NAME}", "-F", f"n={number}"],
                          self.repo)
        if rc != 0:
            raise ManifestError(f"gh query for #{number} failed: {out.strip()[:200]}")
        return json.loads(out)["data"]["repository"]

    def issue(self, number: int) -> dict[str, Any]:
        q = ("query($owner:String!,$name:String!,$n:Int!){repository(owner:$owner,name:$name)"
             "{issue(number:$n){state stateReason closedAt "
             "comments(first:100){nodes{databaseId createdAt body}}}}}")
        data = self.graphql(q, number)["issue"]
        if data is None:
            raise ManifestError(f"#{number} is not an issue")
        return data

    def pull(self, number: int) -> dict[str, Any]:
        q = ("query($owner:String!,$name:String!,$n:Int!){repository(owner:$owner,name:$name)"
             "{pullRequest(number:$n){state title body mergedAt mergeCommit{oid}}}}")
        data = self.graphql(q, number)["pullRequest"]
        if data is None:
            raise ManifestError(f"#{number} is not a pull request")
        return data


# --------------------------------------------------------------------------
# Proposition 2
# --------------------------------------------------------------------------


def files_to_stage(after: pathlib.Path, tests: list[str]) -> list[str]:
    """The row's tests and every tests/ helper and package marker -- never a package file.

    tests/ helpers count as part of the test: proposition 2 asks whether the after
    test fails against the before system. No `lizyml/` file is ever staged, so the
    before system is exactly the before commit. Staging the package files only the
    after tree had (rounds 1-6) could manufacture a false RED through a guarded
    import, and the guard meant to see that was refuted six times; it was removed
    (maintainer decision after round 6, results/pr8c_options_analysis_after_round6.md).
    A row whose tests cannot run in its before tree shows RED by `red_mutation`.
    """
    staged = list(tests)
    for p in sorted((after / "tests").rglob("*.py")):
        if "__pycache__" not in p.parts and (p.name == "__init__.py" or p.name.startswith("_")):
            staged.append(p.relative_to(after).as_posix())
    return list(dict.fromkeys(staged))


@contextlib.contextmanager
def staged(before: pathlib.Path, after: pathlib.Path, files: list[str]) -> Iterator[None]:
    """Copy `files` from after into before, and put before back exactly afterwards."""
    saved = {f: (before / f).read_bytes() for f in files if (before / f).exists()}
    created: list[str] = []
    try:
        for f in files:
            dst = before / f
            if f not in saved:
                created.append(f)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(after / f, dst)
        yield
    finally:
        for f, data in saved.items():
            (before / f).write_bytes(data)
        for f in created:
            (before / f).unlink(missing_ok=True)


@contextlib.contextmanager
def mutated(tree: pathlib.Path, mutation: dict[str, str]) -> Iterator[None]:
    """Apply a red_mutation to `tree` and restore the file afterwards."""
    path = tree / mutation["file"]
    original = path.read_text(encoding="utf-8")
    found = original.count(mutation["old"])
    if found != 1:
        raise ManifestError(f"red_mutation matches {found} times in {mutation['file']}, not 1")
    try:
        path.write_text(original.replace(mutation["old"], mutation["new"]), encoding="utf-8")
        yield
    finally:
        path.write_text(original, encoding="utf-8")


def _merges(row: dict[str, Any], runner: Runner) -> list[tuple[str, str]]:
    merges = []
    for pr_num in row["github_prs"]:
        oid = (runner.pull(pr_num).get("mergeCommit") or {}).get("oid")
        if not oid:
            raise ManifestError(f"#{pr_num} has no merge commit")
        merges.append((runner.pull(pr_num)["mergedAt"], oid))
    return sorted(merges)


def _p2_before_tree(row: dict[str, Any], runner: Runner, after: pathlib.Path,
                    r: dict[str, Any]) -> str | None:
    """Run the row's tests in the before tree; return why it is not red, or None."""
    before = runner.worktree(runner.first_parent(_merges(row, runner)[0][1]))
    r["before"] = before.name[:7]
    with staged(before, after, files_to_stage(after, row["tests"])):
        _, _, cases = runner.run_tests(before, row["tests"])
    r["p2"] = dict(outcomes(cases))
    return None if outcomes(cases)["failed"] else "no failed node in the before tree"


def _p2_mutation(row: dict[str, Any], runner: Runner, after: pathlib.Path,
                 r: dict[str, Any]) -> str | None:
    """Run the row's tests with the defect put back; return why it is not red, or None."""
    m = row["red_mutation"]
    added = [ln for _, oid in _merges(row, runner) for ln in runner.added_lines(oid, m["file"])]
    if not any(m["fix_text"] in ln for ln in added):
        return f"fix_text {m['fix_text']!r} is not text the fixing PRs added to {m['file']}"
    with mutated(after, m):
        rc, out, cases = runner.run_tests(after, row["tests"])
    r["p2"] = f"mutated: {dict(outcomes(cases))}"
    if "error" in outcomes(cases) or "ERROR collecting" in out:
        return "the mutation broke collection; it must leave the tests runnable"
    return None if outcomes(cases)["failed"] else "no failed node with the defect put back"


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def _p6(num: int, row: dict[str, Any], runner: Runner, after_sha: str) -> tuple[bool, str]:
    issue = runner.issue(num)
    disp = row["disposition"]
    if disp == "partial":
        return issue["state"] == "OPEN", f"issue {issue['state']}"
    want = "NOT_PLANNED" if disp == "not-planned" else "COMPLETED"
    if issue["state"] != "CLOSED" or issue["stateReason"] != want:
        return False, f"issue {issue['state']}/{issue['stateReason']}, want CLOSED/{want}"
    if not row["github_prs"]:
        return True, f"closed {want}"
    pinned = [c for c in issue["comments"]["nodes"]
              if c["databaseId"] == row["closure_comment"]]
    if len(pinned) != 1:
        return False, f"pinned comment {row['closure_comment']} is not on #{num}"
    comment = pinned[0]
    if comment["createdAt"] > issue["closedAt"]:
        return False, "the pinned comment was written after the close"
    for pr_num in row["github_prs"]:
        pr = runner.pull(pr_num)
        oid = (pr.get("mergeCommit") or {}).get("oid")
        if pr["state"] != "MERGED" or not oid:
            return False, f"#{pr_num} is {pr['state']}"
        if not runner.is_ancestor(oid, after_sha):
            return False, f"#{pr_num} merge {oid[:7]} is not an ancestor of the after SHA"
        if not cites(pr["body"] or "", num):
            return False, f"#{pr_num} body does not cite #{num}"
        if pr["mergedAt"] > comment["createdAt"]:
            return False, f"the pinned comment predates the merge of #{pr_num}"
        if not names_pr(comment["body"], pr_num, pr["title"]):
            return False, f"the pinned comment does not name #{pr_num}"
    return True, "closed by the named PRs"


def evaluate_row(num: int, row: dict[str, Any], runner: Runner,
                 after: pathlib.Path, after_sha: str) -> dict[str, Any]:
    disp = row["disposition"]
    r: dict[str, Any] = {"disposition": disp, "reasons": []}

    def fail(why: str) -> None:
        r["reasons"].append(why)

    if disp == "not-planned":
        ok, why = _p6(num, row, runner, after_sha)
        r["p6"] = why
        r["verdict"] = "NOT-PLANNED" if ok else "INCOMPLETE"
        if not ok:
            fail(f"p6: {why}")
        return r

    tests = row["tests"]
    missing = [t for t in tests if not (after / t).exists()]
    if missing:
        fail(f"p1: missing at after: {missing}")
    if not row["github_prs"]:
        fail("no fixing PR recorded")
    if r["reasons"]:
        r["verdict"] = "INCOMPLETE"
        return r

    if disp == "regression":
        if row.get("red_mutation"):
            why_not = _p2_mutation(row, runner, after, r)
        else:
            why_not = _p2_before_tree(row, runner, after, r)
        if why_not:
            fail(f"p2: {why_not}")

    rc, _, cases = runner.run_tests(after, tests)
    r["p3"] = dict(outcomes(cases))
    ids = runner.collect(after, tests)
    problems = unexplained(cases, row.get("expected_nonpass", []))
    problems.extend(unreconciled(cases, ids))
    if not outcomes(cases)["passed"]:
        problems.append("nothing passed")
    if rc != 0:
        problems.append(f"pytest exit {rc}")
    for p in problems:
        fail(f"p3: {p}")

    derived = runner.derive(after, row["derived_from"]) if row.get("derived_from") else None
    r["p5"] = derived
    pop = row.get("population")
    if row.get("population_test"):
        prefix = row["population_test"]
        n = sum(1 for i in ids if i == prefix or i.startswith(prefix + "["))
        r["p4"] = n
        want = pop if pop is not None else derived
        if n != want:
            fail(f"p4: {prefix} collects {n}, declared {want}")
    else:
        r["p4"] = f"declared {pop} ({row['population_note']})"
        if not ids:
            fail("p4: the named tests collect nothing")
    if derived is not None and pop is not None and derived != pop:
        fail(f"p5: derived {derived} != declared {pop}")

    ok, why = _p6(num, row, runner, after_sha)
    r["p6"] = why
    if not ok:
        fail(f"p6: {why}")

    if r["reasons"]:
        r["verdict"] = "INCOMPLETE"
    elif disp == "partial":
        r["verdict"] = "PARTIAL"
    elif row.get("red_mutation"):
        r["verdict"] = "COMPLETE-RED-BY-MUTATION"
    else:
        r["verdict"] = "COMPLETE"
    return r


def evaluate(data: dict[str, Any], runner: Runner, after_ref: str) -> dict[int, dict[str, Any]]:
    after_sha = runner.rev(after_ref)
    after = runner.worktree(after_sha)
    results: dict[int, dict[str, Any]] = {}
    for key, row in sorted(data["issues"].items(), key=lambda kv: int(kv[0])):
        num = int(key)
        try:
            results[num] = evaluate_row(num, row, runner, after, after_sha)
        except ManifestError as exc:
            results[num] = {"disposition": row["disposition"], "verdict": "UNKNOWN",
                            "reasons": [str(exc)]}
    return results


def summarise(results: dict[int, dict[str, Any]]) -> Counter[str]:
    return Counter(r["verdict"] for r in results.values())


def report(results: dict[int, dict[str, Any]], out: Any = sys.stdout) -> int:
    for num, r in results.items():
        print(f"#{num}: {r['verdict']} ({r['disposition']})", file=out)
        for k in ("before", "p2", "p3", "p4", "p5", "p6"):
            if k in r:
                print(f"    {k} = {r[k]}", file=out)
        for why in r["reasons"]:
            print(f"    ! {why}", file=out)
    counts = summarise(results)
    print(file=out)
    print("   ".join(f"{v} {counts.get(v, 0)}" for v in VERDICTS) + f"   of {len(results)}",
          file=out)
    print("INCOMPLETE and UNKNOWN count against completion; PARTIAL, NOT-PLANNED and "
          "COMPLETE-RED-BY-MUTATION are reported apart from COMPLETE.", file=out)
    return 1 if counts.get("INCOMPLETE", 0) or counts.get("UNKNOWN", 0) else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Measure Phase 3 completion (plan section 8).")
    ap.add_argument("--after", required=True, help="the commit to measure, e.g. origin/develop")
    ap.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
    ap.add_argument("--python", type=pathlib.Path, default=pathlib.Path(".venv/bin/python"))
    ap.add_argument("--scratch", type=pathlib.Path,
                    default=pathlib.Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
                    / "phase3-gap")
    args = ap.parse_args(argv)
    args.scratch.mkdir(parents=True, exist_ok=True)
    data = load_manifest()
    python = args.python if args.python.is_absolute() else args.repo / args.python
    runner = Runner(args.repo.absolute(), python, args.scratch)
    return report(evaluate(data, runner, args.after))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
