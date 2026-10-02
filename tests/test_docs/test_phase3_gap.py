"""Unit tests for the Phase 3 completion instrument's quiet failure modes.

`docs/audits/2026-09-defect-discovery/instruments/phase3_gap.py` is a run tool:
it creates git worktrees and runs pytest in each, so the suite does not run it.
These tests cover what could report a wrong verdict without erroring: the
manifest grammar, the plan's issue set, node-id counting, staging the before
tree (tests and helpers only, never a package file), the reintroduction
mutation, per-node outcomes and their declared exceptions, the closure
association, and the verdict arithmetic. A fake runner drives each proposition;
staging and the counterexamples of design review rounds 1-7 run real pytest in
temporary trees.
"""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import pathlib
import sys
from collections import Counter
from typing import Any

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
INSTRUMENTS = ROOT / "docs" / "audits" / "2026-09-defect-discovery" / "instruments"
_spec = importlib.util.spec_from_file_location(
    "phase3_gap", INSTRUMENTS / "phase3_gap.py"
)
assert _spec is not None and _spec.loader is not None
gap = importlib.util.module_from_spec(_spec)
sys.modules["phase3_gap"] = gap
_spec.loader.exec_module(gap)

AFTER_SHA = "a" * 40
MERGE = "m" * 40
PARENT = "p" * 40
NODES = [f"tests/test_x.py::test_cell[{i}]" for i in range(3)]

REGRESSION: dict[str, Any] = {
    "github_prs": [500],
    "disposition": "regression",
    "closure_comment": 9001,
    "tests": ["tests/test_x.py"],
    "population_test": "tests/test_x.py::test_cell",
    "population": 3,
}
MUTATION = {
    "file": "lizyml/core.py",
    "old": "return forward(params)",
    "new": "return forward(None)",
    "fix_text": "forward(params)",
    "why": "puts the dropped argument back",
}


def _cases(*specs: tuple[str, str, str]) -> list[Any]:
    return [gap.Case(node, outcome, message) for node, outcome, message in specs]


def _passing(nodes: list[str] = NODES) -> list[Any]:
    return _cases(*((n, "passed", "") for n in nodes))


# --------------------------------------------------------------------------
# A runner whose outside world is scripted
# --------------------------------------------------------------------------


class FakeRunner(gap.Runner):
    def __init__(self, tmp: pathlib.Path) -> None:
        super().__init__(tmp, tmp / "python", tmp / "scratch")
        self.before_tree = tmp / "before"
        (self.before_tree / "lizyml").mkdir(parents=True)
        self.after_tree = tmp / "after"
        (self.after_tree / "tests").mkdir(parents=True)
        (self.after_tree / "lizyml").mkdir()
        (self.after_tree / "tests" / "test_x.py").write_text(
            "def test_cell():\n    pass\n"
        )
        (self.after_tree / "lizyml" / "core.py").write_text(
            "def f(params):\n    return forward(params)\n"
        )
        self.worktrees: list[str] = []
        self.before_cases = _cases(
            (NODES[0], "failed", "boom"), *((n, "passed", "") for n in NODES[1:])
        )
        self.mutated_cases = list(self.before_cases)
        self.mutated_out = ""
        self.after_cases = _passing()
        self.after_rc = 0
        self.collected = list(NODES)
        self.derived = 3
        self.added = {MERGE: ["    return forward(params)"]}
        self.issue_data: dict[str, Any] = {
            "state": "CLOSED",
            "stateReason": "COMPLETED",
            "closedAt": "2026-10-02T00:00:00Z",
            "comments": {
                "nodes": [
                    {
                        "databaseId": 9001,
                        "createdAt": "2026-10-01T12:00:00Z",
                        "body": "Fixed by #500, merged.",
                    }
                ]
            },
        }
        self.prs: dict[int, dict[str, Any]] = {
            500: {
                "state": "MERGED",
                "title": "fix(core): repair the thing properly (#500)",
                "body": "Fixes #42",
                "mergedAt": "2026-10-01T00:00:00Z",
                "mergeCommit": {"oid": MERGE},
            },
        }
        self.ancestors = {MERGE}
        self.first_parent_of: dict[str, str] = {MERGE: PARENT}
        self.comment_issue = 42

    def rev(self, ref: str) -> str:
        return AFTER_SHA

    def first_parent(self, sha: str) -> str:
        return self.first_parent_of[sha]

    def is_ancestor(self, sha: str, head: str) -> bool:
        return sha in self.ancestors

    def added_lines(self, merge: str, path: str) -> list[str]:
        return self.added.get(merge, [])

    def worktree(self, sha: str) -> pathlib.Path:
        if sha == AFTER_SHA:
            return self.after_tree
        self.worktrees.append(sha)
        return self.before_tree

    def run_tests(
        self, tree: pathlib.Path, tests: list[str]
    ) -> tuple[int, str, list[Any]]:
        if tree == self.before_tree:
            return 1, "", list(self.before_cases)
        if "forward(None)" in (tree / "lizyml" / "core.py").read_text():
            return 1, self.mutated_out, list(self.mutated_cases)
        return self.after_rc, "", list(self.after_cases)

    def collect(self, tree: pathlib.Path, tests: list[str]) -> list[str]:
        return list(self.collected)

    def derive(self, tree: pathlib.Path, snippet: str) -> int:
        return self.derived

    def issue(self, number: int) -> dict[str, Any]:
        return self.issue_data

    def comment(self, comment_id: int) -> dict[str, Any] | None:
        for c in self.issue_data["comments"]["nodes"]:
            if c["databaseId"] == comment_id:
                return {"issue": self.comment_issue, **c}
        return None

    def pull(self, number: int) -> dict[str, Any]:
        return self.prs[number]


def _evaluate(runner: FakeRunner, row: dict[str, Any], num: int = 42) -> dict[str, Any]:
    return gap.evaluate_row(num, row, runner, runner.after_tree, AFTER_SHA)


def _tree(root: pathlib.Path, files: dict[str, str]) -> pathlib.Path:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root


# --------------------------------------------------------------------------
# Manifest grammar and the plan's issue set
# --------------------------------------------------------------------------


def _bad_mutation(**change: str) -> Any:
    return lambda r: r.update(red_mutation={**MUTATION, **change})


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda r: r.update(disposition="done"), id="unknown-disposition"),
        pytest.param(lambda r: r.update(disposition=[]), id="unhashable-disposition"),
        pytest.param(
            lambda r: r.update(
                expected_nonpass=[
                    {"test": "t", "outcome": ["skipped"], "reason": "x", "count": 1}
                ]
            ),
            id="unhashable-nonpass-outcome",
        ),
        pytest.param(lambda r: r.update(github_prs=500), id="prs-not-a-list"),
        pytest.param(lambda r: r.update(github_prs=[0]), id="pr-zero"),
        pytest.param(lambda r: r.update(github_prs=[True]), id="pr-bool"),
        pytest.param(lambda r: r.pop("closure_comment"), id="no-pinned-comment"),
        pytest.param(lambda r: r.update(closure_comment="url"), id="pinned-not-an-id"),
        pytest.param(
            lambda r: r.update(disposition="partial"), id="partial-no-justification"
        ),
        pytest.param(lambda r: r.update(tests=[]), id="no-tests"),
        pytest.param(
            lambda r: r.update(tests="tests/test_x.py"), id="tests-not-a-list"
        ),
        pytest.param(
            lambda r: r.update(tests=["tests/test_x.py", "lizyml/new.py"]),
            id="round-7-package-file-as-a-test",
        ),
        pytest.param(
            lambda r: r.update(tests=["tests/../lizyml/new.py"]), id="test-path-escapes"
        ),
        pytest.param(
            lambda r: r.update(tests=["/abs/tests/test_x.py"]), id="test-path-absolute"
        ),
        pytest.param(
            lambda r: r.update(tests=["tests/_helpers.py"]), id="test-path-not-a-test"
        ),
        pytest.param(lambda r: r.pop("population_test"), id="neither-test-nor-note"),
        pytest.param(
            lambda r: r.update(population_note="both"), id="both-test-and-note"
        ),
        pytest.param(lambda r: r.update(population=0), id="population-zero"),
        pytest.param(lambda r: r.update(population=True), id="population-bool"),
        pytest.param(lambda r: r.pop("population"), id="no-population-no-derivation"),
        pytest.param(lambda r: r.update(derived_from=" "), id="blank-derivation"),
        pytest.param(lambda r: r.update(expected_nonpass={}), id="nonpass-not-a-list"),
        pytest.param(
            lambda r: r.update(
                expected_nonpass=[
                    {"test": "t", "outcome": "failed", "reason": "x", "count": 1}
                ]
            ),
            id="nonpass-allows-a-failure",
        ),
        pytest.param(
            lambda r: r.update(
                expected_nonpass=[{"test": "t", "outcome": "skipped", "reason": "x"}]
            ),
            id="nonpass-without-a-count",
        ),
        pytest.param(
            _bad_mutation(fix_text="not in old"), id="mutation-keeps-no-fix-text"
        ),
        pytest.param(
            _bad_mutation(new="return forward(params) or 1"),
            id="mutation-leaves-the-fix-text",
        ),
        pytest.param(_bad_mutation(why=""), id="mutation-without-a-reason"),
        pytest.param(
            lambda r: r.update(
                disposition="decision-only",
                justification="x",
                red_mutation=dict(MUTATION),
            ),
            id="mutation-off-regression",
        ),
        pytest.param(
            lambda r: r.update(disposition="not-planned", justification="x"),
            id="not-planned-with-tests",
        ),
        pytest.param(
            lambda r: r.update(disposition="partial", justification="x", github_prs=[]),
            id="partial-without-prs",
        ),
    ],
)
def test_a_malformed_row_is_refused(mutate: Any) -> None:
    row = copy.deepcopy(REGRESSION)
    mutate(row)
    with pytest.raises(gap.ManifestError):
        gap.validate_row("42", row)


def test_a_well_formed_row_is_accepted() -> None:
    gap.validate_row("42", copy.deepcopy(REGRESSION))
    gap.validate_row(
        "43", {**copy.deepcopy(REGRESSION), "red_mutation": dict(MUTATION)}
    )
    gap.validate_row(
        "44",
        {
            "github_prs": [],
            "disposition": "not-planned",
            "justification": "closed not planned",
            "tests": [],
        },
    )


def test_the_shipped_manifest_covers_the_plan_issue_set() -> None:
    data = gap.load_manifest()
    planned = gap.plan_issue_set(gap.PLAN.read_text(encoding="utf-8"))
    assert {int(n) for n in data["issues"]} == planned
    assert len(planned) == 25


PLAN_HEAD = (
    "## 3. The sequence\n\n"
    "| PR | Title | Fixes | Refs | Status |\n|---|---|---|---|---|\n"
)


def test_the_plan_table_reader_reads_both_columns() -> None:
    text = (
        PLAN_HEAD + "| 1 | a | #10, #11 | #12 | x |\n| 2 | b | — | — | #99 |\n\nafter\n"
    )
    assert gap.plan_issue_set(text) == {10, 11, 12}


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("no section here\n", id="no-section"),
        pytest.param("## 3. The sequence\n\nno table\n", id="no-table"),
        pytest.param(
            PLAN_HEAD.replace("Refs", "Notes") + "| 1 | a | #10 | x | y |\n",
            id="no-refs-column",
        ),
        pytest.param(PLAN_HEAD + "| 1 | a | #10 |\n", id="short-row"),
        pytest.param(PLAN_HEAD + "| 1 | a | — | — | x |\n", id="no-issue"),
    ],
)
def test_the_plan_table_reader_refuses_a_malformed_table(text: str) -> None:
    with pytest.raises(gap.ManifestError):
        gap.plan_issue_set(text)


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        pytest.param(
            "tests/a.py::test_x[1]\ntests/a.py::test_x[2]\n\n2 tests collected\n",
            2,
            id="plain",
        ),
        pytest.param(
            "tests/a.py::test_x[a b::c]\n", 1, id="param-with-space-and-colons"
        ),
        pytest.param("tests/a.py::TestK::test_y\n", 1, id="class-method"),
    ],
)
def test_node_ids_are_counted(stdout: str, expected: int) -> None:
    assert len(gap.parse_node_ids(stdout)) == expected


@pytest.mark.parametrize(
    "stdout",
    [
        pytest.param("tests/a.py::test_x\ntests/a.py::test_x\n", id="duplicate"),
        pytest.param("ERROR tests/a.py::test_x - boom\n", id="not-a-node-id"),
        pytest.param("tests/a.txt::test_x\n", id="not-a-python-file"),
    ],
)
def test_node_ids_are_parsed_with_a_closed_grammar(stdout: str) -> None:
    with pytest.raises(gap.ManifestError):
        gap.parse_node_ids(stdout)


# --------------------------------------------------------------------------
# Proposition 2 -- the before tree
# --------------------------------------------------------------------------


def test_p2_red_by_a_failed_node(tmp_path: pathlib.Path) -> None:
    r = _evaluate(FakeRunner(tmp_path), copy.deepcopy(REGRESSION))
    assert r["verdict"] == "COMPLETE", r["reasons"]


@pytest.mark.parametrize(
    "cases",
    [
        pytest.param(_passing(), id="everything-passes"),
        pytest.param(
            _cases(("tests/test_x.py", "error", "collection failure")),
            id="collection-error",
        ),
    ],
)
def test_p2_not_red(tmp_path: pathlib.Path, cases: list[Any]) -> None:
    runner = FakeRunner(tmp_path)
    runner.before_cases = cases
    r = _evaluate(runner, copy.deepcopy(REGRESSION))
    assert r["verdict"] == "INCOMPLETE"
    assert any(why.startswith("p2:") for why in r["reasons"])


def test_p2_runs_at_the_first_parent_of_the_earliest_fix(
    tmp_path: pathlib.Path,
) -> None:
    runner = FakeRunner(tmp_path)
    later = "l" * 40
    runner.prs[501] = {
        **runner.prs[500],
        "mergedAt": "2026-10-01T06:00:00Z",
        "mergeCommit": {"oid": later},
        "title": "fix(core): the second half here",
        "body": "Refs #42",
    }
    runner.ancestors.add(later)
    runner.first_parent_of[later] = "q" * 40
    runner.issue_data["comments"]["nodes"][0]["body"] = "Fixed by #500 and #501."
    r = _evaluate(runner, {**copy.deepcopy(REGRESSION), "github_prs": [501, 500]})
    assert runner.worktrees == [PARENT]
    assert r["verdict"] == "COMPLETE", r["reasons"]


def test_p2_stages_no_package_file(tmp_path: pathlib.Path) -> None:
    """Option C (after design review round 6): only tests and tests/ helpers."""
    after = _tree(
        tmp_path / "a",
        {
            "lizyml/__init__.py": "",
            "lizyml/new.py": "Y = 1\n",
            "lizyml/_version.py": "",
            "tests/__init__.py": "",
            "tests/_helpers.py": "NEW = 1\n",
            "tests/_spy.py": "",
            "tests/test_x.py": "",
            "tests/test_other.py": "",
        },
    )
    staged = gap.files_to_stage(after, ["tests/test_x.py"])
    assert sorted(staged) == sorted(
        ["tests/test_x.py", "tests/__init__.py", "tests/_helpers.py", "tests/_spy.py"]
    )
    assert not [f for f in staged if f.startswith("lizyml/")]


def test_staging_refuses_a_path_outside_tests(tmp_path: pathlib.Path) -> None:
    """Round 7: a package file listed as a row's test is never copied."""
    after = _tree(tmp_path / "a", {"lizyml/new.py": "Y = 1\n", "tests/test_x.py": ""})
    with pytest.raises(gap.ManifestError):
        gap.files_to_stage(after, ["tests/test_x.py", "lizyml/new.py"])


def test_p2_restores_the_before_tree(tmp_path: pathlib.Path) -> None:
    before = _tree(tmp_path / "b", {"tests/_helpers.py": "OLD = 1\n"})
    after = _tree(
        tmp_path / "a",
        {"tests/_helpers.py": "NEW = 1\n", "tests/sub/test_new.py": "x = 1\n"},
    )
    files = ["tests/_helpers.py", "tests/sub/test_new.py"]
    with gap.staged(before, after, files):
        assert (before / "tests/_helpers.py").read_text() == "NEW = 1\n"
        assert (before / "tests/sub/test_new.py").exists()
    assert (before / "tests/_helpers.py").read_text() == "OLD = 1\n"
    assert not (before / "tests/sub/test_new.py").exists()


def _nested_exec(depth: int = 10) -> str:
    """Round 6: an import wrapped in `depth` literal `exec` calls."""
    code = "from lizyml.other.new import Y"
    for _ in range(depth):
        code = f"exec({code!r})"
    return code


_NESTED_EXEC = _nested_exec()


def _real_runner(tmp_path: pathlib.Path) -> Any:
    return gap.Runner(tmp_path, pathlib.Path(sys.executable), tmp_path / "scratch")


def test_an_unused_import_of_a_new_module_is_not_red(tmp_path: pathlib.Path) -> None:
    """Design review round 1's counterexample: an unrelated import of a new module.

    No package file is staged, so the test cannot collect in the before tree: a
    collection error, which p2 never counts as RED.
    """
    before = _tree(tmp_path / "b", {"lizyml/__init__.py": "", "tests/__init__.py": ""})
    after = _tree(
        tmp_path / "a",
        {
            "lizyml/__init__.py": "",
            "lizyml/_optimizer.py": "",
            "tests/__init__.py": "",
            "tests/test_x.py": "import lizyml._optimizer\n\ndef test_unrelated():\n"
            "    assert 1 + 1 == 2\n",
        },
    )
    tests = ["tests/test_x.py"]
    with gap.staged(before, after, gap.files_to_stage(after, tests)):
        _, _, cases = _real_runner(tmp_path).run_tests(before, tests)
    assert gap.outcomes(cases) == Counter(error=1)


_GUARDED = "except (ImportError, AttributeError, NameError):\n    Y = 0\n"
_OTHER = {"lizyml/other/__init__.py": ""}


@pytest.mark.parametrize(
    ("before_files", "new_file"),
    [
        pytest.param(
            {"lizyml/a.py": "try:\n    from lizyml.new import Y\n" + _GUARDED},
            "lizyml/new.py",
            id="round-2-guarded-import",
        ),
        pytest.param(
            {
                "lizyml/a.py": "from importlib import import_module\ntry:\n"
                "    Y = import_module('.new', __package__).Y\n" + _GUARDED
            },
            "lizyml/new.py",
            id="round-3-relative-import-module",
        ),
        pytest.param(
            {
                **_OTHER,
                "lizyml/a.py": "from importlib import import_module\ntry:\n"
                "    Y = import_module('.new', 'lizyml.other').Y\n" + _GUARDED,
            },
            "lizyml/other/new.py",
            id="round-4-explicit-package",
        ),
        pytest.param(
            {
                **_OTHER,
                "lizyml/a.py": "try:\n"
                "    Y = __import__('lizyml.other', fromlist=['new']).new.Y\n"
                + _GUARDED,
            },
            "lizyml/other/new.py",
            id="round-4-fromlist",
        ),
        pytest.param(
            {
                **_OTHER,
                "lizyml/a.py": "try:\n"
                "    Y = __import__('other', globals(), None, ['new'], 1).new.Y\n"
                + _GUARDED,
            },
            "lizyml/other/new.py",
            id="round-5-relative-dunder-import",
        ),
        pytest.param(
            {
                "lizyml/other/__init__.py": "__all__ = []\n__all__[:] = ['new']\n",
                "lizyml/a.py": "try:\n    from .other import *\n    Y = new.Y\n"
                + _GUARDED,
            },
            "lizyml/other/new.py",
            id="round-6-all-slice-assignment",
        ),
        pytest.param(
            {**_OTHER, "lizyml/a.py": "try:\n    " + _NESTED_EXEC + "\n" + _GUARDED},
            "lizyml/other/new.py",
            id="round-6-nested-exec",
        ),
        pytest.param(
            {
                **_OTHER,
                "lizyml/a.py": "from importlib import import_module\ntry:\n"
                "    Y = import_module('lizyml.other.' + 'new').Y\n" + _GUARDED,
            },
            "lizyml/other/new.py",
            id="runtime-built-name",
        ),
    ],
)
def test_staging_cannot_manufacture_red(
    tmp_path: pathlib.Path, before_files: dict[str, str], new_file: str
) -> None:
    """Every staging counterexample of design review rounds 2-6, under option C.

    The before code reaches a module only the after tree has, falling back to 0.
    Staging that module (the rule rounds 1-6 used) turns the passing test RED with
    no defect behind it; staging only the tests leaves it passing.
    """
    base = {"lizyml/__init__.py": "", "tests/__init__.py": "", **before_files}
    test = "from lizyml.a import Y\n\ndef test_y():\n    assert Y == 0\n"
    before = _tree(tmp_path / "b", base)
    after = _tree(
        tmp_path / "a", {**base, new_file: "Y = 1\n", "tests/test_x.py": test}
    )
    tests = ["tests/test_x.py"]
    runner = _real_runner(tmp_path)
    with gap.staged(before, after, [*tests, new_file]):
        _, _, old_rule = runner.run_tests(before, tests)
    assert gap.outcomes(old_rule) == Counter(failed=1)
    with gap.staged(before, after, gap.files_to_stage(after, tests)):
        _, _, cases = runner.run_tests(before, tests)
    assert gap.outcomes(cases) == Counter(passed=1)


# --------------------------------------------------------------------------
# Proposition 2 -- the reintroduction mutation
# --------------------------------------------------------------------------


def _mutation_row() -> dict[str, Any]:
    return {**copy.deepcopy(REGRESSION), "red_mutation": dict(MUTATION)}


def test_a_mutation_that_reddens_the_tests_is_its_own_verdict(
    tmp_path: pathlib.Path,
) -> None:
    runner = FakeRunner(tmp_path)
    r = _evaluate(runner, _mutation_row())
    assert r["verdict"] == "COMPLETE-RED-BY-MUTATION", r["reasons"]
    assert runner.worktrees == []
    assert "forward(params)" in (runner.after_tree / "lizyml" / "core.py").read_text()


@pytest.mark.parametrize(
    ("setup", "fragment"),
    [
        pytest.param(
            lambda r: setattr(r, "mutated_cases", _passing()),
            "no failed node",
            id="tests-pass-with-the-defect-back",
        ),
        pytest.param(
            lambda r: setattr(r, "added", {}),
            "not text the fixing PRs added",
            id="fix-text-not-added-by-the-pr",
        ),
        pytest.param(
            lambda r: (
                setattr(
                    r, "mutated_cases", _cases(("tests/test_x.py", "error", "boom"))
                ),
                setattr(r, "mutated_out", "ERROR collecting tests/test_x.py"),
            ),
            "broke collection",
            id="mutation-breaks-collection",
        ),
    ],
)
def test_a_mutation_is_not_red_evidence_when(
    tmp_path: pathlib.Path,
    setup: Any,
    fragment: str,
) -> None:
    runner = FakeRunner(tmp_path)
    setup(runner)
    r = _evaluate(runner, _mutation_row())
    assert r["verdict"] == "INCOMPLETE"
    assert any(fragment in why for why in r["reasons"]), r["reasons"]


def test_a_mutation_must_match_exactly_once(tmp_path: pathlib.Path) -> None:
    runner = FakeRunner(tmp_path)
    (runner.after_tree / "lizyml" / "core.py").write_text(
        "return forward(params)\n" * 2
    )
    with pytest.raises(gap.ManifestError, match="2 times"):
        _evaluate(runner, _mutation_row())


def test_an_unrelated_test_is_not_red_under_a_mutation(tmp_path: pathlib.Path) -> None:
    """Round 2's waiver counterexample, under the rule that replaced the waiver."""
    after = _tree(
        tmp_path / "a",
        {
            "lizyml/__init__.py": "",
            "lizyml/core.py": "def f(params):\n    return params\n",
            "tests/__init__.py": "",
            "tests/test_x.py": "from lizyml.core import f\n\ndef test_unrelated():\n"
            "    assert 1 + 1 == 2\n",
        },
    )
    runner = _real_runner(tmp_path)
    with gap.mutated(
        after,
        {
            "file": "lizyml/core.py",
            "old": "return params",
            "new": "return None",
            "fix_text": "return params",
            "why": "x",
        },
    ):
        _, _, cases = runner.run_tests(after, ["tests/test_x.py"])
    assert gap.outcomes(cases) == Counter(passed=1)
    assert "return params" in (after / "lizyml/core.py").read_text()


# --------------------------------------------------------------------------
# Propositions 3-5
# --------------------------------------------------------------------------

XF = "tests/test_x.py::test_cell"


@pytest.mark.parametrize(
    ("cases", "expected", "ok"),
    [
        pytest.param(_passing(), [], True, id="all-passed"),
        pytest.param(
            _cases(
                (NODES[0], "skipped", "why"), *((n, "passed", "") for n in NODES[1:])
            ),
            [],
            False,
            id="an-undeclared-skip",
        ),
        pytest.param(
            _cases(
                (NODES[0], "xfailed", "#299 inert"),
                *((n, "passed", "") for n in NODES[1:]),
            ),
            [{"test": NODES[0], "outcome": "xfailed", "reason": "#299", "count": 1}],
            True,
            id="declared-xfail-on-its-node",
        ),
        pytest.param(
            _cases(
                (NODES[1], "xfailed", "#299 inert"),
                (NODES[0], "passed", ""),
                (NODES[2], "passed", ""),
            ),
            [{"test": NODES[0], "outcome": "xfailed", "reason": "#299", "count": 1}],
            False,
            id="xfail-swapped-to-another-node",
        ),
        pytest.param(
            _cases(
                (NODES[0], "xfailed", "other reason"),
                *((n, "passed", "") for n in NODES[1:]),
            ),
            [{"test": NODES[0], "outcome": "xfailed", "reason": "#299", "count": 1}],
            False,
            id="xfail-for-another-reason",
        ),
        pytest.param(
            _cases(*((n, "skipped", "holds a mapping") for n in NODES)),
            [
                {
                    "test": XF,
                    "outcome": "skipped",
                    "reason": "holds a mapping",
                    "count": 2,
                }
            ],
            False,
            id="more-skips-than-declared",
        ),
        pytest.param(
            _cases(
                (NODES[0], "skipped", "holds a mapping"),
                *((n, "passed", "") for n in NODES[1:]),
            ),
            [
                {
                    "test": XF,
                    "outcome": "skipped",
                    "reason": "holds a mapping",
                    "count": 2,
                }
            ],
            False,
            id="fewer-skips-than-declared",
        ),
        pytest.param(
            _cases((NODES[0], "failed", "x"), *((n, "passed", "") for n in NODES[1:])),
            [],
            False,
            id="a-failure",
        ),
        pytest.param(
            _passing(NODES[:2]), [], False, id="a-collected-node-not-reported"
        ),
        pytest.param(
            _passing([*NODES[:2], "tests/test_x.py::test_cell[substitute]"]),
            [],
            False,
            id="a-substituted-node-with-the-same-count",
        ),
        pytest.param(
            _passing([NODES[0], NODES[0], NODES[1]]),
            [],
            False,
            id="a-duplicate-report-with-the-same-count",
        ),
    ],
)
def test_p3_accounts_for_every_node(
    tmp_path: pathlib.Path,
    cases: list[Any],
    expected: list[dict[str, Any]],
    ok: bool,
) -> None:
    runner = FakeRunner(tmp_path)
    runner.after_cases = cases
    r = _evaluate(runner, {**copy.deepcopy(REGRESSION), "expected_nonpass": expected})
    assert (r["verdict"] == "COMPLETE") is ok, r["reasons"]


def test_a_skip_in_a_real_run_is_seen(tmp_path: pathlib.Path) -> None:
    """Design review round 1's counterexample: `@pytest.mark.skip` on `assert False`."""
    tree = _tree(
        tmp_path / "t",
        {
            "tests/__init__.py": "",
            "tests/test_x.py": "import pytest\n\n@pytest.mark.skip(reason='later')\n"
            "def test_x():\n    assert False\n",
        },
    )
    rc, _, cases = _real_runner(tmp_path).run_tests(tree, ["tests/test_x.py"])
    assert rc == 0
    assert cases == [gap.Case("tests/test_x.py::test_x", "skipped", "later")]
    assert gap.unexplained(cases, []) != []


def test_a_collection_error_in_a_real_run_is_an_error_case(
    tmp_path: pathlib.Path,
) -> None:
    """Round 3's finding 4: pytest reports it with an empty classname."""
    tree = _tree(
        tmp_path / "t",
        {
            "tests/__init__.py": "",
            "tests/sub/__init__.py": "",
            "tests/sub/test_x.py": "from lizyml_absent import x\n\ndef test_x():\n"
            "    pass\n",
        },
    )
    rc, _, cases = _real_runner(tmp_path).run_tests(tree, ["tests/sub/test_x.py"])
    assert rc != 0
    assert cases == [gap.Case("tests/sub/test_x.py", "error", "collection failure")]
    assert gap.outcomes(cases)["failed"] == 0


def test_an_unmappable_collection_error_is_refused(tmp_path: pathlib.Path) -> None:
    xml = (
        "<testsuites><testsuite><testcase classname='' name='tests.absent'>"
        "<error message='collection failure'/></testcase></testsuite></testsuites>"
    )
    with pytest.raises(gap.ManifestError):
        gap.parse_junit(xml, tmp_path)


def test_junit_cases_carry_node_ids(tmp_path: pathlib.Path) -> None:
    tree = _tree(tmp_path, {"tests/test_k.py": ""})
    xml = (
        "<testsuites><testsuite>"
        "<testcase classname='tests.test_k' name='test_a[1]'/>"
        "<testcase classname='tests.test_k.TestK' name='test_b'><failure message='m'/>"
        "</testcase>"
        "<testcase classname='tests.test_k' name='test_c'><error message='e'/>"
        "</testcase>"
        "<testcase classname='tests.test_k' name='test_d'>"
        "<skipped type='pytest.skip' message='s'/></testcase>"
        "<testcase classname='tests.test_k' name='test_e'>"
        "<skipped type='pytest.xfail' message='x'/></testcase>"
        "</testsuite></testsuites>"
    )
    assert gap.parse_junit(xml, tree) == [
        gap.Case("tests/test_k.py::test_a[1]", "passed", ""),
        gap.Case("tests/test_k.py::TestK::test_b", "failed", "m"),
        gap.Case("tests/test_k.py::test_c", "error", "e"),
        gap.Case("tests/test_k.py::test_d", "skipped", "s"),
        gap.Case("tests/test_k.py::test_e", "xfailed", "x"),
    ]


@pytest.mark.parametrize(
    ("change", "derived", "verdict"),
    [
        pytest.param(
            {"population": 4}, 3, "INCOMPLETE", id="collected-differs-from-declared"
        ),
        pytest.param(
            {"derived_from": "print(4)"}, 4, "INCOMPLETE", id="derived-differs"
        ),
        pytest.param({"derived_from": "print(3)"}, 3, "COMPLETE", id="derived-agrees"),
    ],
)
def test_population_mismatch_is_not_complete(
    tmp_path: pathlib.Path,
    change: dict[str, Any],
    derived: int,
    verdict: str,
) -> None:
    runner = FakeRunner(tmp_path)
    runner.derived = derived
    r = _evaluate(runner, {**copy.deepcopy(REGRESSION), **change})
    assert r["verdict"] == verdict, r["reasons"]


def test_a_derived_only_row_compares_the_collection_with_the_derivation(
    tmp_path: pathlib.Path,
) -> None:
    runner = FakeRunner(tmp_path)
    runner.derived = 5
    row = copy.deepcopy(REGRESSION)
    row.pop("population")
    row["derived_from"] = "print(5)"
    r = _evaluate(runner, row)
    assert r["verdict"] == "INCOMPLETE"
    assert any("collects 3, declared 5" in why for why in r["reasons"])


def test_a_failing_derivation_is_unknown(tmp_path: pathlib.Path) -> None:
    class Broken(FakeRunner):
        def derive(self, tree: pathlib.Path, snippet: str) -> int:
            raise gap.ManifestError("derivation printed 'x', not an int")

    row = {**copy.deepcopy(REGRESSION), "derived_from": "print('x')"}
    results = gap.evaluate({"issues": {"42": row}}, Broken(tmp_path), "HEAD")
    assert results[42]["verdict"] == "UNKNOWN"


def test_note_row_counts_are_reported_as_declared(tmp_path: pathlib.Path) -> None:
    row = copy.deepcopy(REGRESSION)
    row.pop("population_test")
    row["population_note"] = "one node walks the grid"
    row["population"] = 440
    r = _evaluate(FakeRunner(tmp_path), row)
    assert r["verdict"] == "COMPLETE", r["reasons"]
    assert r["p4"].startswith("declared 440")


# --------------------------------------------------------------------------
# Proposition 6
# --------------------------------------------------------------------------


def _comment(runner: FakeRunner) -> dict[str, Any]:
    return runner.issue_data["comments"]["nodes"][0]


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda r: r.issue_data.update(state="OPEN"), id="issue-open"),
        pytest.param(
            lambda r: r.issue_data.update(stateReason="NOT_PLANNED"), id="wrong-reason"
        ),
        pytest.param(lambda r: r.prs[500].update(state="OPEN"), id="pr-not-merged"),
        pytest.param(lambda r: r.ancestors.clear(), id="merge-not-an-ancestor"),
        pytest.param(
            lambda r: r.prs[500].update(body="Unrelated work"), id="body-does-not-cite"
        ),
        pytest.param(
            lambda r: r.prs[500].update(body="Fixes #420"),
            id="body-cites-a-superstring",
        ),
        pytest.param(
            lambda r: r.prs[500].update(mergedAt="2026-10-01T13:00:00Z"),
            id="comment-predates-the-merge",
        ),
        pytest.param(
            lambda r: _comment(r).update(createdAt="2026-10-03T00:00:00Z"),
            id="comment-after-the-close",
        ),
        pytest.param(
            lambda r: _comment(r).update(databaseId=1), id="pinned-comment-absent"
        ),
        pytest.param(
            lambda r: _comment(r).update(body="Closing: no longer relevant."),
            id="pinned-comment-names-no-pr",
        ),
        pytest.param(
            lambda r: setattr(r, "comment_issue", 43),
            id="pinned-comment-on-another-issue",
        ),
    ],
)
def test_p6_requires_every_condition(tmp_path: pathlib.Path, mutate: Any) -> None:
    runner = FakeRunner(tmp_path)
    mutate(runner)
    r = _evaluate(runner, copy.deepcopy(REGRESSION))
    assert r["verdict"] == "INCOMPLETE"
    assert any(why.startswith("p6:") for why in r["reasons"]), r["reasons"]


def test_p6_reads_the_pinned_comment_not_the_last_one(tmp_path: pathlib.Path) -> None:
    """Round 2: a fixing comment, then an acknowledgement, is a correct record."""
    runner = FakeRunner(tmp_path)
    runner.issue_data["comments"]["nodes"].append(
        {
            "databaseId": 9002,
            "createdAt": "2026-10-01T18:00:00Z",
            "body": "Thanks for confirming.",
        }
    )
    r = _evaluate(runner, copy.deepcopy(REGRESSION))
    assert r["verdict"] == "COMPLETE", r["reasons"]


def test_p6_accepts_the_pr_named_by_its_exact_title(tmp_path: pathlib.Path) -> None:
    runner = FakeRunner(tmp_path)
    _comment(runner)["body"] = (
        "Completed by the merged PR titled `fix(core): repair the thing properly`."
    )
    r = _evaluate(runner, copy.deepcopy(REGRESSION))
    assert r["verdict"] == "COMPLETE", r["reasons"]


@pytest.mark.parametrize(
    ("text", "number", "cited"),
    [
        pytest.param("Fixes #263", 263, True, id="exact"),
        pytest.param("Fixes #263", 26, False, id="prefix"),
        pytest.param("Fixes #1263", 263, False, id="suffix"),
        pytest.param("see nbx-liz/other#263", 263, False, id="other-repository"),
        pytest.param("(#263).", 263, True, id="punctuated"),
    ],
)
def test_citations_are_digit_bounded(text: str, number: int, cited: bool) -> None:
    assert gap.cites(text, number) is cited


@pytest.mark.parametrize(
    ("comment", "title", "named"),
    [
        pytest.param(
            "the PR titled fix(core): repair the thing",
            "fix(core): repair the thing (#500)",
            True,
            id="squash-suffix-dropped",
        ),
        pytest.param(
            "the PR titled fix(core): repair",
            "fix(core): repair the thing",
            False,
            id="partial-title",
        ),
        pytest.param("a docs fix", "docs: fix", False, id="title-too-short"),
    ],
)
def test_titles_must_match_exactly(comment: str, title: str, named: bool) -> None:
    assert gap.names_pr(comment, 500, title) is named


# --------------------------------------------------------------------------
# Rows outside `regression`
# --------------------------------------------------------------------------


def test_a_partial_row_needs_its_issue_open(tmp_path: pathlib.Path) -> None:
    runner = FakeRunner(tmp_path)
    row = {
        **copy.deepcopy(REGRESSION),
        "disposition": "partial",
        "justification": "2/179",
    }
    assert _evaluate(runner, row)["verdict"] == "INCOMPLETE"
    runner.issue_data["state"] = "OPEN"
    assert _evaluate(runner, row)["verdict"] == "PARTIAL"
    assert runner.worktrees == []


def test_a_not_planned_row(tmp_path: pathlib.Path) -> None:
    runner = FakeRunner(tmp_path)
    row = {
        "github_prs": [],
        "disposition": "not-planned",
        "justification": "x",
        "tests": [],
    }
    assert _evaluate(runner, row)["verdict"] == "INCOMPLETE"
    runner.issue_data["stateReason"] = "NOT_PLANNED"
    assert _evaluate(runner, row)["verdict"] == "NOT-PLANNED"


def test_a_decision_only_row_skips_p2(tmp_path: pathlib.Path) -> None:
    runner = FakeRunner(tmp_path)
    runner.before_cases = _passing()
    row = {
        **copy.deepcopy(REGRESSION),
        "disposition": "decision-only",
        "justification": "x",
    }
    assert _evaluate(runner, row)["verdict"] == "COMPLETE"
    assert runner.worktrees == []


def test_a_regression_row_without_a_fixing_pr_is_incomplete(
    tmp_path: pathlib.Path,
) -> None:
    row = {**copy.deepcopy(REGRESSION), "github_prs": []}
    r = _evaluate(FakeRunner(tmp_path), row)
    assert r["verdict"] == "INCOMPLETE"
    assert "no fixing PR recorded" in r["reasons"]


def test_a_missing_test_file_is_incomplete(tmp_path: pathlib.Path) -> None:
    row = {**copy.deepcopy(REGRESSION), "tests": ["tests/test_not_written.py"]}
    r = _evaluate(FakeRunner(tmp_path), row)
    assert r["verdict"] == "INCOMPLETE"
    assert r["reasons"][0].startswith("p1:")


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def test_runner_prepares_each_worktree(tmp_path: pathlib.Path) -> None:
    repo = _tree(tmp_path / "repo", {"lizyml/_version.py": "__version__ = 'x'\n"})
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "python").symlink_to(sys.executable)
    runner = gap.Runner(repo, venv / "python", tmp_path / "scratch")
    calls: list[tuple[list[str], pathlib.Path, dict[str, str]]] = []

    def sh(cmd: list[str], cwd: pathlib.Path) -> tuple[int, str]:
        calls.append((cmd, cwd, runner.env()))
        if cmd[:3] == ["git", "worktree", "add"]:
            (pathlib.Path(cmd[4]) / "lizyml").mkdir(parents=True)
        return 0, (AFTER_SHA + "\n") if cmd[:2] == ["git", "rev-parse"] else ""

    runner.sh = sh  # type: ignore[method-assign]
    tree = runner.worktree(AFTER_SHA)
    assert (tree / "lizyml" / "_version.py").read_text() == "__version__ = 'x'\n"
    assert runner.python == str(venv / "python")
    runner.python_c(tree, "print(1)")
    assert calls[-1][0] == [str(venv / "python"), "-c", "print(1)"]
    assert calls[-1][1] == tree
    assert calls[-1][2]["PYTHONDONTWRITEBYTECODE"] == "1"


@pytest.mark.parametrize(
    ("rc", "out", "expected"),
    [
        pytest.param(
            0,
            json.dumps(
                {
                    "issue_url": "https://api.github.com/repos/o/r/issues/277",
                    "created_at": "2026-09-15T08:31:25Z",
                    "body": "Fixed by #296.",
                }
            ),
            {
                "issue": 277,
                "createdAt": "2026-09-15T08:31:25Z",
                "body": "Fixed by #296.",
            },
            id="found",
        ),
        pytest.param(1, "gh: Not Found (HTTP 404)", None, id="absent"),
    ],
)
def test_runner_reads_a_comment_by_id(
    tmp_path: pathlib.Path, rc: int, out: str, expected: Any
) -> None:
    """Round 7: read directly, so a comment past the hundredth is still found."""
    runner = gap.Runner(tmp_path, pathlib.Path(sys.executable), tmp_path / "scratch")
    calls: list[list[str]] = []

    def sh(cmd: list[str], cwd: pathlib.Path) -> tuple[int, str]:
        calls.append(cmd)
        return rc, out

    runner.sh = sh  # type: ignore[method-assign]
    assert runner.comment(5677178764) == expected
    assert calls == [["gh", "api", "repos/nbx-liz/LizyML/issues/comments/5677178764"]]


def test_runner_refuses_a_worktree_at_the_wrong_commit(tmp_path: pathlib.Path) -> None:
    runner = gap.Runner(tmp_path, pathlib.Path(sys.executable), tmp_path / "scratch")
    (tmp_path / "scratch" / AFTER_SHA).mkdir(parents=True)
    runner.sh = lambda cmd, cwd: (0, "b" * 40 + "\n")  # type: ignore[method-assign]
    with pytest.raises(gap.ManifestError, match="not " + AFTER_SHA):
        runner.worktree(AFTER_SHA)


# --------------------------------------------------------------------------
# Verdict arithmetic
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("verdicts", "code"),
    [
        pytest.param(
            ["COMPLETE", "PARTIAL", "NOT-PLANNED", "COMPLETE-RED-BY-MUTATION"],
            0,
            id="nothing-against",
        ),
        pytest.param(["COMPLETE", "INCOMPLETE"], 1, id="an-incomplete"),
        pytest.param(["COMPLETE", "UNKNOWN"], 1, id="an-unknown"),
    ],
)
def test_exit_code_counts_incomplete_and_unknown(
    verdicts: list[str], code: int
) -> None:
    results = {
        i: {"verdict": v, "disposition": "regression", "reasons": []}
        for i, v in enumerate(verdicts)
    }
    assert gap.report(results, out=io.StringIO()) == code


def test_the_summary_keeps_each_verdict_separate() -> None:
    results = {
        i: {"verdict": v, "disposition": "regression", "reasons": []}
        for i, v in enumerate(
            [
                "COMPLETE",
                "COMPLETE-RED-BY-MUTATION",
                "PARTIAL",
                "NOT-PLANNED",
                "INCOMPLETE",
            ]
        )
    }
    buf = io.StringIO()
    gap.report(results, out=buf)
    last = [ln for ln in buf.getvalue().splitlines() if ln.startswith("COMPLETE ")][0]
    assert last.startswith(
        "COMPLETE 1   COMPLETE-RED-BY-MUTATION 1   PARTIAL 1   "
        "NOT-PLANNED 1   INCOMPLETE 1"
    )
    assert "of 5" in last


@pytest.mark.parametrize(
    "text", ["[]", '"issues"', "3"], ids=["list", "string", "number"]
)
def test_a_manifest_that_is_not_an_object_is_refused(
    tmp_path: pathlib.Path, text: str
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(text)
    with pytest.raises(gap.ManifestError):
        gap.load_manifest(path)


def test_the_shipped_manifest_is_valid_json_with_every_row_validated() -> None:
    data = json.loads(gap.MANIFEST.read_text(encoding="utf-8"))
    for num, row in data["issues"].items():
        gap.validate_row(num, row)
