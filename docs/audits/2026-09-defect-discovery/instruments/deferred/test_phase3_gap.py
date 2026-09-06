"""Unit tests for the completion tool's quiet failure modes.

Not the worktree plumbing -- that is what the tool does when run. What these
cover is everything that could report a wrong number without erroring: the
manifest grammar, the node-id counting, and the verdict arithmetic. A fake
`Runner` drives `evaluate` end to end without git or pytest, so the propositions
are exercised rather than described.

Place at: tests/test_docs/test_phase3_gap.py
(`pyproject.toml` restricts pytest discovery to `tests`, so it does not run
from the audit archive.)
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

AUDIT = pathlib.Path(__file__).resolve().parents[2] / "docs" / "audits" / "2026-09-defect-discovery"
if AUDIT.exists():  # in-repo location
    sys.path.insert(0, str(AUDIT))
else:  # running from the discovery working set
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import phase3_gap as gap  # noqa: E402

GOOD = {
    "pr": 1,
    "tests": ["tests/t.py"],
    "population": 3,
    "derived_from": "len([1, 2, 3])",
    "disposition": "regression",
}


def _manifest(issues: dict) -> dict:
    return {"head_before": "abc1234", "issues": issues}


def _write(tmp_path: pathlib.Path, issues: dict) -> pathlib.Path:
    p = tmp_path / "m.json"
    p.write_text(json.dumps(_manifest(issues)))
    return p


# --------------------------------------------------------------------------
# Manifest grammar
# --------------------------------------------------------------------------


def test_the_shipped_manifest_parses() -> None:
    data = gap.load_manifest()
    assert len(data["issues"]) == 15, "the manifest must cover every filed issue"


@pytest.mark.parametrize(
    "mutate,fragment",
    [
        (lambda r: r.pop("population"), "missing"),
        (lambda r: r.update(population=0), "non-positive"),
        (lambda r: r.update(tests=[]), "names no test"),
        (lambda r: r.update(derived_from="  "), "no derivation"),
        (lambda r: r.update(disposition="probably-fine"), "disposition"),
        (lambda r: r.update(disposition="decision-only"), "justification"),
    ],
)
def test_a_malformed_manifest_row_raises(tmp_path, mutate, fragment) -> None:
    row = dict(GOOD)
    mutate(row)
    with pytest.raises(gap.ManifestError) as exc:
        gap.load_manifest(_write(tmp_path, {"258": row}))
    assert fragment in str(exc.value)


def test_a_non_numeric_issue_key_raises(tmp_path) -> None:
    with pytest.raises(gap.ManifestError):
        gap.load_manifest(_write(tmp_path, {"two-fifty-eight": dict(GOOD)}))


# --------------------------------------------------------------------------
# Node-id counting
# --------------------------------------------------------------------------


def test_node_ids_are_counted_exactly() -> None:
    out = (
        "tests/t.py::test_a\n"
        "tests/t.py::test_b[x]\n"
        "tests/t.py::TestC::test_d[y-z]\n"
        "\n3 tests collected in 0.01s\n"
    )
    assert len(gap.parse_node_ids(out)) == 3


def test_duplicate_node_ids_raise_rather_than_inflate_the_count() -> None:
    with pytest.raises(gap.ManifestError, match="duplicate"):
        gap.parse_node_ids("tests/t.py::test_a\ntests/t.py::test_a\n2 tests collected\n")


def test_an_unparseable_collected_line_raises_rather_than_being_skipped() -> None:
    """The grammar is closed: a line with `::` that is not a node id is a
    failure, not something to step over. That is DC1's whole shape."""
    with pytest.raises(gap.ManifestError, match="unparseable"):
        gap.parse_node_ids("tests/t.py::test_a\nsome::weird output line\n")


# --------------------------------------------------------------------------
# Verdict arithmetic, driven through `evaluate` with a fake Runner
# --------------------------------------------------------------------------


class FakeRunner(gap.Runner):
    def __init__(self, *, before_rc=1, after_rc=0, collected=3, derived=3,
                 state="CLOSED", worktree_sha_ok=True) -> None:
        super().__init__("python")
        self.before_rc, self.after_rc = before_rc, after_rc
        self.collected, self.derived, self.state = collected, derived, state
        self.worktree_sha_ok = worktree_sha_ok
        self.before_runs = 0

    def worktree(self, repo, sha, dest):
        if not self.worktree_sha_ok:
            raise gap.ManifestError(f"worktree {dest} is at deadbeef, not {sha}")
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "tests").mkdir(exist_ok=True)
        (dest / "tests" / "t.py").write_text("")
        return dest

    def pytest(self, tree, args):
        if "--collect-only" in args:
            return 0, "".join(f"tests/t.py::test_{i}\n" for i in range(self.collected))
        if tree.name == "before":
            self.before_runs += 1
            return self.before_rc, ""
        return self.after_rc, ""

    def evaluate_expression(self, tree, expr):
        return self.derived

    def issue_state(self, repo, num):
        return {"state": self.state, "closedByPullRequestsReferences": [{"number": 9}]}


def _run(tmp_path, runner, row_overrides=None) -> dict:
    row = dict(GOOD)
    row.update(row_overrides or {})
    return gap.evaluate(tmp_path, "aaaa111", "bbbb222", tmp_path / "s", runner,
                        _manifest({"258": row}))


def test_a_fully_satisfied_issue_is_complete(tmp_path) -> None:
    r = _run(tmp_path, FakeRunner())["258"]
    assert r["verdict"] == "COMPLETE"


def test_the_before_head_is_actually_run(tmp_path) -> None:
    """p2 must execute, not be inferred from the test's absence."""
    runner = FakeRunner()
    _run(tmp_path, runner)
    assert runner.before_runs == 1


def test_a_test_that_passes_before_is_incomplete(tmp_path) -> None:
    r = _run(tmp_path, FakeRunner(before_rc=0))["258"]
    assert r["verdict"] == "INCOMPLETE" and r["p2_fails_before"] is False


def test_a_larger_collection_than_declared_is_incomplete(tmp_path) -> None:
    """`>=` would pass this. The population is an equality, not a floor."""
    r = _run(tmp_path, FakeRunner(collected=5))["258"]
    assert r["verdict"] == "INCOMPLETE" and r["p4_matches"] is False


def test_a_population_the_source_does_not_produce_is_incomplete(tmp_path) -> None:
    r = _run(tmp_path, FakeRunner(derived=4))["258"]
    assert r["verdict"] == "INCOMPLETE" and r["p5_matches"] is False


def test_an_open_issue_is_incomplete(tmp_path) -> None:
    r = _run(tmp_path, FakeRunner(state="OPEN"))["258"]
    assert r["verdict"] == "INCOMPLETE" and r["p6_ok"] is False


def test_a_partial_issue_must_stay_open(tmp_path) -> None:
    over = {"disposition": "partial", "justification": "2 of 3 repaired", "repaired": 2}
    assert _run(tmp_path, FakeRunner(state="OPEN"), over)["258"]["verdict"] == "PARTIAL"
    assert _run(tmp_path, FakeRunner(state="CLOSED"), over)["258"]["verdict"] == "INCOMPLETE"


def test_a_stale_worktree_is_refused_rather_than_measured(tmp_path) -> None:
    with pytest.raises(gap.ManifestError, match="not"):
        _run(tmp_path, FakeRunner(worktree_sha_ok=False))


def test_an_unreadable_issue_state_is_unknown_not_a_pass(tmp_path) -> None:
    class NoGh(FakeRunner):
        def issue_state(self, repo, num):
            raise gap.ManifestError("cannot read issue state")

    r = _run(tmp_path, NoGh())["258"]
    assert r["verdict"] == "UNKNOWN"


def test_report_exits_non_zero_on_incomplete() -> None:
    """The predecessor of this tool exited 0 while work was outstanding."""
    assert gap.report({"1": {"verdict": "INCOMPLETE"}}) == 1
    assert gap.report({"1": {"verdict": "UNKNOWN"}}) == 1
    assert gap.report({"1": {"verdict": "COMPLETE"}}) == 0
    assert gap.report({"1": {"verdict": "PARTIAL"}}) == 0
