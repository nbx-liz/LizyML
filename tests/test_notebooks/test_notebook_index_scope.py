"""The scope decision for the notebook index jobs (H-0119 section 6).

``.github/scripts/notebook_index_scope.sh`` prints ``run=true`` or
``run=false`` for the ``notebook-index-scope`` job. These tests run it against
a temporary Git repository, so the paths Git prints are real: a non-ASCII or
quote-containing path must still count (review run 4, round 1).
"""

from __future__ import annotations

import os
import pathlib
import subprocess

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github" / "scripts" / "notebook_index_scope.sh"
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def _git(repo: pathlib.Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repo(tmp_path: pathlib.Path, paths: list[str]) -> tuple[pathlib.Path, str, str]:
    """A repository whose head commit adds ``paths`` on top of an empty base."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")
    for name in paths:
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "change")
    return repo, base, _git(repo, "rev-parse", "HEAD")


def _scope(
    repo: pathlib.Path,
    base: str,
    head: str,
    event: str = "pull_request",
    ref: str = "develop",
) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": os.environ["PATH"],
        "EVENT": event,
        "BASE_REF": ref,
        "BASE_SHA": base,
        "HEAD_SHA": head,
    }
    return subprocess.run(
        ["bash", str(SCRIPT)], cwd=repo, env=env, capture_output=True, text=True
    )


INDEX_PATHS = [
    "notebooks/tutorial_x.ipynb",
    "notebooks/café.ipynb",
    'notebooks/a"b.ipynb',
    "notebooks/a\\b.ipynb",
    "notebooks/a\nb.ipynb",
    "docs/examples.md",
    "lizyml/_extras.py",
    "scripts/examples_index.py",
    "tests/test_notebooks/test_x.py",
    ".github/workflows/ci.yml",
]


@pytest.mark.parametrize("path", INDEX_PATHS)
def test_a_changed_index_path_runs_the_jobs(tmp_path: pathlib.Path, path: str) -> None:
    repo, base, head = _repo(tmp_path, ["README.md", path])
    result = _scope(repo, base, head)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "run=true\n"


@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "docs/examples.md.bak",
        "xnotebooks/a.ipynb",
        "docs/api.md",
        # One unrelated path whose name holds a newline: each NUL-delimited
        # record is matched whole, so no part of it starts a record (review
        # run 4, round 2).
        "other/a\nnotebooks/fake.ipynb",
        "other/a\ndocs/examples.md",
    ],
)
def test_no_index_path_skips_the_jobs(tmp_path: pathlib.Path, path: str) -> None:
    repo, base, head = _repo(tmp_path, [path])
    result = _scope(repo, base, head)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "run=false\n"


def test_a_large_diff_with_an_index_path_runs_the_jobs(tmp_path: pathlib.Path) -> None:
    # More than a pipe buffer of names before the match: an early-exiting
    # matcher on a pipe must not turn the match into a failure.
    names = [f"other/{'n' * 60}_{i:05d}.txt" for i in range(2000)]
    repo, base, head = _repo(tmp_path, [*names, "notebooks/zz.ipynb"])
    result = _scope(repo, base, head)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "run=true\n"


@pytest.mark.parametrize(
    ("event", "ref"),
    [("push", ""), ("pull_request", "main"), ("workflow_dispatch", "")],
)
def test_anything_but_a_pr_to_develop_runs_the_jobs(
    tmp_path: pathlib.Path, event: str, ref: str
) -> None:
    repo, base, head = _repo(tmp_path, ["README.md"])
    result = _scope(repo, base, head, event=event, ref=ref)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "run=true\n"


def test_a_failing_diff_fails_instead_of_skipping(tmp_path: pathlib.Path) -> None:
    repo, base, _ = _repo(tmp_path, ["notebooks/a.ipynb"])
    result = _scope(repo, base, "0" * 40)
    assert result.returncode != 0
    assert "run=" not in result.stdout


def test_the_scope_step_runs_the_script() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["notebook-index-scope"]["steps"]
    (step,) = [s for s in steps if s.get("id") == "scope"]
    assert step["run"].strip() == (
        'bash .github/scripts/notebook_index_scope.sh >> "$GITHUB_OUTPUT"'
    )
    assert step["env"] == {
        "EVENT": "${{ github.event_name }}",
        "BASE_REF": "${{ github.base_ref }}",
        "BASE_SHA": "${{ github.event.pull_request.base.sha }}",
        "HEAD_SHA": "${{ github.event.pull_request.head.sha }}",
    }
