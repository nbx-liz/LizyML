"""``auto-release.yml`` tags only a merge commit of develop (H-0117).

The guard is a shell script the workflow runs before ``Create tag``. It is
executed here against a real repository, not a fixture of its output, and the
workflow file is read to check where the guard sits and how values reach it.
"""

from __future__ import annotations

import os
import pathlib
import subprocess

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
GUARD = ROOT / ".github" / "scripts" / "check_release_merge.sh"
WORKFLOW = ROOT / ".github" / "workflows" / "auto-release.yml"
REPO = "nbx-liz/LizyML"


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture(scope="module")
def history(tmp_path_factory: pytest.TempPathFactory) -> tuple[pathlib.Path, str, str]:
    """A repository with one two-parent merge and one single-parent commit."""
    repo = tmp_path_factory.mktemp("release-guard")
    _git(repo, "init", "-q", "-b", "base")
    (repo / "a.txt").write_text("a\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "base")
    _git(repo, "checkout", "-q", "-b", "feature")
    (repo / "b.txt").write_text("b\n")
    _git(repo, "add", "b.txt")
    _git(repo, "commit", "-q", "-m", "feature")
    single = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "base")
    _git(repo, "merge", "-q", "--no-ff", "-m", "merge", "feature")
    merge = _git(repo, "rev-parse", "HEAD")
    return repo, merge, single


def _guard(repo: pathlib.Path, **env: str) -> subprocess.CompletedProcess[str]:
    full = {
        **os.environ,
        "HEAD_REF": "develop",
        "HEAD_REPO": REPO,
        "BASE_REPO": REPO,
        **env,
    }
    return subprocess.run(
        ["bash", str(GUARD)], cwd=repo, env=full, capture_output=True, text=True
    )


def test_a_merge_commit_of_develop_passes(
    history: tuple[pathlib.Path, str, str],
) -> None:
    repo, merge, _ = history
    result = _guard(repo, MERGE_SHA=merge)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("case", "env"),
    [
        ("single parent (squash or rebase)", {"MERGE_SHA": "SINGLE"}),
        (
            "head is a release branch",
            {"MERGE_SHA": "MERGE", "HEAD_REF": "release/v0.18.0"},
        ),
        ("head is a fork", {"MERGE_SHA": "MERGE", "HEAD_REPO": "someone/LizyML"}),
        ("merge commit not in the checkout", {"MERGE_SHA": "f" * 40}),
        ("merge commit missing", {"MERGE_SHA": ""}),
    ],
)
def test_anything_else_is_refused(
    history: tuple[pathlib.Path, str, str], case: str, env: dict[str, str]
) -> None:
    repo, merge, single = history
    resolved = {k: {"MERGE": merge, "SINGLE": single}.get(v, v) for k, v in env.items()}
    result = _guard(repo, **resolved)
    assert result.returncode != 0, f"{case}: the guard passed"
    assert "::error::" in result.stdout, f"{case}: no error annotation"


def _steps() -> list[dict[str, object]]:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps: list[dict[str, object]] = workflow["jobs"]["tag-and-release"]["steps"]
    return steps


def test_the_guard_runs_before_the_tag_and_the_tag_is_the_merge_commit() -> None:
    steps = _steps()
    names = [str(s.get("name", "")) for s in steps]
    guard_at = next(
        i for i, s in enumerate(steps) if GUARD.name in str(s.get("run", ""))
    )
    tag_at = names.index("Create tag")
    assert guard_at < tag_at
    guard_env = steps[guard_at]["env"]
    assert isinstance(guard_env, dict)
    assert guard_env["HEAD_REF"] == "${{ github.event.pull_request.head.ref }}"
    assert guard_env["MERGE_SHA"] == "${{ github.event.pull_request.merge_commit_sha }}"
    tag_step = steps[tag_at]
    assert "$MERGE_SHA" in str(tag_step["run"])
    assert isinstance(tag_step.get("env"), dict)
    assert (
        tag_step["env"]["MERGE_SHA"]
        == "${{ github.event.pull_request.merge_commit_sha }}"
    )


def test_no_step_expands_pull_request_fields_inside_its_script() -> None:
    for step in _steps():
        assert "${{ github.event.pull_request" not in str(step.get("run", "")), (
            step.get("name")
        )
