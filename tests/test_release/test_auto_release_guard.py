"""``auto-release.yml`` tags only a merge commit of develop (H-0117).

The workflow's decisions live in three shell scripts under ``.github/scripts/``:
the merge guard, the title-to-version reader and the idempotent tag step. Each is
executed here against a real repository, not a fixture of its output, and the
workflow file is read to check the step order and every value each step reads.
"""

from __future__ import annotations

import os
import pathlib
import subprocess

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".github" / "scripts"
GUARD = SCRIPTS / "check_release_merge.sh"
VERSION = SCRIPTS / "release_version.sh"
TAGGER = SCRIPTS / "create_release_tag.sh"
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


def _commit(repo: pathlib.Path, name: str) -> None:
    (repo / f"{name}.txt").write_text(f"{name}\n")
    _git(repo, "add", f"{name}.txt")
    _git(repo, "commit", "-q", "-m", name)


def _run(
    script: pathlib.Path, cwd: pathlib.Path, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script)],
        cwd=cwd,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# The merge guard
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def history(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """A repository with a two-parent merge, a single-parent commit and an octopus."""
    repo = tmp_path_factory.mktemp("release-guard")
    _git(repo, "init", "-q", "-b", "base")
    _commit(repo, "a")
    _git(repo, "checkout", "-q", "-b", "feature")
    _commit(repo, "b")
    single = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "base")
    _git(repo, "merge", "-q", "--no-ff", "-m", "merge", "feature")
    merge = _git(repo, "rev-parse", "HEAD")
    for branch in ("one", "two"):
        _git(repo, "checkout", "-q", "-b", branch, "base")
        _commit(repo, branch)
    _git(repo, "checkout", "-q", "base")
    _git(repo, "merge", "-q", "--no-ff", "-m", "octopus", "one", "two")
    octopus = _git(repo, "rev-parse", "HEAD")
    assert len(_git(repo, "rev-list", "--parents", "-n", "1", octopus).split()) == 4
    return {"repo": repo, "MERGE": merge, "SINGLE": single, "OCTOPUS": octopus}


def _guard(history: dict[str, object], **env: str) -> subprocess.CompletedProcess[str]:
    resolved = {k: str(history.get(v, v)) for k, v in env.items()}
    base = {"HEAD_REF": "develop", "HEAD_REPO": REPO, "BASE_REPO": REPO}
    repo = history["repo"]
    assert isinstance(repo, pathlib.Path)
    return _run(GUARD, repo, {**base, **resolved})


def test_a_merge_commit_of_develop_passes(history: dict[str, object]) -> None:
    result = _guard(history, MERGE_SHA="MERGE")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("case", "env"),
    [
        ("single parent (squash or rebase)", {"MERGE_SHA": "SINGLE"}),
        ("three parents (octopus)", {"MERGE_SHA": "OCTOPUS"}),
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
    history: dict[str, object], case: str, env: dict[str, str]
) -> None:
    result = _guard(history, **env)
    assert result.returncode != 0, f"{case}: the guard passed"
    assert "::error::" in result.stdout, f"{case}: no error annotation"


# ---------------------------------------------------------------------------
# The version comes from an exact title
# ---------------------------------------------------------------------------


def _version(tmp_path: pathlib.Path, title: str) -> tuple[int, str]:
    output = tmp_path / "github_output"
    output.write_text("")
    result = _run(VERSION, tmp_path, {"TITLE": title, "GITHUB_OUTPUT": str(output)})
    return result.returncode, output.read_text()


def test_the_canonical_title_yields_its_version(tmp_path: pathlib.Path) -> None:
    code, output = _version(tmp_path, "release: v0.18.0")
    assert code == 0
    assert output.splitlines() == ["tag=v0.18.0", "version=0.18.0"]


@pytest.mark.parametrize(
    "title",
    [
        "release: v0.18.0-rc1",
        "release: v0.18.0 v9.9.9",
        "release: v0.18.0\nv9.9.9",
        "release: v0.18.0 — LizyML 0.18.0",
        "release: 0.18.0",
        "Release: v0.18.0",
        " release: v0.18.0",
        "release:",
        "",
    ],
)
def test_any_other_title_is_refused_and_writes_nothing(
    tmp_path: pathlib.Path, title: str
) -> None:
    code, output = _version(tmp_path, title)
    assert code != 0, f"{title!r} was accepted"
    assert output == "", f"{title!r} wrote step outputs: {output!r}"


# ---------------------------------------------------------------------------
# The tag step is idempotent on the merge commit and refuses any other target
# ---------------------------------------------------------------------------


@pytest.fixture
def checkout(tmp_path: pathlib.Path) -> dict[str, object]:
    """A checkout with an ``origin`` bare remote, a merge commit and another commit."""
    remote = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", str(remote))
    repo = tmp_path / "work"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "base")
    _git(repo, "remote", "add", "origin", str(remote))
    _commit(repo, "a")
    other = _git(repo, "rev-parse", "HEAD")
    _commit(repo, "b")
    merge = _git(repo, "rev-parse", "HEAD")
    return {"repo": repo, "remote": remote, "MERGE": merge, "OTHER": other}


def _tag(
    checkout: dict[str, object], merge_sha: str
) -> subprocess.CompletedProcess[str]:
    repo = checkout["repo"]
    assert isinstance(repo, pathlib.Path)
    return _run(TAGGER, repo, {"TAG": "v0.18.0", "MERGE_SHA": merge_sha})


def _tag_target(where: object) -> str:
    assert isinstance(where, pathlib.Path)
    return subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", "refs/tags/v0.18.0^{commit}"],
        cwd=where,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_a_fresh_tag_lands_on_the_merge_commit(checkout: dict[str, object]) -> None:
    result = _tag(checkout, str(checkout["MERGE"]))
    assert result.returncode == 0, result.stdout + result.stderr
    assert _tag_target(checkout["repo"]) == checkout["MERGE"]
    assert _tag_target(checkout["remote"]) == checkout["MERGE"]


def test_a_rerun_reuses_the_tag_it_already_made(checkout: dict[str, object]) -> None:
    assert _tag(checkout, str(checkout["MERGE"])).returncode == 0
    again = _tag(checkout, str(checkout["MERGE"]))
    assert again.returncode == 0, again.stdout + again.stderr
    assert _tag_target(checkout["remote"]) == checkout["MERGE"]


def test_a_tag_made_but_not_pushed_is_pushed_on_rerun(
    checkout: dict[str, object],
) -> None:
    repo = checkout["repo"]
    assert isinstance(repo, pathlib.Path)
    _git(repo, "tag", "v0.18.0", str(checkout["MERGE"]))
    result = _tag(checkout, str(checkout["MERGE"]))
    assert result.returncode == 0, result.stdout + result.stderr
    assert _tag_target(checkout["remote"]) == checkout["MERGE"]


def test_a_tag_on_another_commit_is_refused(checkout: dict[str, object]) -> None:
    repo = checkout["repo"]
    assert isinstance(repo, pathlib.Path)
    _git(repo, "tag", "v0.18.0", str(checkout["OTHER"]))
    result = _tag(checkout, str(checkout["MERGE"]))
    assert result.returncode != 0
    assert "::error::" in result.stdout
    assert _tag_target(repo) == checkout["OTHER"]
    assert _tag_target(checkout["remote"]) == ""


def test_a_merge_commit_missing_from_the_checkout_is_refused(
    checkout: dict[str, object],
) -> None:
    result = _tag(checkout, "f" * 40)
    assert result.returncode != 0
    assert _tag_target(checkout["repo"]) == ""


# ---------------------------------------------------------------------------
# The workflow wires the scripts in order and feeds each step exactly its inputs
# ---------------------------------------------------------------------------


def _steps() -> list[dict[str, object]]:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps: list[dict[str, object]] = workflow["jobs"]["tag-and-release"]["steps"]
    return steps


def test_the_steps_run_the_scripts_in_order() -> None:
    by_name = {str(s.get("name")): s for s in _steps()}
    names = list(by_name)
    order = [
        "Verify the release is a merge commit of develop",
        "Extract version from PR title",
        "Create tag",
        "Create GitHub Release",
        "Trigger PyPI publish workflow",
    ]
    assert [names.index(n) for n in order] == sorted(names.index(n) for n in order)
    assert by_name[order[0]]["run"] == f"bash .github/scripts/{GUARD.name}"
    assert by_name[order[1]]["run"] == f"bash .github/scripts/{VERSION.name}"
    assert by_name[order[2]]["run"] == f"bash .github/scripts/{TAGGER.name}"


#: Every step's whole ``env`` mapping, by step name. A step not listed must have
#: none. Pinning only some inputs let a miswiring through twice in review: a
#: ``TITLE`` read from the PR body would tag whatever version the body names.
EXPECTED_ENV: dict[str, dict[str, str]] = {
    "Verify the release is a merge commit of develop": {
        "HEAD_REF": "${{ github.event.pull_request.head.ref }}",
        "HEAD_REPO": "${{ github.event.pull_request.head.repo.full_name }}",
        "BASE_REPO": "${{ github.repository }}",
        "MERGE_SHA": "${{ github.event.pull_request.merge_commit_sha }}",
    },
    "Extract version from PR title": {
        "TITLE": "${{ github.event.pull_request.title }}",
    },
    "Extract release notes from CHANGELOG.md": {
        "VERSION": "${{ steps.version.outputs.version }}",
    },
    "Create tag": {
        "TAG": "${{ steps.version.outputs.tag }}",
        "MERGE_SHA": "${{ github.event.pull_request.merge_commit_sha }}",
    },
    "Create GitHub Release": {
        "GH_TOKEN": "${{ secrets.GITHUB_TOKEN }}",
        "TAG": "${{ steps.version.outputs.tag }}",
    },
    "Trigger PyPI publish workflow": {
        "GH_TOKEN": "${{ secrets.GITHUB_TOKEN }}",
        "TAG": "${{ steps.version.outputs.tag }}",
    },
}


def test_every_step_reads_exactly_the_inputs_it_should() -> None:
    actual = {str(s.get("name")): s.get("env") for s in _steps()}
    assert {name: env for name, env in actual.items() if env} == EXPECTED_ENV


def test_no_step_expands_an_expression_inside_its_script() -> None:
    for step in _steps():
        assert "${{" not in str(step.get("run", "")), step.get("name")
