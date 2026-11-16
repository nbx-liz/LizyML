"""Every workflow follows the H-0146 supply-chain rules (acceptance 3).

zizmor (the ``Workflow lint`` job) audits pins, credentials, permissions and
cache use as well; these tests pin the rules this repository decided, so a
zizmor version that stops flagging one of them does not let it back in.
"""

from __future__ import annotations

import pathlib
import re
from typing import Any

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
PINNED = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._/-]+@[0-9a-f]{40} # v\d+(\.\d+)*$")


def _load(path: pathlib.Path) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data


def _uses_lines(path: pathlib.Path) -> list[str]:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip().removeprefix("- ")
        if stripped.startswith("uses:"):
            lines.append(stripped.removeprefix("uses:").strip())
    return lines


def _steps(path: pathlib.Path) -> list[tuple[str, dict[str, Any]]]:
    return [
        (job_id, step)
        for job_id, job in _load(path)["jobs"].items()
        for step in job.get("steps", [])
    ]


def test_the_four_workflows_are_covered() -> None:
    assert [p.name for p in WORKFLOWS] == [
        "artifact-language.yml",
        "auto-release.yml",
        "ci.yml",
        "release.yml",
    ]


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_every_action_is_pinned_to_a_commit_with_its_version(
    path: pathlib.Path,
) -> None:
    uses = _uses_lines(path)
    assert uses, path.name
    unpinned = [u for u in uses if not PINNED.match(u)]
    assert unpinned == [], unpinned


def test_each_action_is_pinned_to_one_commit_everywhere() -> None:
    by_action: dict[str, set[str]] = {}
    for path in WORKFLOWS:
        for line in _uses_lines(path):
            action, _, rest = line.partition("@")
            by_action.setdefault(action, set()).add(rest)
    assert {a: v for a, v in by_action.items() if len(v) > 1} == {}


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_no_checkout_keeps_credentials(path: pathlib.Path) -> None:
    checkouts = [
        (job, step)
        for job, step in _steps(path)
        if str(step.get("uses", "")).startswith("actions/checkout@")
    ]
    assert checkouts, path.name
    for job, step in checkouts:
        assert (step.get("with") or {}).get("persist-credentials") is False, job


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_every_job_has_a_timeout(path: pathlib.Path) -> None:
    jobs = _load(path)["jobs"]
    missing = [job_id for job_id, job in jobs.items() if not job.get("timeout-minutes")]
    assert missing == []


def test_every_workflow_declares_its_token() -> None:
    expected = {
        "artifact-language.yml": {
            "contents": "read",
            "pull-requests": "read",
            "issues": "read",
        },
        "auto-release.yml": {},
        "ci.yml": {"contents": "read"},
        "release.yml": {},
    }
    assert {p.name: _load(p).get("permissions") for p in WORKFLOWS} == expected


def test_the_release_build_restores_no_cache() -> None:
    steps = _load(ROOT / ".github" / "workflows" / "release.yml")["jobs"]["build"][
        "steps"
    ]
    setup = [
        s for s in steps if str(s.get("uses", "")).startswith("astral-sh/setup-uv@")
    ]
    assert len(setup) == 1
    assert setup[0]["with"]["enable-cache"] is False
    assert not any(str(s.get("uses", "")).startswith("actions/cache") for s in steps)


def test_the_release_jobs_carry_only_their_own_permissions() -> None:
    jobs = _load(ROOT / ".github" / "workflows" / "release.yml")["jobs"]
    assert jobs["build"]["permissions"] == {"contents": "read"}
    assert jobs["publish-testpypi"]["permissions"] == {"id-token": "write"}
    assert jobs["publish-pypi"]["permissions"] == {"id-token": "write"}
    assert jobs["publish-testpypi"]["environment"] == "testpypi"
    assert jobs["publish-pypi"]["environment"] == "pypi"
    assert jobs["publish-pypi"]["needs"] == "publish-testpypi"


def test_auto_release_dispatches_the_publish_on_the_tag() -> None:
    job = _load(ROOT / ".github" / "workflows" / "auto-release.yml")["jobs"][
        "tag-and-release"
    ]
    assert job["permissions"] == {"contents": "write", "actions": "write"}
    runs = [str(s.get("run", "")) for s in job["steps"]]
    dispatch = [r for r in runs if "gh workflow run" in r]
    assert dispatch == ['gh workflow run release.yml --ref "$TAG" -f tag="$TAG"']


def test_ci_cancels_superseded_pull_request_runs_only() -> None:
    concurrency = _load(ROOT / ".github" / "workflows" / "ci.yml")["concurrency"]
    assert concurrency == {
        "group": "ci-${{ github.event.pull_request.number || github.ref }}",
        "cancel-in-progress": "${{ github.event_name == 'pull_request' }}",
    }


@pytest.mark.parametrize("name", ["release.yml", "auto-release.yml"])
def test_release_runs_are_never_cancelled(name: str) -> None:
    concurrency = _load(ROOT / ".github" / "workflows" / name)["concurrency"]
    assert concurrency["cancel-in-progress"] is False


def test_dependabot_moves_the_lock_and_actions_on_develop() -> None:
    config = _load(ROOT / ".github" / "dependabot.yml")
    updates = {u["package-ecosystem"]: u for u in config["updates"]}
    assert set(updates) == {"uv", "github-actions"}
    for update in updates.values():
        assert update["target-branch"] == "develop"
        assert update["cooldown"] == {"default-days": 7}
    assert updates["uv"]["versioning-strategy"] == "lockfile-only"
