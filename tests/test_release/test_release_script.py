"""``scripts/release.py`` opens the release PR and nothing else (H-0117).

The script used to commit a dirty ``CHANGELOG.md`` on develop and push develop
directly, which ``CONTRIBUTING.md`` forbids. These tests replace its command
runner with a recorder, so each case states exactly which commands ran.
"""

from __future__ import annotations

import importlib.util
import pathlib
from collections.abc import Callable
from types import ModuleType

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
VERSION = "v0.18.0"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "lizyml_release_script", ROOT / "scripts" / "release.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _recorder(
    *, changelog_dirty: bool = False, local: str = "a" * 40, remote: str = "a" * 40
) -> tuple[list[str], Callable[..., str]]:
    ran: list[str] = []
    answers = {
        "git branch --show-current": "develop",
        "git tag --sort=-v:refname": "v0.17.1\nv0.17.0",
        "git status --porcelain CHANGELOG.md": " M CHANGELOG.md"
        if changelog_dirty
        else "",
        "git fetch origin develop": "",
        "git rev-parse HEAD": local,
        "git rev-parse origin/develop": remote,
    }

    def run(cmd: str, *, check: bool = True) -> str:
        ran.append(cmd)
        if cmd in answers:
            return answers[cmd]
        if cmd.startswith("gh pr create "):
            return "https://github.com/nbx-liz/LizyML/pull/999"
        raise AssertionError(f"release.py ran an unexpected command: {cmd!r}")

    return ran, run


@pytest.fixture
def release(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [0.18.0] - 2026-10-09\n\n- Notes.\n\n"
        "## [0.17.1] - 2026-07-04\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["release.py", VERSION])
    return _load()


def _fetched_before_comparing(ran: list[str]) -> bool:
    """origin/develop is refreshed before it is read, or a stale ref passes."""
    fetch = "git fetch origin develop"
    read = "git rev-parse origin/develop"
    return fetch in ran and read in ran and ran.index(fetch) < ran.index(read)


def _mutating(ran: list[str]) -> list[str]:
    return [c for c in ran if c.startswith(("git commit", "git push", "git add"))]


def test_a_dirty_changelog_is_refused_without_committing(
    release: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran, run = _recorder(changelog_dirty=True)
    monkeypatch.setattr(release, "run", run)
    with pytest.raises(SystemExit) as exc:
        release.main()
    assert exc.value.code == 1
    assert _mutating(ran) == []
    assert not any(c.startswith("gh pr create") for c in ran)


def test_a_develop_out_of_sync_with_origin_is_refused(
    release: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran, run = _recorder(local="a" * 40, remote="b" * 40)
    monkeypatch.setattr(release, "run", run)
    with pytest.raises(SystemExit) as exc:
        release.main()
    assert exc.value.code == 1
    assert _mutating(ran) == []
    assert not any(c.startswith("gh pr create") for c in ran)
    assert _fetched_before_comparing(ran)


def test_the_happy_path_only_opens_the_release_pr(
    release: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran, run = _recorder()
    monkeypatch.setattr(release, "run", run)
    release.main()
    assert _mutating(ran) == []
    assert _fetched_before_comparing(ran)
    created = [c for c in ran if c.startswith("gh pr create")]
    assert len(created) == 1
    assert "--base main --head develop" in created[0]
    assert f'--title "release: {VERSION}"' in created[0]
