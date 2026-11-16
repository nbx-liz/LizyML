"""``release.yml`` builds only on its own release tag (H-0146 acceptance 4).

The ``pypi`` / ``testpypi`` environments allow ``v*`` tags only, so the run must
be on ``refs/tags/<tag>``. ``.github/scripts/check_release_ref.sh`` is executed
here with the two inputs it reads; the workflow file is read to check the step
runs before anything is installed or built.
"""

from __future__ import annotations

import os
import pathlib
import subprocess

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
CHECK = ROOT / ".github" / "scripts" / "check_release_ref.sh"
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"


def _check(
    tmp_path: pathlib.Path, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    base = {k: v for k, v in os.environ.items() if k not in ("TAG", "GITHUB_REF")}
    return subprocess.run(
        ["bash", str(CHECK)],
        cwd=tmp_path,
        env={**base, **env},
        capture_output=True,
        text=True,
        check=False,
    )


def test_a_run_on_its_tag_passes(tmp_path: pathlib.Path) -> None:
    result = _check(tmp_path, {"TAG": "v0.21.2", "GITHUB_REF": "refs/tags/v0.21.2"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "refs/tags/v0.21.2" in result.stdout


@pytest.mark.parametrize(
    ("tag", "ref"),
    [
        # The dispatch without --ref that every release used before H-0146.
        ("v0.21.2", "refs/heads/main"),
        ("v0.21.2", "refs/heads/develop"),
        # Another tag than the one asked for.
        ("v0.21.2", "refs/tags/v0.21.1"),
        ("v0.21.2", "refs/tags/v0.21.2x"),
        ("v0.21.2", "refs/tags/v0.21.2/x"),
        # Tags that are not a release.
        ("v0.21.2-rc1", "refs/tags/v0.21.2-rc1"),
        ("0.21.2", "refs/tags/0.21.2"),
        ("v0.21", "refs/tags/v0.21"),
        ("v0.21.2\nv9.9.9", "refs/tags/v0.21.2\nv9.9.9"),
        # Missing inputs.
        ("", "refs/tags/v0.21.2"),
        ("v0.21.2", ""),
    ],
)
def test_any_other_ref_or_tag_is_refused(
    tmp_path: pathlib.Path, tag: str, ref: str
) -> None:
    result = _check(tmp_path, {"TAG": tag, "GITHUB_REF": ref})
    assert result.returncode != 0, f"TAG={tag!r} GITHUB_REF={ref!r} was accepted"
    assert "::error::" in result.stdout


def test_an_unset_input_is_refused(tmp_path: pathlib.Path) -> None:
    result = _check(tmp_path, {"GITHUB_REF": "refs/tags/v0.21.2"})
    assert result.returncode != 0
    assert "TAG is not set" in result.stdout


def _build_steps() -> list[dict[str, object]]:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps: list[dict[str, object]] = workflow["jobs"]["build"]["steps"]
    return steps


def test_the_check_runs_first_after_the_checkout() -> None:
    steps = _build_steps()
    assert str(steps[0].get("uses", "")).startswith("actions/checkout@")
    check = steps[1]
    assert check["run"] == f"bash .github/scripts/{CHECK.name}"
    assert check["env"] == {"TAG": "${{ inputs.tag || github.event.release.tag_name }}"}


def test_the_checkout_takes_the_runs_own_ref() -> None:
    with_ = _build_steps()[0].get("with", {})
    assert isinstance(with_, dict)
    # No `ref:` input: the build checks out the run's ref, which the check pins.
    assert "ref" not in with_
