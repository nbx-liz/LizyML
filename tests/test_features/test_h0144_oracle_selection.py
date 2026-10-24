"""H-0144 acceptance 2-4: which H-0138 oracle a lane compares against.

Only the latest-deps CI job may fall back to the nearest older oracle of the
same pandas minor; every other lane compares its own version's file or fails.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import lightgbm
import numpy as np
import pytest
import yaml
from packaging.version import InvalidVersion

from tests.test_features import _h0138_oracle as h

ROOT = Path(__file__).resolve().parents[2]
LATEST = {"GITHUB_ACTIONS": "true", "GITHUB_JOB": h.LATEST_JOB}
ON = {**LATEST, h.MODE_ENV: h.SAME_MINOR}
#: A fixed, synthetic set of oracle versions for the selection rule; the
#: shipped inventory may grow (CONTRIBUTING.md onboarding) without touching it.
EXAMPLE = ["2.0.0", "2.3.3", "3.0.1"]


@pytest.mark.parametrize(
    "environ",
    [
        {},
        LATEST,  # the job without the mode
        {"GITHUB_ACTIONS": "true", "GITHUB_JOB": "quality"},
    ],
)
def test_without_the_mode_a_lane_is_strict(environ: dict[str, str]) -> None:
    assert h.oracle_mode(environ) is None


@pytest.mark.parametrize(
    "environ",
    [
        {h.MODE_ENV: h.SAME_MINOR},  # outside CI
        {h.MODE_ENV: h.SAME_MINOR, "GITHUB_ACTIONS": "true"},
        {h.MODE_ENV: h.SAME_MINOR, "GITHUB_ACTIONS": "true", "GITHUB_JOB": "quality"},
        {
            h.MODE_ENV: h.SAME_MINOR,
            "GITHUB_ACTIONS": "false",
            "GITHUB_JOB": h.LATEST_JOB,
        },
        {**LATEST, h.MODE_ENV: ""},
        {**LATEST, h.MODE_ENV: "latest"},
        {**LATEST, h.MODE_ENV: "Latest-Same-Minor"},
    ],
)
def test_a_misused_mode_fails_instead_of_loosening(environ: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        h.oracle_mode(environ)
    resolved = h.resolve(environ, running="9.9.9")
    assert resolved.version is None and resolved.error is not None


def test_the_mode_is_accepted_only_in_the_latest_job() -> None:
    assert h.oracle_mode(ON) == h.SAME_MINOR


@pytest.mark.parametrize("mode", [None, h.SAME_MINOR])
def test_a_lanes_own_version_comes_first(mode: str | None) -> None:
    assert h.select_version("3.0.1", ["3.0.0", "3.0.1"], mode) == "3.0.1"


def test_a_strict_lane_without_its_version_has_no_oracle() -> None:
    assert h.select_version("3.0.6", EXAMPLE, None) is None


@pytest.mark.parametrize(
    ("running", "available", "selected"),
    [
        ("3.0.6", EXAMPLE, "3.0.1"),
        ("3.0.10", ["3.0.1", "3.0.9", "3.0.2"], "3.0.9"),  # numeric, not text
        ("3.0.1.post1", EXAMPLE, "3.0.1"),
        ("3.0.2rc1", EXAMPLE, "3.0.1"),
        ("3.0.2.dev1", EXAMPLE, "3.0.1"),
        ("3.0.6+local", EXAMPLE, "3.0.1"),
        ("2.3.5", EXAMPLE, "2.3.3"),
        ("2.0.3", EXAMPLE, "2.0.0"),
        ("3.0.1rc1", EXAMPLE, None),  # older than 3.0.1
        ("3.1.0", EXAMPLE, None),  # a new minor: red until its oracle exists
        ("3.1.0rc1", EXAMPLE, None),
        ("3.1.0.dev0", EXAMPLE, None),
        ("4.0.0", EXAMPLE, None),
        ("3.0.0", EXAMPLE, None),
        ("1!3.0.6", EXAMPLE, None),  # another epoch
    ],
)
def test_same_minor_selection(
    running: str, available: list[str], selected: str | None
) -> None:
    assert h.select_version(running, available, h.SAME_MINOR) == selected


@pytest.mark.parametrize(
    ("running", "available"),
    [("not-a-version", EXAMPLE), ("3.0.6", ["3.0.1", "garbage"])],
)
def test_an_unparsable_version_fails(running: str, available: list[str]) -> None:
    with pytest.raises(InvalidVersion):
        h.select_version(running, available, h.SAME_MINOR)


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_bytes(gzip.compress(json.dumps(payload).encode(), mtime=0))


def test_resolve_reads_the_files_and_keeps_the_versions_together(
    tmp_path: Path,
) -> None:
    for v in ("3.0.1", "3.0.4"):
        _write(tmp_path / f"h0138_oracle_pandas-{v}.json.gz", {"meta": {"pandas": v}})
    _write(
        tmp_path / "h0139_overlay_pandas-3.0.1.json.gz", {"meta": {"pandas": "3.0.1"}}
    )
    assert h.oracle_versions(tmp_path) == ["3.0.1", "3.0.4"]
    resolved = h.resolve(ON, running="3.0.6", data=tmp_path)
    assert (resolved.version, resolved.exact, resolved.error) == ("3.0.4", False, None)
    strict = h.resolve({}, running="3.0.6", data=tmp_path)
    assert (strict.version, strict.error) == (None, None)
    broken = h.resolve(ON, running="3.0.6", data=tmp_path / "missing")
    assert broken.version is None


def test_an_unparsable_file_name_is_an_error(tmp_path: Path) -> None:
    _write(tmp_path / "h0138_oracle_pandas-3.0.x.json.gz", {"meta": {}})
    resolved = h.resolve(ON, running="3.0.6", data=tmp_path)
    assert resolved.version is None and resolved.error is not None


def test_the_shipped_oracles_and_overlays_pair_up() -> None:
    """Every shipped oracle has its overlay and every overlay its oracle, so a
    selected oracle is never paired with another version's overlay."""
    versions = h.oracle_versions()
    assert {"2.0.0", "2.3.3", "3.0.1"} <= set(versions)
    overlays = sorted(
        p.name.removeprefix("h0139_overlay_pandas-").removesuffix(".json.gz")
        for p in h.DATA.glob("h0139_overlay_pandas-*.json.gz")
    )
    assert overlays == versions


def test_models_are_skipped_only_in_same_minor_mode() -> None:
    meta = {"numpy": "0.0.0", "lightgbm": "0.0.0"}
    assert h.compare_models(meta, None) is True  # strict: run and fail
    assert h.compare_models(meta, h.SAME_MINOR) is False
    same = {"numpy": np.__version__, "lightgbm": lightgbm.__version__}
    assert h.compare_models(same, h.SAME_MINOR) is True


def test_only_the_latest_job_sets_the_mode() -> None:
    """The CI job that sets the mode is the one the guard names (DC3)."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    setters = {
        (job_id, step.get("name"))
        for job_id, job in workflow["jobs"].items()
        for step in job.get("steps", [])
        if h.MODE_ENV in (step.get("env") or {}) or h.MODE_ENV in (job.get("env") or {})
    }
    assert setters == {(h.LATEST_JOB, "Test (pytest)")}
    assert h.MODE_ENV not in (workflow.get("env") or {})
    step = next(
        s
        for s in workflow["jobs"][h.LATEST_JOB]["steps"]
        if s.get("name") == "Test (pytest)"
    )
    assert step["env"][h.MODE_ENV] == h.SAME_MINOR
