"""H-0144 対応方針 8 / acceptance 1a: every oracle and overlay is recorded.

``docs/audits/2026-10-h0138/oracle/provenance.json`` names each H-0138 oracle
and H-0139 overlay with its SHA-256 and the environment it was made in, and
the frozen codec the overlay generator reads. A file without a record, a
record without a file, or a changed byte fails here (DC3).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests/test_features/data"
RECORD = json.loads(
    (ROOT / "docs/audits/2026-10-h0138/oracle/provenance.json").read_text()
)
FROZEN = ROOT / "docs/audits/2026-10-i389/oracle/frozen_core"
GEN = "docs/audits/2026-10-h0139/oracle/make_overlay.py"
FAMILIES = ("h0138_oracle_pandas-*.json.gz", "h0139_overlay_pandas-*.json.gz")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_every_file_has_its_record_and_no_record_lacks_a_file() -> None:
    on_disk = {p.name for pattern in FAMILIES for p in DATA.glob(pattern)}
    assert on_disk == set(RECORD["files"])


@pytest.mark.parametrize("name", sorted(RECORD["files"]))
def test_the_bytes_and_environment_match_the_record(name: str) -> None:
    entry: dict[str, Any] = RECORD["files"][name]
    path = DATA / name
    assert _sha(path) == entry["sha256"]
    env = entry["environment"]
    version = name.rsplit("pandas-", 1)[1].removesuffix(".json.gz")
    assert env["pandas"] == version
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        meta = json.load(fh)["meta"]
    assert meta["pandas"] == version
    assert meta["numpy"] == env["numpy"]
    if name.startswith("h0138_oracle"):
        assert meta["lightgbm"] == env["lightgbm"]
    # Every oracle has its overlay, made in the same environment.
    if name.startswith("h0138_oracle"):
        partner = name.replace("h0138_oracle", "h0139_overlay")
    else:
        partner = name.replace("h0139_overlay", "h0138_oracle")
    assert RECORD["files"][partner]["environment"] == env


def test_the_frozen_codec_is_the_recorded_one() -> None:
    files = {p.name: _sha(p) for p in sorted(FROZEN.glob("*.py"))}
    assert files == RECORD["frozen_core"]["files"]
    assert set(files) == {"category_values.py", "category_temporal.py", "exceptions.py"}


def test_the_overlay_generator_reads_only_the_frozen_codec() -> None:
    """Acceptance 1a: with the generator's hook installed, the codec modules
    are the frozen files and any other lizyml module is refused."""
    probe = (
        "import importlib, runpy, sys\n"
        f"runpy.run_path({GEN!r}, run_name='x')\n"
        "import lizyml.core.category_values as cv\n"
        "print(cv.__file__)\n"
        "try:\n"
        "    importlib.import_module('lizyml.core.model')\n"
        "except ImportError:\n"
        "    print('refused')\n"
    )
    env = {
        **os.environ,
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": f"{ROOT}{os.pathsep}{ROOT / 'docs/audits/2026-10-h0139/firing'}",
    }
    out = subprocess.run(
        [sys.executable, "-B", "-c", probe],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    ).stdout.split()
    assert Path(out[0]) == FROZEN / "category_values.py"
    assert out[1] == "refused"


def test_the_generator_refuses_to_run_after_lizyml_was_imported() -> None:
    probe = (
        "import lizyml.core.category_values, runpy\n"
        f"runpy.run_path({GEN!r}, run_name='x')\n"
    )
    env = {
        **os.environ,
        "PYTHONPATH": f"{ROOT}{os.pathsep}{ROOT / 'docs/audits/2026-10-h0139/firing'}",
    }
    result = subprocess.run(
        [sys.executable, "-B", "-c", probe],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode != 0
    assert "AssertionError" in result.stderr
