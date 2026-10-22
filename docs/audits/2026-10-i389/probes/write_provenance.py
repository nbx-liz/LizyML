"""Write docs/audits/2026-10-h0138/oracle/provenance.json (H-0144 対応方針 8).

Hashes come from the committed files; environments are the ones the
regeneration in ``regen_existing.sh`` (and, for a new minor, the onboarding
run) used, given on the command line as ``version=python,numpy,lightgbm,pyarrow``.

    python docs/audits/2026-10-i389/probes/write_provenance.py \
        3.0.1=3.12.3,2.4.2,4.6.0,23.0.1 2.3.3=3.10.22,2.2.6,4.6.0,23.0.1 \
        2.0.0=3.10.22,1.24.0,4.4.0,23.0.1
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

DATA = Path("tests/test_features/data")
FROZEN = Path("docs/audits/2026-10-i389/oracle/frozen_core")
OUT = Path("docs/audits/2026-10-h0138/oracle/provenance.json")
ORACLE_COMMIT = "b297de42c1146cb6e7132366cdd9f03e4400e6bb"
FROZEN_COMMIT = "f0256208195c061b55441a049f8fdcd8f6abc9aa"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    envs: dict[str, dict[str, str | None]] = {}
    for arg in sys.argv[1:]:
        version, values = arg.split("=")
        python, numpy, lightgbm, pyarrow = values.split(",")
        envs[version] = {
            "python": python,
            "pandas": version,
            "numpy": numpy,
            "lightgbm": lightgbm,
            "pyarrow": None if pyarrow == "none" else pyarrow,
        }
    files: dict[str, dict[str, object]] = {}
    for version, env in sorted(envs.items()):
        oracle = DATA / f"h0138_oracle_pandas-{version}.json.gz"
        overlay = DATA / f"h0139_overlay_pandas-{version}.json.gz"
        files[oracle.name] = {
            "sha256": sha(oracle),
            "generator": "docs/audits/2026-10-h0138/oracle/make_oracle.py",
            "library_commit": ORACLE_COMMIT,
            "environment": env,
            "command": "PYTHONHASHSEED=0 PYTHONPATH=.:docs/audits/2026-10-h0138/firing "
            "python -B docs/audits/2026-10-h0138/oracle/make_oracle.py "
            "(in a b297de4 worktree with develop's docs/audits/2026-10-h0138/{oracle,firing})",
        }
        files[overlay.name] = {
            "sha256": sha(overlay),
            "generator": "docs/audits/2026-10-h0139/oracle/make_overlay.py",
            "library_commit": f"frozen_core at {FROZEN_COMMIT}",
            "environment": env,
            "command": "PYTHONHASHSEED=0 PYTHONPATH=.:docs/audits/2026-10-h0139/firing "
            "python -B docs/audits/2026-10-h0139/oracle/make_overlay.py",
        }
    payload = {
        "about": "H-0144: how every H-0138 oracle and H-0139 overlay was made; "
        "tests/test_features/test_h0144_oracle_provenance.py holds the files to it",
        "frozen_core": {
            "source_commit": FROZEN_COMMIT,
            "files": {p.name: sha(p) for p in sorted(FROZEN.glob("*.py"))},
        },
        "files": files,
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(OUT, len(files), "files")


if __name__ == "__main__":
    main()
