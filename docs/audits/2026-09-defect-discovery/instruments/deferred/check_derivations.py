"""Every manifest derivation must run against the shipped code and print its
declared population. Run before the manifest is written, not after."""

from __future__ import annotations

import json
import pathlib
import subprocess

PY = "/home/rem/repos/LizyML/.venv/bin/python"
REPO = pathlib.Path("/home/rem/repos/LizyML")
MANIFEST = pathlib.Path("/tmp/lizyml-discovery-plan/phase3_manifest.json")

data = json.loads(MANIFEST.read_text())
bad = 0
for num, row in sorted(data["issues"].items(), key=lambda kv: int(kv[0])):
    snippet = row["derived_from"]
    p = subprocess.run([PY, "-c", snippet], cwd=REPO, capture_output=True, text=True)
    got = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
    # #263's population is the enum AFTER PR 6 removes a member, so its
    # derivation legitimately prints one more at the before-SHA. Every other
    # row must match exactly; this exemption is named, not a tolerance.
    expected = {str(row["population"])}
    if num == "263":
        expected.add(str(row["population"] + 1))
    ok = p.returncode == 0 and got in expected
    flag = "ok " if ok else "FAIL"
    if not ok:
        bad += 1
    print(f"{flag} #{num}: declared {row['population']:>3}  got {got or p.stderr.strip()[-90:]!r}")
print()
print(f"{bad} derivation(s) disagree with the manifest")
raise SystemExit(1 if bad else 0)
