"""H-0129 should-fail mutations for layer_rules.py.

Copies ``lizyml/`` into a temporary directory, appends one import form at a
time to ``persistence/loader.py`` (a layer-3 module), and checks that
layer_rules.py reports it in section [A]. Each form is one H-0128 review
finding: the original absolute import, a relative import, and an imported name.
A last case (no mutation, ``exporter.py``'s current import removed) checks the
probe exits 0 once the real violation is gone, so a probe that always fails
cannot pass this file.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PROBE = HERE / "layer_rules.py"
VIOLATION = "from lizyml.estimators.lgbm.category_json import"

FORMS = {
    "absolute": "from lizyml.estimators.lgbm.category_json import encoder_categories\n",
    "relative": "from ..estimators.lgbm import category_json\n",
    "imported name": "from lizyml.estimators import lgbm\n",
    "plain import": "import lizyml.estimators.lgbm.category_json\n",
}


def run(root: Path) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-I", str(PROBE), "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    section = proc.stdout.split("[A]", 1)[1].split("[B]", 1)[0]
    return proc.returncode, section


def fresh_tree(tmp: Path) -> Path:
    root = tmp / "lizyml"
    if root.exists():
        shutil.rmtree(root)
    shutil.copytree(REPO / "lizyml", root, ignore=shutil.ignore_patterns("__pycache__"))
    exporter = root / "persistence" / "exporter.py"
    lines = exporter.read_text(encoding="utf-8").splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if line.startswith(VIOLATION))
    end = start
    while not lines[end].rstrip().endswith(")") and "(" in lines[start]:
        end += 1
    exporter.write_text("".join(lines[:start] + lines[end + 1 :]), encoding="utf-8")
    return root


failures = 0
with tempfile.TemporaryDirectory() as name:
    tmp = Path(name)
    root = fresh_tree(tmp)
    code, section = run(root)
    ok = code == 0 and "persistence" not in section
    print(f"control (violation removed): exit {code} -> {'PASS' if ok else 'FAIL'}")
    failures += not ok
    for label, line in FORMS.items():
        root = fresh_tree(tmp)
        loader = root / "persistence" / "loader.py"
        loader.write_text(loader.read_text(encoding="utf-8") + line, encoding="utf-8")
        code, section = run(root)
        ok = code == 1 and "persistence/loader.py" in section
        print(f"{label}: exit {code} -> {'caught' if ok else 'MISSED'}")
        failures += not ok
print(f"failures: {failures}")
raise SystemExit(1 if failures else 0)
