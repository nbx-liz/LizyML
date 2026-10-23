"""H-0144: the overlay generator's import hook serves only the frozen codec.

Run from the repository root:
    PYTHONHASHSEED=0 PYTHONPATH=.:docs/audits/2026-10-h0139/firing \
        python -B docs/audits/2026-10-i389/probes/frozen_hook_check.py
"""

from __future__ import annotations

import hashlib
import importlib
import runpy
import sys
from pathlib import Path

GEN = Path("docs/audits/2026-10-h0139/oracle/make_overlay.py")
# Load the generator's module body without running main(): the hook installs.
runpy.run_path(str(GEN), run_name="h0144_hook_check")
for name in ("lizyml.core.category_values", "lizyml.core.category_temporal",
             "lizyml.core.exceptions"):  # fmt: skip
    mod = importlib.import_module(name)
    path = Path(mod.__file__)
    print(name, path.relative_to(Path.cwd()), hashlib.sha256(path.read_bytes()).hexdigest()[:16])
for name in ("lizyml.core.model", "lizyml.features.encoders.category_identity", "lizyml.codegen"):
    try:
        importlib.import_module(name)
    except ImportError as exc:
        print("refused", name, "->", exc)
    else:
        print("NOT REFUSED", name)
        sys.exit(1)
print("lizyml modules loaded:", sorted(m for m in sys.modules if m.startswith("lizyml")))
