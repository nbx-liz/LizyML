"""Re-derive every population `phase3-plan.md` declares, at the current head.

D5 in DECISIONS-PENDING asserts that the later PRs' scopes were written the way
PR 1's was -- from issue text rather than from a scan. That claim was written
without checking. This checks it: each number the plan states is recomputed from
the tree, and the plan is right or wrong per row.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import re
import sys

REPO = pathlib.Path("/home/rem/repos/LizyML")
sys.path.insert(0, str(REPO))

rows: list[tuple[str, str, str, str]] = []


def row(pr: str, what: str, claimed: object, measured: object) -> None:
    ok = "OK" if str(claimed) == str(measured) else "MISMATCH"
    rows.append((pr, what, f"{claimed} -> {measured}", ok))


# --- PR 3: (task, metric) pairs in _TASK_METRICS ---------------------------
from lizyml.core._model_metrics import _DEFAULT_METRICS  # noqa: E402

try:
    from lizyml.metrics.registry import _TASK_METRICS  # type: ignore[attr-defined]
except Exception:
    _TASK_METRICS = None

if _TASK_METRICS is None:
    # Find it wherever it lives.
    import importlib
    import pkgutil

    import lizyml

    for mod in pkgutil.walk_packages(lizyml.__path__, "lizyml."):
        try:
            m = importlib.import_module(mod.name)
        except Exception:
            continue
        if hasattr(m, "_TASK_METRICS"):
            _TASK_METRICS = m._TASK_METRICS
            print(f"_TASK_METRICS found in {mod.name}")
            break

if _TASK_METRICS is not None:
    pairs = sum(len(v) for v in _TASK_METRICS.values())
    row("PR 3", "(task, metric) pairs in _TASK_METRICS", 22, pairs)
else:
    row("PR 3", "(task, metric) pairs in _TASK_METRICS", 22, "NOT FOUND")

# --- PR 4: CVTrainer.fit / RefitTrainer.fit parameters ----------------------
from lizyml.training.cv_trainer import CVTrainer  # noqa: E402

try:
    from lizyml.training.refit_trainer import RefitTrainer  # noqa: E402
except Exception:  # pragma: no cover
    from lizyml.training.refit import RefitTrainer  # type: ignore[no-redef]

cv = [p for p in inspect.signature(CVTrainer.fit).parameters if p != "self"]
rf = [p for p in inspect.signature(RefitTrainer.fit).parameters if p != "self"]
row("PR 4", "CVTrainer.fit parameters", 7, len(cv))
row("PR 4", "RefitTrainer.fit parameters", 3, len(rf))
row("PR 4", "union", 7, len(set(cv) | set(rf)))
print(f"  CVTrainer.fit    : {cv}")
print(f"  RefitTrainer.fit : {rf}")

# --- PR 5: UnseenPolicy values ---------------------------------------------
import typing  # noqa: E402

found_policy = None
import importlib  # noqa: E402
import pkgutil  # noqa: E402

import lizyml  # noqa: E402

for mod in pkgutil.walk_packages(lizyml.__path__, "lizyml."):
    try:
        m = importlib.import_module(mod.name)
    except Exception:
        continue
    if hasattr(m, "UnseenPolicy"):
        found_policy = m.UnseenPolicy
        break
if found_policy is not None:
    vals = typing.get_args(found_policy) or tuple(
        getattr(found_policy, "__members__", {})
    )
    row("PR 5", "UnseenPolicy values", 3, len(vals))
    print(f"  UnseenPolicy: {vals}")
else:
    row("PR 5", "UnseenPolicy values", 3, "NOT FOUND")

# --- PR 6: ErrorCode members ------------------------------------------------
from lizyml.core.exceptions import ErrorCode  # noqa: E402

row("PR 6", "ErrorCode members", 20, len(list(ErrorCode)))

# --- PR 8: defaulted / keyword-only __init__ parameters of public classes ---
knobs: list[str] = []
for path in sorted((REPO / "lizyml").rglob("*.py")):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name.startswith("_"):
            continue
        for sub in node.body:
            if not isinstance(sub, ast.FunctionDef) or sub.name != "__init__":
                continue
            a = sub.args
            n_default = len(a.defaults)
            for arg in a.args[len(a.args) - n_default :]:
                knobs.append(f"{node.name}.{arg.arg}")
            for arg in a.kwonlyargs:
                knobs.append(f"{node.name}.{arg.arg}")
row("PR 8", "defaulted / kw-only __init__ params", 74, len(knobs))

# --- PR 9: proposals in HISTORY.md -----------------------------------------
history = (REPO / "HISTORY.md").read_text(encoding="utf-8")
proposals = re.findall(r"^## (H-\d{4})", history, re.MULTILINE)
row("PR 9", "H-NNNN proposals in HISTORY.md", 92, len(set(proposals)))

# --- PR 2: Model public methods -------------------------------------------
from lizyml.core.model import Model  # noqa: E402

public = [
    name
    for name, obj in vars(Model).items()
    if not name.startswith("_") and callable(obj)
]
row("PR 2", "Model public callables", "(unstated)", len(public))
print(f"  Model public callables: {sorted(public)}")

print()
print(f"{'PR':<6} {'what':<45} {'claimed -> measured':<26} verdict")
print("-" * 92)
for pr, what, delta, ok in rows:
    print(f"{pr:<6} {what:<45} {delta:<26} {ok}")
print()
print("_DEFAULT_METRICS keys:", sorted(_DEFAULT_METRICS))
