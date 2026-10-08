"""D6 — test hollowness sweep.

Population: every `def test_*` under tests/. Axis 1 is a binary partition of the
claim (structural / behavioural). Axis 2, for behavioural claims, is which of the
444 schema operations produced the asserted value, read from the execution trace
the instrumented suite recorded.

Candidate hollow = a behavioural claim whose trace contains none of its producers.
A claim matching no effect noun is CANNOT-TELL, never sound.
"""

from __future__ import annotations

import ast
import json
import os
import pathlib
import re
import sys


def _out_dir() -> pathlib.Path:
    """The D6 work directory, from LIZYML_D6_OUT. Refuses to guess one."""
    value = os.environ.get("LIZYML_D6_OUT")
    if not value:
        raise SystemExit("set LIZYML_D6_OUT to the D6 work directory")
    path = pathlib.Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path

ROOT = pathlib.Path(__file__).resolve().parents[4]
RES = _out_dir()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from measure_extra import operation_schema  # noqa: E402

TREES = {}
for p in sorted((ROOT / "lizyml").rglob("*.py")):
    TREES[p] = ast.parse(p.read_text(encoding="utf-8"))
SCHEMA = set(operation_schema(TREES))

# ------------------------------------------------- populations -> operations
STAGES: set[str] = set()
for p, tree in TREES.items():
    rel = p.relative_to(ROOT).as_posix()
    if not (rel.startswith("lizyml/training/") or rel.startswith("lizyml/calibration/")):
        continue
    mod = rel[:-3].replace("/", ".")
    for n in tree.body:
        if not isinstance(n, ast.ClassDef) or n.name.startswith("_"):
            continue
        for m in n.body:
            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name in ("fit", "split", "run"):
                STAGES.add(f"{mod}.{n.name}.{m.name}")
                break

SPLITTERS: set[str] = set()
for p, tree in TREES.items():
    if "/splitters/" not in p.as_posix():
        continue
    mod = p.relative_to(ROOT).as_posix()[:-3].replace("/", ".")
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name != "BaseSplitter":
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name == "split":
                    SPLITTERS.add(f"{mod}.{n.name}.split")

from lizyml.core.model import Model  # noqa: E402

PUBLIC_API: set[str] = set()
for name in (n for n in dir(Model) if not n.startswith("_")):
    for op in SCHEMA:
        if op.endswith(f".Model.{name}") or op.endswith(f"Mixin.{name}"):
            PUBLIC_API.add(op)

# D3b enumerates 22 task-metric entries in `lizyml/metrics/registry.py`; the
# operations that implement them are the metric subsystem. Adding this is the
# same repair the plan already made once: `D5.splitter_operations` was added
# after `test_correct_fold_count` -- which calls `KFoldSplitter.split` -- came
# out CANDIDATE-HOLLOW because the split producer set named only D4's training
# and calibration stages. The metric nouns had the identical gap: a unit test of
# `Evaluator` or of a metric claims "metrics" and is compared against `Model`'s
# public members, which it never calls. The value is a measured population, not
# an authorial choice.
METRIC_OPS = {op for op in SCHEMA if op.startswith("lizyml.metrics.")}
EVAL_OPS = {op for op in SCHEMA if op.startswith("lizyml.evaluation.")}

POPS = {
    "D1a.boundary_operations": {"lightgbm.train"},
    "D4.stage_entry_points": STAGES,
    "D5.splitter_operations": SPLITTERS,
    "D3.public_api": PUBLIC_API,
    "D3b.metric_operations": METRIC_OPS | EVAL_OPS,
}
for k, v in POPS.items():
    outside = v - SCHEMA
    assert not outside, f"{k} names operations outside the schema: {sorted(outside)[:3]}"

EFFECT_TABLE = [
    (("training", "trains", "trained", "fit", "fitting", "fitted", "boosting",
      "booster", "importance", "importances"), ["D1a.boundary_operations"]),
    # The plan's nouns verbatim. An earlier version substituted "inner",
    # "valid", "early", "stopping" for the plan's `inner_valid` and
    # `early_stopping` on the ground that the word tokenizer can never match an
    # underscored token -- which is true, and which is the plan's own choice.
    # Widening them pulled every config test that mentions "valid" into the
    # split producer set and manufactured 58 candidates. A noun that cannot
    # match is a declared gap; replacing it with looser ones is authorial
    # latitude, which is exactly what the effect table exists to remove.
    (("split", "splits", "fold", "folds", "oof", "inner_valid", "early_stopping"),
     ["D4.stage_entry_points", "D5.splitter_operations"]),
    (("predict", "prediction", "predictions", "proba", "probability",
      "calibration", "calibrated", "export", "explain", "shap", "plot"),
     ["D3.public_api"]),
    (("metric", "metrics", "score", "scores", "evaluation", "evaluate"),
     ["D3.public_api", "D3b.metric_operations"]),
]

STRUCTURAL = ("exists", "signature", "type", "isinstance", "member", "enum",
              "attribute", "defined", "raises", "immutable", "frozen", "dataclass")


def claim_populations(text: str) -> tuple[list[str], set[str]]:
    # Tokenised on word boundaries, not on [a-z_]+: keeping the underscores
    # inside a token made `test_fit_isotonic_works` one unmatchable word and
    # every snake_case name unmappable.
    words = set(re.findall(r"[a-z]+", text.lower()))
    names, ops = [], set()
    for nouns, pops in EFFECT_TABLE:
        if words & set(nouns):
            for pn in pops:
                if pn not in names:
                    names.append(pn)
                    ops |= POPS[pn]
    return names, ops


# ------------------------------------------------------------ the population
tests: dict[str, str] = {}  # nodeid-ish key -> name + docstring
for p in sorted((ROOT / "tests").rglob("*.py")):
    rel = p.relative_to(ROOT).as_posix()
    try:
        tree = ast.parse(p.read_text(encoding="utf-8"))
    except SyntaxError:
        continue
    owner: dict[int, str] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef):
            for s in ast.walk(n):
                if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    owner.setdefault(id(s), n.name)
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test"):
            cls = owner.get(id(n))
            key = f"{rel}::{cls}::{n.name}" if cls else f"{rel}::{n.name}"
            tests[key] = f"{n.name} {ast.get_docstring(n) or ''}"

print(f"test functions: {len(tests)}   (plan declares 1803)")
files = len({k.split('::')[0] for k in tests})
print(f"files containing a test: {files}   (plan declares 150)")
print()

# ------------------------------------------------------------------- traces
TRACES: dict[str, set[str]] = {}
for line in (RES / "d6_traces.jsonl").read_text(encoding="utf-8").splitlines():
    d = json.loads(line)
    nodeid = d["nodeid"]
    path, _, rest = nodeid.partition("::")
    parts = [x for x in rest.split("::") if x]
    fn = parts[-1].split("[")[0] if parts else ""
    cls = parts[-2] if len(parts) > 1 else None
    key = f"{path}::{cls}::{fn}" if cls else f"{path}::{fn}"
    TRACES.setdefault(key, set()).update(d["ops"])
print(f"traced test items merged onto {len(TRACES)} declarations")
print()

rows = []
for key, text in tests.items():
    words = set(re.findall(r"[a-z]+", text.lower()))
    trace = TRACES.get(key)
    if words & set(STRUCTURAL) and not (words & {"training", "predict", "split", "metric"}):
        rows.append({"test": key, "verdict": "STRUCTURAL",
                     "why": "claims a structural property, which it observes directly"})
        continue
    names, producers = claim_populations(text)
    if not names:
        rows.append({"test": key, "verdict": "CANNOT-TELL",
                     "why": "the claim matches no effect noun; no relation maps it to the "
                            "operations capable of producing its effect"})
        continue
    if trace is None:
        rows.append({"test": key, "verdict": "CANNOT-TELL",
                     "why": "not present in the traced run (deselected or skipped)"})
        continue
    hit = trace & producers
    rows.append({
        "test": key,
        "verdict": "SOUND" if hit else "CANDIDATE-HOLLOW",
        "why": f"claim -> {names}; traced {len(trace)} ops, producers {len(producers)}, "
               f"intersection {len(hit)}",
        "producers": sorted(names),
        "traced": len(trace),
        "hit": sorted(hit)[:3],
    })

tally: dict[str, int] = {}
for r in rows:
    tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
print("D6 verdicts")
for k in sorted(tally):
    print(f"  {tally[k]:4d}  {k}")
print(f"  {sum(tally.values()):4d}  TOTAL")
print()

with (RES / "d6_rows.jsonl").open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

CTRL = "tests/test_estimators/test_param_behavioral_effect.py::TestSmartParamsBehavior::test_feature_weights_changes_importance"
c = next((r for r in rows if r["test"] == CTRL), None)
# The control trains, so a working trace records lightgbm.train for it and the
# classifier must call it SOUND. Anything else means the trace or the noun map
# is broken, and the run fails below instead of printing a label.
ctrl_ok = (
    c is not None
    and c["verdict"] == "SOUND"
    and "lightgbm.train" in TRACES.get(CTRL, set())
)
print("positive control (trains: must be SOUND with lightgbm.train traced)")
print(f"  {'OK  ' if ctrl_ok else 'FAIL'} "
      f"{c['verdict'] if c else 'ABSENT'} - {c['why'] if c else ''}")
print()
print("resolved population sizes")
for k, v in POPS.items():
    print(f"  {len(v):4d}  {k}")
print()
print("CANDIDATE-HOLLOW, top 25 by trace size:")
ch = sorted((r for r in rows if r["verdict"] == "CANDIDATE-HOLLOW"),
            key=lambda r: -r.get("traced", 0))
for r in ch[:25]:
    print(f"  {r['test']}")
    print(f"      {r['why']}")

if not ctrl_ok:
    raise SystemExit("positive control failed: " + CTRL)
