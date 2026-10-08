"""Sections added to measure.py. Imported by measure.py, not run alone.

Everything here EXECUTES. Nothing in this file describes a measurement it does
not perform, and nothing the plan asserts as measured lives outside it.
"""

from __future__ import annotations

import ast
import itertools
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[4]


# ---------------------------------------------------------------------------
# D3 set 5 — every defaulted parameter declaration, keyed by file + qualname.
# ---------------------------------------------------------------------------
def defaulted_param_declarations(trees: dict[pathlib.Path, ast.Module]) -> list[str]:
    """Every defaulted or keyword-defaulted parameter declared in lizyml/.

    `ast.walk` over the whole module, so declarations nested under
    `if TYPE_CHECKING:` are included — they are declarations, and an AST sweep
    that silently skipped a nesting context would not be exhaustive. (v5's
    helper walked only class bodies and module top level and missed three:
    `ModelPlotsMixin.importance(kind)`, `ModelTuningMixin._merge_params(override)`
    and `ModelTuningMixin._build_train_components(training_overrides)`.)

    No never-passed prefilter is computed. Deciding whether a given call site
    binds a given definition needs name resolution across 16 same-named `split`
    definitions; that is a static analyser, it belongs to D3's execution, and a
    bare-name approximation of it is worse than none.
    """
    out: list[str] = []
    for p, tree in trees.items():
        rel = p.relative_to(ROOT).as_posix()
        owner: dict[int, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for s in ast.walk(node):
                    if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        owner.setdefault(id(s), node.name)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            a = node.args
            d = a.args[len(a.args) - len(a.defaults):] if a.defaults else []
            kw = [k for k, v in zip(a.kwonlyargs, a.kw_defaults) if v is not None]
            cls = owner.get(id(node))
            qual = f"{cls}.{node.name}" if cls else node.name
            for x in list(d) + kw:
                if x.arg != "self":
                    out.append(f"{rel}::{qual}({x.arg})")
    return out


# ---------------------------------------------------------------------------
# D6 axis 2 — the operation schema. Unique, class-qualified, plus the boundary.
# ---------------------------------------------------------------------------
def operation_schema(trees: dict[pathlib.Path, ast.Module]) -> list[str]:
    """Every function and method lizyml/ defines, class-qualified and unique,
    plus the one external operation an assertion can observe.

    This is the observation-position *population*. It is not a call graph: no
    caller/callee edges are resolved here, and the plan does not claim any.
    """
    ops: list[str] = []
    for p, tree in trees.items():
        mod = p.relative_to(ROOT).with_suffix("").as_posix().replace("/", ".")
        owner: dict[int, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for s in ast.walk(node):
                    if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        owner.setdefault(id(s), node.name)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                cls = owner.get(id(node))
                ops.append(f"{mod}.{cls}.{node.name}" if cls else f"{mod}.{node.name}")
    ops.append("lightgbm.train")  # the one external operation
    return sorted(set(ops))


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------
def _frame(task: str, n: int = 200):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    if task == "regression":
        df["y"] = df["f1"] * 2 + rng.normal(scale=0.3, size=n)
    elif task == "binary":
        df["y"] = (df["f1"] + rng.normal(scale=0.3, size=n) > 0).astype(int)
    else:
        df["y"] = rng.integers(0, 3, size=n)
    return df


def _spy():
    import lizyml.estimators.lgbm.adapter as ad

    seen: list[frozenset] = []
    real = ad.lgb.train

    def spy(params, *a, **k):
        seen.append(frozenset(params.keys()))
        return real(params, *a, **k)

    ad.lgb.train = spy
    return seen


def _base_cfg(task: str) -> dict:
    return {
        "config_version": 1,
        "task": task,
        "data": {"target": "y"},
        "split": {"method": "kfold", "n_splits": 2, "shuffle": True, "random_state": 0},
        "model": {"name": "lgbm", "params": {"n_estimators": 3, "verbosity": -1}},
    }


def calibration_key_set_comparison():
    """The measured basis for dropping the calibration axis from D1a."""
    from lizyml.core.model import Model

    seen = _spy()
    df = _frame("binary")
    union = {}
    for calib in (False, True):
        seen.clear()
        cfg = _base_cfg("binary")
        if calib:
            cfg["calibration"] = {"method": "platt"}
        Model(cfg).fit(df)
        union[calib] = frozenset().union(*seen) if seen else frozenset()
    return union[False], union[True]


# D1a axis 3: the tuning surface, as the full declared product.
NAME_CLASSES = {
    "A in LightGBM's table": ("num_leaves", {"type": "int", "low": 4, "high": 8}),
    "B a smart-parameter name": ("num_leaves_ratio", {"type": "float", "low": 0.5, "high": 1.0}),
    "C in neither set": ("not_a_lightgbm_parameter", {"type": "int", "low": 1, "high": 2}),
}
# split_by_category (tuning/search_space.py:174-199) routes on this field, so it
# is a third axis of the accepted configuration surface, not a detail.
CATEGORIES = ["model", "smart", "training"]
TASKS = ["regression", "binary", "multiclass"]


def trace_test(modpath: str, cls: str | None, fn: str) -> set[str]:
    """Every operation a test actually executes, recorded by sys.setprofile.

    Round 6 showed D6's call-graph formulation was simply wrong: `lgb.train`
    does not call `resolve_smart_params`; they are sibling branches of the
    orchestration, so no edge direction relates them. The question D6 actually
    needs is not "is P reachable from O" but "did this test ever RUN an
    operation capable of producing the effect it claims?" — which a trace
    answers directly, mechanically, and without any static analysis.
    """
    import sys
    import threading

    import lightgbm

    seen: set[str] = set()
    lizyml_dir = str(ROOT / "lizyml")
    lgb_dir = str(pathlib.Path(lightgbm.__file__).parent)

    def prof(frame, event, arg):
        if event != "call":
            return
        code = frame.f_code
        f = code.co_filename
        if f.startswith(lizyml_dir):
            rel = (pathlib.Path(f).relative_to(ROOT).with_suffix("")
                   .as_posix().replace("/", "."))
            # co_qualname carries the class, so a traced method matches the
            # class-qualified name `operation_schema` emits. Using co_name here
            # silently reduced the trace to module-level functions only — every
            # method fell out of the schema intersection unmatched, which is the
            # DC1 shape this plan hunts, in its own instrument.
            qual = getattr(code, "co_qualname", code.co_name)
            seen.add(f"{rel}.{qual}")
        elif f.startswith(lgb_dir) and code.co_name == "train":
            seen.add("lightgbm.train")

    sys.path.insert(0, str(ROOT))
    mod = __import__(modpath, fromlist=["x"])
    obj = getattr(mod, cls)() if cls else mod
    sys.setprofile(prof)
    threading.setprofile(prof)
    try:
        getattr(obj, fn)()
    finally:
        sys.setprofile(None)
        threading.setprofile(None)
    return seen


def search_space_matrix(lgb_names: set[str]):
    """Execute every cell of 3 name classes x 3 categories x 3 tasks = 27."""
    from lizyml.core.model import Model

    seen = _spy()
    frames = {t: _frame(t) for t in TASKS}
    rows = []
    for label, category, task in itertools.product(NAME_CLASSES, CATEGORIES, TASKS):
        key, spec = NAME_CLASSES[label]
        cfg = _base_cfg(task)
        cfg["tuning"] = {"optuna": {
            "params": {"n_trials": 1, "direction": "minimize"},
            "space": {key: {**spec, "category": category}},
        }}
        seen.clear()
        try:
            m = Model(cfg)
            m.tune(frames[task])
            m.fit(frames[task])
            keys = frozenset().union(*seen) if seen else frozenset()
            rows.append((label, category, task, key, key in keys, key in lgb_names, None))
        except Exception as e:  # noqa: BLE001
            rows.append((label, category, task, key, False, key in lgb_names,
                         f"{type(e).__name__}: {str(e)[:60]}"))
    return rows
