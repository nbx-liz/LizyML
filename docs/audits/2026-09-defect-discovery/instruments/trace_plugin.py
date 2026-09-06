"""One pytest run, two instruments.

D6  — which of the 444 schema operations each test actually executes.
D3 set 5 — for each of the 201 defaulted parameter declarations, every value it
           is bound to over the whole suite.

Both read the same `sys.setprofile` 'call' event, so they share one traversal.
Combining them is a deliberate departure from "one instrument per process": the
profiled suite is the expensive thing, the two consumers touch disjoint output,
and every result is flushed per test so a crash mid-run keeps what ran.

Nothing here writes to the repository.
"""

from __future__ import annotations

import ast
import json
import pathlib
import sys
import threading

ROOT = pathlib.Path("/home/rem/repos/LizyML")
OUT = pathlib.Path("/tmp/lizyml-discovery-plan/results")
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, "/tmp/lizyml-discovery-plan")
from measure_extra import defaulted_param_declarations, operation_schema  # noqa: E402


def _trees() -> dict[pathlib.Path, ast.Module]:
    out = {}
    for p in sorted((ROOT / "lizyml").rglob("*.py")):
        try:
            out[p] = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError:
            pass
    return out


TREES = _trees()
SCHEMA = set(operation_schema(TREES))

# {(relfile, qualname): {param: default_repr_or_None_for_non_literal}}
DECLS: dict[tuple[str, str], dict[str, str | None]] = {}
DECL_ORDER: list[str] = defaulted_param_declarations(TREES)


def _literal_repr(node: ast.expr) -> str | None:
    """repr of a literal default, or None when the default is an expression.

    A non-literal default cannot be compared against an observed value by
    string equality, so those rows are marked and never silently counted as
    "bound at the default".
    """
    try:
        return repr(ast.literal_eval(node))
    except (ValueError, TypeError, SyntaxError):
        return None


for _p, _tree in TREES.items():
    _rel = _p.relative_to(ROOT).as_posix()
    _owner: dict[int, str] = {}
    for _n in ast.walk(_tree):
        if isinstance(_n, ast.ClassDef):
            for _s in ast.walk(_n):
                if isinstance(_s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    _owner.setdefault(id(_s), _n.name)
    for _n in ast.walk(_tree):
        if not isinstance(_n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        _a = _n.args
        _pos = list(zip(_a.args[len(_a.args) - len(_a.defaults):], _a.defaults)) if _a.defaults else []
        _kw = [(k, v) for k, v in zip(_a.kwonlyargs, _a.kw_defaults) if v is not None]
        _cls = _owner.get(id(_n))
        _qual = f"{_cls}.{_n.name}" if _cls else _n.name
        for _arg, _dflt in _pos + _kw:
            if _arg.arg == "self":
                continue
            DECLS.setdefault((_rel, _qual), {})[_arg.arg] = _literal_repr(_dflt)

BINDINGS: dict[str, dict] = {}
_current: set[str] = set()


def _norm_qual(q: str) -> str:
    """Runtime co_qualname -> the AST-style qualname.

    `ast.walk` over a ClassDef reaches functions nested inside its methods, so
    the AST names a helper `C.helper` while the runtime names it
    `C.method.<locals>.helper`. Dropping each `<locals>` segment together with
    the segment before it maps one onto the other; without this every nested
    declaration would record zero invocations and land in CANNOT-TELL for a
    reason that is an artefact of the instrument, not of the code.
    """
    segs = q.split(".")
    out: list[str] = []
    for s in segs:
        if s == "<locals>":
            if out:
                out.pop()
            continue
        out.append(s)
    return ".".join(out)


def _cheap(v: object) -> str:
    if v is None or isinstance(v, (bool, int, float, str, bytes)):
        r = repr(v)
        return r if len(r) <= 60 else r[:57] + "..."
    return f"<{type(v).__name__}>"


def _profile(frame, event, arg):  # noqa: ANN001
    if event != "call":
        return
    code = frame.f_code
    f = code.co_filename
    if not f.startswith(_LIZYML_DIR):
        if f.startswith(_LGB_DIR) and code.co_name == "train":
            _current.add("lightgbm.train")
        return
    p = pathlib.Path(f)
    rel = p.relative_to(ROOT).as_posix()
    qual = _norm_qual(getattr(code, "co_qualname", code.co_name))

    op = rel[:-3].replace("/", ".") + "." + qual
    if op in SCHEMA:
        _current.add(op)

    params = DECLS.get((rel, qual))
    if params:
        loc = frame.f_locals
        for name, dflt in params.items():
            if name not in loc:
                continue
            key = f"{rel}::{qual}({name})"
            rec = BINDINGS.setdefault(
                key,
                {"n": 0, "n_prod": 0, "default": dflt, "vals": [], "vals_prod": [],
                 "truncated": False},
            )
            rec["n"] += 1
            cv = _cheap(loc[name])
            if cv not in rec["vals"]:
                if len(rec["vals"]) < 10:
                    rec["vals"].append(cv)
                else:
                    rec["truncated"] = True
            # Where the call came from decides what the binding proves. A test
            # that reaches into a private method and passes the parameter shows
            # the parameter works; it does not show any caller reaches it
            # through the declared contract. Counting those together made
            # `Model._merge_params(override)` -- this step's own control, and a
            # confirmed DC4 -- come out OK.
            back = frame.f_back
            if back is not None and back.f_code.co_filename.startswith(_LIZYML_DIR):
                rec["n_prod"] += 1
                if cv not in rec["vals_prod"] and len(rec["vals_prod"]) < 10:
                    rec["vals_prod"].append(cv)


import lightgbm as _lgb  # noqa: E402

_LIZYML_DIR = str(ROOT / "lizyml")
_LGB_DIR = str(pathlib.Path(_lgb.__file__).parent)

_traces = open(OUT / "d6_traces.jsonl", "w", encoding="utf-8")  # noqa: SIM115
_n_tests = 0


def pytest_runtest_call(item):
    global _n_tests
    _current.clear()
    sys.setprofile(_profile)
    threading.setprofile(_profile)
    try:
        yield
    finally:
        sys.setprofile(None)
        threading.setprofile(None)
        _traces.write(
            json.dumps({"nodeid": item.nodeid, "ops": sorted(_current)}) + "\n"
        )
        _n_tests += 1
        if _n_tests % 50 == 0:
            _traces.flush()
            _dump_bindings()


pytest_runtest_call.__dict__["hookwrapper"] = True


def _dump_bindings():
    tmp = OUT / "d3set5_bindings.json.tmp"
    tmp.write_text(
        json.dumps(
            {"declarations": len(DECL_ORDER), "observed": BINDINGS},
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    tmp.replace(OUT / "d3set5_bindings.json")


def pytest_sessionfinish(session, exitstatus):  # noqa: ANN001, ARG001
    _traces.flush()
    _traces.close()
    _dump_bindings()
    (OUT / "instrument_meta.json").write_text(
        json.dumps(
            {
                "schema_operations": len(SCHEMA),
                "declarations_from_ast": len(DECL_ORDER),
                "declaration_keys": len(DECLS),
                "tests_traced": _n_tests,
                "exitstatus": exitstatus,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
