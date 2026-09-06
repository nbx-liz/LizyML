"""R6 — confirm D6's hollow candidates by disabling their producers.

The plan's confirmation is behavioural: construct the input, run the named
operation, observe whether the claimed effect occurs -- and where the claim is
about a change the code should make, mutate that code and observe that the test
still passes. Doing that 182 times by hand is not feasible, but the candidates
partition into five producer sets, and one mutation per set settles every
candidate in it at once.

The mutation is the sharpest form available: **make the producer raise.** A test
that claims an effect on training, and still passes when `lightgbm.train` raises
on every call, provably never trained. There is no weaker reading of that.

Selected by the `LIZYML_KILL` environment variable, so one pytest run per set:

  train    lightgbm.train
  api      Model's public members
  metric   lizyml.metrics.* and lizyml.evaluation.*
  split    the 13 stage entry points and the 8 concrete splitters' split()
  all      the union

Run as:  LIZYML_KILL=train pytest <the candidate node ids> -p kill_producers

Every kill is applied by patching the attribute on the live module, so nothing
under /home/rem/repos/LizyML is modified.
"""

from __future__ import annotations

import ast
import os
import pathlib

ROOT = pathlib.Path("/home/rem/repos/LizyML")
MODE = os.environ.get("LIZYML_KILL", "")


class ProducerRan(RuntimeError):
    """Raised in place of a producer, so a test that needed it cannot pass."""


def _kill(obj: object, name: str, label: str) -> None:
    def boom(*a: object, **k: object):
        raise ProducerRan(f"producer disabled for the hollowness check: {label}")

    try:
        setattr(obj, name, boom)
    except Exception:  # noqa: BLE001, S110
        pass


def _stage_and_splitter_targets() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for p in sorted((ROOT / "lizyml").rglob("*.py")):
        rel = p.relative_to(ROOT).as_posix()
        stage = rel.startswith(("lizyml/training/", "lizyml/calibration/"))
        splitter = "/splitters/" in rel
        if not (stage or splitter):
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"))
        mod = rel[:-3].replace("/", ".")
        for n in tree.body:
            if not isinstance(n, ast.ClassDef) or n.name.startswith("_"):
                continue
            if splitter and n.name == "BaseSplitter":
                continue
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name in (
                        "fit", "split", "run"):
                    out.append((f"{mod}.{n.name}", m.name))
                    break
    return out


def pytest_configure(config) -> None:  # noqa: ANN001, ARG001
    if not MODE:
        return
    killed: list[str] = []

    if MODE in ("train", "all"):
        import lightgbm

        _kill(lightgbm, "train", "lightgbm.train")
        killed.append("lightgbm.train")
        for modname in ("lizyml.estimators.lgbm.adapter",):
            try:
                mod = __import__(modname, fromlist=["x"])
                _kill(mod.lgb, "train", "lightgbm.train (adapter alias)")
            except Exception:  # noqa: BLE001, S110
                pass

    if MODE in ("api", "all"):
        from lizyml.core.model import Model

        for name in [n for n in dir(Model) if not n.startswith("_")]:
            attr = getattr(Model, name, None)
            if callable(attr):
                _kill(Model, name, f"Model.{name}")
                killed.append(f"Model.{name}")

    if MODE in ("metric", "all"):
        for pkgname in ("lizyml.metrics", "lizyml.evaluation"):
            base = ROOT / pkgname.replace(".", "/")
            for p in sorted(base.rglob("*.py")):
                modname = p.relative_to(ROOT).as_posix()[:-3].replace("/", ".")
                try:
                    mod = __import__(modname, fromlist=["x"])
                except Exception:  # noqa: BLE001
                    continue
                for name in dir(mod):
                    obj = getattr(mod, name, None)
                    if callable(obj) and getattr(obj, "__module__", "") == modname:
                        if isinstance(obj, type):
                            for meth in ("compute", "__call__"):
                                if hasattr(obj, meth):
                                    _kill(obj, meth, f"{modname}.{name}.{meth}")
                        else:
                            _kill(mod, name, f"{modname}.{name}")
                        killed.append(f"{modname}.{name}")

    if MODE in ("split", "all"):
        for qual, meth in _stage_and_splitter_targets():
            modname, cls = qual.rsplit(".", 1)
            try:
                mod = __import__(modname, fromlist=["x"])
                _kill(getattr(mod, cls), meth, f"{qual}.{meth}")
                killed.append(f"{qual}.{meth}")
            except Exception:  # noqa: BLE001, S110
                pass

    print(f"\n[kill_producers] MODE={MODE!r}: disabled {len(killed)} producers")
