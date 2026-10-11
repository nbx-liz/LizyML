"""H-0122 firing rate: how many existing resume/storage tests the new rules refuse.

Measured by execution, for the H-0122 decision record (spec, last line).

* Population N: the test cases of the suite at ``3a36b02`` (the commit before
  H-0122) that call ``Model.tune`` with ``resume=True`` or a ``storage``.
* n: those of them whose ``Model.tune`` call now raises one of the new H-0122
  refusals -- a ``LizyMLError`` raised from ``lizyml/core/_tuning_resume.py``
  or ``lizyml/core/_tuning_plan.py``, the two modules that hold every H-0122
  refusal (the call matrix, the study states, identity, the config space, U2).

Method: ``git archive 3a36b02`` extracts that commit's tree (its ``tests/``,
the probes, and the docs and scripts some old tests read) into ``$TMPDIR``;
its ``lizyml/`` is replaced by a symlink to the current ``lizyml/``. The old suite runs with this file as a pytest plugin
(``-p firing_rate``), which wraps ``Model.tune`` and records, per test, every
call with ``resume=True`` or ``storage`` and how it ended. Whether the old
test itself passes is recorded too but is not the measure: tests that pinned
the old behaviour are expected to fail against the new code.

Then the P1-P7 scenarios of ``design-vs-implementation.md`` (the configs of
``probe_resume.py`` / ``probe_p4b.py``) are driven one by one with the
outcome caught, and the two probe scripts are run as they are.

Usage (from the repository root; CPU heavy, so under the lock)::

    bash docs/audits/2026-09-defect-discovery/instruments/run-exclusive.sh \\
        h0122-firing .venv/bin/python \\
        docs/audits/2026-10-tuning-resume/probes/firing_rate.py

Writes ``firing_rate.txt`` next to this file.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

BASE = "3a36b02"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
NEW_RULE_FILES = ("lizyml/core/_tuning_resume.py", "lizyml/core/_tuning_plan.py")
#: Refusals these modules raise that existed before H-0122 with the same code,
#: only raised elsewhere. Counted apart, not in n:
#: * `storage` without `study_name`: CONFIG_INVALID from `Tuner.__init__`
#:   (H-0072);
#: * `resume=True` with no study and no storage: TUNING_FAILED from
#:   `_validate_tune_inputs` (H-0068; spec §3 row 8 "unchanged").
RELOCATED = {
    (
        "lizyml/core/_tuning_resume.py:precheck_study_args",
        "tune(storage=)",
        "CONFIG_INVALID",
    ),
    ("lizyml/core/_tuning_resume.py:resolve_call_row", None, "TUNING_FAILED"),
}


def _is_relocated(site: str, surface: Any, code: str) -> bool:
    return (site, surface, code) in RELOCATED


def _is_new_rule(site: str, surface: Any, code: str) -> bool:
    in_files = site.split(":")[0] in NEW_RULE_FILES
    return in_files and not _is_relocated(site, surface, code)


# ---------------------------------------------------------------------------
# Plugin mode (loaded by the old suite with `-p firing_rate`)
# ---------------------------------------------------------------------------

_current: dict[str, str | None] = {"nodeid": None}


def _raise_site(exc: BaseException) -> str:
    """``path:function`` of the innermost ``lizyml/`` frame that raised."""
    site = "?"
    for frame in traceback.extract_tb(exc.__traceback__):
        path = frame.filename.replace(os.sep, "/")
        if "/lizyml/" in path:
            site = "lizyml/" + path.split("/lizyml/", 1)[1] + ":" + frame.name
    return site


def _emit(record: dict[str, Any]) -> None:
    with open(os.environ["FIRING_OUT"], "a", encoding="utf-8") as out:
        out.write(json.dumps(record, default=str) + "\n")


def pytest_configure(config: Any) -> None:
    import lizyml
    import tests
    from lizyml.core import _model_tuning
    from lizyml.core.exceptions import LizyMLError

    _emit(
        {
            "kind": "header",
            "lizyml": lizyml.__file__,
            "tests": tests.__file__,
        }
    )
    real = _model_tuning.ModelTuningMixin.tune

    def tune(self: Any, *args: Any, **kwargs: Any) -> Any:
        resume = kwargs.get("resume") is True
        storage = kwargs.get("storage") is not None
        if not (resume or storage):
            return real(self, *args, **kwargs)
        base = {
            "kind": "call",
            "nodeid": _current["nodeid"],
            "resume": resume,
            "storage": storage,
        }
        try:
            result = real(self, *args, **kwargs)
        except LizyMLError as exc:
            site = _raise_site(exc)
            surface = exc.context.get("surface")
            _emit(
                {
                    **base,
                    "outcome": "LizyMLError",
                    "code": exc.code.name,
                    "site": site,
                    "new_rule": _is_new_rule(site, surface, exc.code.name),
                    "relocated": _is_relocated(site, surface, exc.code.name),
                    "surface": exc.context.get("surface"),
                    "state": exc.context.get("state"),
                    "mismatched": exc.context.get("mismatched"),
                    "message": exc.user_message[:160],
                }
            )
            raise
        except BaseException as exc:
            _emit({**base, "outcome": type(exc).__name__, "site": _raise_site(exc)})
            raise
        _emit({**base, "outcome": "ok"})
        return result

    _model_tuning.ModelTuningMixin.tune = tune  # type: ignore[method-assign]


def pytest_runtest_protocol(item: Any, nextitem: Any) -> None:
    _current["nodeid"] = item.nodeid


def pytest_runtest_logreport(report: Any) -> None:
    if report.when == "call" or (report.when == "setup" and report.failed):
        _emit({"kind": "result", "nodeid": report.nodeid, "outcome": report.outcome})


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _extract(dest: Path) -> None:
    """The tree at BASE, with the current ``lizyml/`` in place of its own."""
    import shutil

    archive = subprocess.run(
        ["git", "archive", BASE], cwd=REPO, capture_output=True, check=True
    )
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive.stdout, check=True)
    shutil.rmtree(dest / "lizyml")
    (dest / "lizyml").symlink_to(REPO / "lizyml", target_is_directory=True)


def _run_old_suite(dest: Path, out: Path, paths: list[str]) -> int:
    # cwd is the extracted tree: `tests` must resolve to the old tests, never
    # to the repository's (the plugin asserts which files were imported).
    env = dict(os.environ, PYTHONPATH=str(HERE), FIRING_OUT=str(out))
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *paths,
            "-p",
            "firing_rate",
            "-p",
            "no:cacheprovider",
            "-q",
            "-m",
            "not slow",
        ],
        cwd=dest,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    tail = done.stdout.strip().splitlines()[-1:] or ["<no output>"]
    print(f"old suite exit={done.returncode}: {tail[0]}")
    # 0 / 1 mean the tests ran (some may fail, as expected). Anything else --
    # interrupted, a collection error, a usage error -- means the population
    # was not observed, and must not read as "nothing refused" (DC1).
    if done.returncode not in (0, 1):
        raise SystemExit(
            f"old suite did not run (exit {done.returncode}):\n"
            + "\n".join(done.stdout.splitlines()[-40:])
            + "\n"
            + done.stderr[-2000:]
        )
    return done.returncode


def _summarise(out: Path, dest: Path) -> list[str]:
    records = [json.loads(line) for line in out.read_text().splitlines()]
    header = next(r for r in records if r["kind"] == "header")
    lines = [
        f"imports: lizyml={header['lizyml']} tests={header['tests']}",
    ]
    # The old tests, the current lizyml/ (DC1: a wrong import would measure
    # the old code, or the new tests).
    assert Path(header["lizyml"]).resolve().is_relative_to(REPO / "lizyml"), header
    assert Path(header["tests"]).resolve().is_relative_to(dest.resolve()), header
    calls = [r for r in records if r["kind"] == "call"]
    results = {r["nodeid"]: r["outcome"] for r in records if r["kind"] == "result"}
    by_test: dict[str, list[dict[str, Any]]] = {}
    for call in calls:
        by_test.setdefault(str(call["nodeid"]), []).append(call)
    refused = {
        node: [c for c in cs if c.get("new_rule")]
        for node, cs in by_test.items()
        if any(c.get("new_rule") for c in cs)
    }
    other_errors = {
        node: [c for c in cs if c["outcome"] != "ok" and not c.get("new_rule")]
        for node, cs in by_test.items()
    }
    relocated = sorted(
        node for node, cs in by_test.items() if any(c.get("relocated") for c in cs)
    )
    kinds: dict[str, int] = {}
    for cs in refused.values():
        for key in sorted(
            {
                f"{c['site'].split(':')[1]} "
                f"({c.get('state') or c.get('surface')}"
                f"{' ' + str(c['mismatched']) if c.get('mismatched') else ''})"
                for c in cs
            }
        ):
            kinds[key] = kinds.get(key, 0) + 1
    old_outcome: dict[str, int] = {}
    for node in refused:
        outcome = results.get(node, "?")
        old_outcome[outcome] = old_outcome.get(outcome, 0) + 1
    n_total, n_refused = len(by_test), len(refused)
    if n_total == 0:  # no observed population is not a rate of 0 (DC1)
        raise SystemExit("no Model.tune call with resume=True or storage observed")
    lines += [
        "",
        f"population N = {n_total} test cases calling Model.tune with resume=True "
        "or storage (setup-time calls are attributed to the test being set up)",
        f"refused n = {n_refused}",
        f"Firing rate: {n_refused}/{n_total} of test cases at {BASE} that call "
        "Model.tune with resume=True or storage (old suite run against the "
        "current lizyml/ with a Model.tune-wrapping pytest plugin; a refusal is "
        "a new-rule LizyMLError raised from _tuning_resume.py or _tuning_plan.py)",
        "",
        "Refusal kinds (test cases per kind; one test can have several):",
        *[f"- {k}: {v}" for k, v in sorted(kinds.items())],
        f"Old test outcome of the refused cases: {old_outcome}",
        f"Relocated pre-H-0122 refusals (same code as before H-0122), not in n: "
        f"{len(relocated)} {relocated}",
        "",
        "Per test case (old test outcome against the current code; "
        "'refusals' = new-rule refusals, 'other' = other errors):",
    ]
    for node in sorted(by_test):
        cs = by_test[node]
        tag = "REFUSED" if node in refused else "passes "
        lines.append(f"- [{tag}] {node}  (test {results.get(node, '?')})")
        for c in refused.get(node, []):
            detail = c.get("state") or c.get("mismatched") or c.get("surface")
            lines.append(f"    refusal: {c['code']} at {c['site']} ({detail})")
            lines.append(f"      {c['message']}")
        for c in other_errors.get(node, []):
            lines.append(
                f"    other: {c['outcome']} {c.get('code', '')} at {c.get('site')}"
            )
    return lines


# ---------------------------------------------------------------------------
# P1-P7, driven one by one (configs of probe_resume.py / probe_p4b.py)
# ---------------------------------------------------------------------------

SPACE = {
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
    "n_estimators": {"type": "int", "low": 5, "high": 20},
}
RATIO_SPACE = {
    "num_leaves_ratio": {"type": "float", "low": 0.5, "high": 1.0, "category": "smart"}
}


def _cfg(
    metric: str = "rmse",
    space: dict[str, Any] | None = None,
    auto_nl: bool = True,
    n_trials: int = 4,
) -> dict[str, Any]:
    return {
        "config_version": 1,
        "task": "regression",
        "data": {"target": "target"},
        "model": {"name": "lgbm", "auto_num_leaves": auto_nl},
        "split": {"method": "kfold", "n_splits": 3},
        "evaluation": {"metrics": [metric]},
        "training": {"early_stopping": {"enabled": False}},
        "tuning": {
            "optuna": {
                "params": {"n_trials": n_trials},
                "space": space if space is not None else SPACE,
                "space_mode": "replace",
            }
        },
    }


def _outcome(step: Callable[[], Any]) -> str:
    from lizyml.core.exceptions import LizyMLError

    try:
        step()
    except LizyMLError as exc:
        ctx = exc.context
        detail = ctx.get("state") or ctx.get("mismatched") or ctx.get("surface")
        return f"{exc.code.name} at {_raise_site(exc)} ({detail})"
    return "ran"


def _scenarios() -> list[tuple[str, str, str]]:
    """``(scenario, outcome now, expected by the spec)`` for P1-P7."""
    from lizyml import Model
    from tests._helpers import make_regression_df

    d1 = make_regression_df(n=150, seed=0)
    d2 = make_regression_df(n=150, seed=1)
    d2["target"] = d2["target"] * 100.0
    tmp = tempfile.mkdtemp()
    rows: list[tuple[str, str, str]] = []

    m = Model(_cfg())
    m.tune(data=d1)
    rows.append(
        (
            "P1 in-process resume after a data change",
            _outcome(lambda: m.tune(data=d2, resume=True, expand_boundary=False)),
            "CONFIG_INVALID, identity (data.*) -- §4.3",
        )
    )

    m2 = Model(_cfg("rmse"))
    m2.tune(data=d1)
    m2._cfg.evaluation.metrics = ["mae"]
    rows.append(
        (
            "P2 in-process resume after rmse -> mae",
            _outcome(lambda: m2.tune(data=d1, resume=True, expand_boundary=False)),
            "CONFIG_INVALID, identity (metric) -- §4.3",
        )
    )

    m3 = Model(_cfg())
    m3.tune(data=d1)
    assert m3._cfg.tuning is not None
    m3._cfg.tuning.optuna.space = {
        "learning_rate": {"type": "float", "low": 0.2, "high": 0.25}
    }
    rows.append(
        (
            "P3 resume after editing the config space",
            _outcome(lambda: m3.tune(data=d1, resume=True, expand_boundary=False)),
            "CONFIG_INVALID, config space -- §5.1",
        )
    )

    url = f"sqlite:///{tmp}/s.db"
    Model(_cfg("rmse")).tune(data=d1, storage=url, study_name="job")
    mb = Model(_cfg("mae", space=RATIO_SPACE))
    rows.append(
        (
            "P4 new Model, other metric + space, resume=False onto the stored study",
            _outcome(lambda: mb.tune(data=d1, storage=url, study_name="job")),
            "CONFIG_INVALID, study already exists -- §3 row 3",
        )
    )

    url_b = f"sqlite:///{tmp}/p4b.db"
    Model(_cfg("mae", n_trials=3)).tune(data=d1, storage=url_b, study_name="job")
    mb2 = Model(_cfg("rmse", space=RATIO_SPACE, n_trials=3))
    rows.append(
        (
            "P4b reattach a mae study under rmse + other space, resume=False",
            _outcome(lambda: mb2.tune(data=d1, storage=url_b, study_name="job")),
            "CONFIG_INVALID, study already exists -- §3 row 3",
        )
    )
    mb3 = Model(_cfg("rmse", space=RATIO_SPACE, n_trials=3))
    rows.append(
        (
            "P4b' the same with resume=True",
            _outcome(
                lambda: mb3.tune(data=d1, storage=url_b, study_name="job", resume=True)
            ),
            "CONFIG_INVALID, config space or identity -- §5.1 / §4.3",
        )
    )

    m5 = Model(_cfg())
    m5.tune(data=d1)
    url2 = f"sqlite:///{tmp}/s2.db"
    rows.append(
        (
            "P5 resume=True with another storage on an in-memory study",
            _outcome(
                lambda: m5.tune(
                    data=d1,
                    resume=True,
                    storage=url2,
                    study_name="other",
                    expand_boundary=False,
                )
            ),
            "CONFIG_INVALID, not the held study -- §3 row 5",
        )
    )

    m6 = Model(_cfg())
    m6.tune(data=d1)
    m6.fit(data=d1)
    loaded = Model.load(m6.export(Path(tempfile.mkdtemp()) / "m"))
    rows.append(
        (
            "P6 Model.load, then resume=True without storage",
            _outcome(lambda: loaded.tune(data=d1, resume=True)),
            "TUNING_FAILED (unchanged) -- §3 row 8, §7.5",
        )
    )

    sp = {**RATIO_SPACE, "n_estimators": {"type": "int", "low": 5, "high": 20}}
    m7 = Model(_cfg(space=sp, auto_nl=True))
    m7.tune(data=d1)
    m7._cfg.model.auto_num_leaves = False
    rows.append(
        (
            "P7 auto_num_leaves true -> false, then resume",
            _outcome(lambda: m7.tune(data=d1, resume=True, expand_boundary=False)),
            "CONFIG_INVALID, identity (config.model) -- §4.3",
        )
    )
    return rows


def _run_probe_scripts() -> list[str]:
    lines = []
    for name in ("probe_resume.py", "probe_p4b.py"):
        done = subprocess.run(
            [sys.executable, str(HERE / name)],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        printed = [ln for ln in done.stdout.splitlines() if ln.startswith("== P")]
        last_error = [
            ln
            for ln in done.stderr.splitlines()
            if "LizyMLError" in ln or "Error" in ln
        ][-1:]
        lines.append(
            f"- {name}: exit={done.returncode}; scenarios started: "
            f"{[p.strip('= ').split(':')[0] for p in printed]}; "
            f"stopped by: {last_error[0][:200] if last_error else '-'}"
        )
    return lines


def main(smoke: list[str]) -> None:
    """Measure; with *smoke* paths, run only those old tests and print."""
    import logging
    import shutil
    import warnings

    work = Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "h0122-firing"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    _extract(work)
    out = work / "firing.jsonl"
    _run_old_suite(work, out, smoke or ["tests"])
    if smoke:
        print("\n".join(_summarise(out, work)))
        return

    def git(*args: str) -> str:
        done = subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
        )
        return done.stdout.strip()

    head = git("rev-parse", "--short", "HEAD")
    dirty = git("status", "--porcelain", "--", "lizyml")
    lines = [
        f"H-0122 firing rate, measured against lizyml/ of the working tree "
        f"(old tests from {BASE})",
        f"lizyml/: HEAD {head}, "
        + ("no uncommitted changes" if not dirty else f"uncommitted: {dirty!r}"),
        "",
        *_summarise(out, work),
    ]

    sys.path.insert(0, str(REPO))
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    import optuna

    optuna.logging.set_verbosity(optuna.logging.ERROR)
    lines += ["", "P1-P7 (design-vs-implementation.md), driven one by one:", ""]
    lines += ["| Scenario | Now | Spec |", "|---|---|---|"]
    lines += [f"| {s} | {o} | {e} |" for s, o, e in _scenarios()]
    lines += ["", "The probe scripts as they are:", *_run_probe_scripts()]
    text = "\n".join(lines) + "\n"
    (HERE / "firing_rate.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:])
