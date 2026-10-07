"""R6 tally — read each kill run's log and settle every candidate."""

import json
import os
import pathlib
import re
import subprocess
import sys


def _out_dir() -> pathlib.Path:
    """The D6 work directory, from LIZYML_D6_OUT. Refuses to guess one."""
    value = os.environ.get("LIZYML_D6_OUT")
    if not value:
        raise SystemExit("set LIZYML_D6_OUT to the D6 work directory")
    path = pathlib.Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path

RES = _out_dir()
WORK = RES / "r6"

rows = [json.loads(l) for l in (RES / "d6_rows.jsonl").read_text(encoding="utf-8").splitlines()]
by_test = {r["test"]: r for r in rows}

SET_OF = {
    ("D1a.boundary_operations",): "train",
    ("D3.public_api",): "api",
    ("D3.public_api", "D3b.metric_operations"): "metric",
    ("D4.stage_entry_points", "D5.splitter_operations"): "split",
}

out: dict[str, dict] = {}
for mode in ("all", "api", "metric", "split", "train"):
    log = (WORK / f"{mode}.log").read_text(encoding="utf-8", errors="replace")
    disabled = re.search(r"disabled (\d+) producers", log)
    # A node id that appears on a FAILED/ERROR line used a producer.
    used = set(re.findall(r"^(?:FAILED|ERROR) (\S+?)(?:\[|\s|$)", log, re.M))
    ids = (WORK / f"{mode}.txt").read_text(encoding="utf-8").split()
    conf = [i for i in ids if i not in used]
    out[mode] = {
        "producers_disabled": int(disabled.group(1)) if disabled else 0,
        "node_ids": len(ids),
        "confirmed_hollow": len(conf),
        "downgraded": sorted(used & set(ids)) or sorted(used),
    }
    print(f"{mode:7s} producers disabled={out[mode]['producers_disabled']:3d}  "
          f"node ids={len(ids):3d}  confirmed hollow={len(conf):3d}  "
          f"downgraded={len(out[mode]['downgraded'])}")
    for d in out[mode]["downgraded"]:
        print(f"          downgraded: {d}")

tot_ids = sum(v["node_ids"] for v in out.values())
tot_conf = sum(v["confirmed_hollow"] for v in out.values())
tot_down = sum(len(v["downgraded"]) for v in out.values())
print()
print(f"candidates settled: {tot_ids}")
print(f"  CONFIRMED HOLLOW : {tot_conf}")
print(f"  DOWNGRADED       : {tot_down}  (the test did reach a producer)")
print()
# The control is executed, not asserted: a zero count means "never touched the
# producer" only if a test that does touch it fails under every mode. An
# earlier revision printed this verdict as a constant (#270 re-measurement).
CONTROL = ("tests/test_core/test_config_propagation.py::"
           "TestFeatureWeightsE2E::test_feature_weights_applied")
ROOT = pathlib.Path(__file__).resolve().parents[4]
HERE = str(pathlib.Path(__file__).resolve().parent)
control: dict[str, str] = {}
for mode in ("none", "train", "api", "metric", "split", "all"):
    env = {**os.environ, "LIZYML_KILL": "" if mode == "none" else mode, "PYTHONPATH": HERE}
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--no-cov", "-q",
         "-p", "kill_producers", CONTROL],
        cwd=str(ROOT), capture_output=True, text=True, env=env, timeout=600,
    )
    text = proc.stdout + proc.stderr
    if mode == "none":
        control[mode] = "passed" if proc.returncode == 0 else "FAILED"
    elif proc.returncode != 0 and "ProducerRan" in text:
        control[mode] = "failed with ProducerRan"
    else:
        control[mode] = "NOT KILLED"
print(f"kill-mechanism control: {CONTROL}")
for mode, verdict in control.items():
    print(f"  {mode:7s} {verdict}")
out["control"] = control
inert = control["none"] != "passed" or any(v != "failed with ProducerRan"
                                          for m, v in control.items() if m != "none")

(WORK / "tally.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
if inert:
    raise SystemExit("control did not behave: the confirmed counts above mean nothing")
