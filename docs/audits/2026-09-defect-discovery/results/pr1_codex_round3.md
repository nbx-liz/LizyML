# PR 1 — Codex review, round 3 (2026-09-07)

Rounds 1 and 2 are in the sibling files. Before this round a **relational**
monitor comparing rounds 1–2 returned `CONVERGING` / `redirect`; the
disposition is recorded in `scratchpad/monitor-pr1-r2.md` and was acted on
before the round-3 prompt was written, which declared further extractor
hardening against call shapes absent from `templates.py` out of scope.

The round was steered at the still-unbounded route-coverage claim. It found the
claim was not merely unbounded — the scan behind it was false-clean.

---

## Verdict

```
VERDICT: REQUEST_CHANGES
```

Two blocking findings. Both reproduced before being accepted.

### 1 — the route inventory was blind to an alias, and missed a live ungated route

`_is_lgb()` compared the receiver against a hardcoded `{"lgb", "lightgbm"}`.
`lizyml/calibration/isotonic.py:14` says `import lightgbm as lgbm` and calls
`lgbm.train()` at `:105` with `calibration.params` merged over its defaults, so
the whole module was invisible to the scan while `ESTIMATOR_ROUTES` reported
three routes, all covered.

Codex ran `IsotonicCalibrator({"not_a_lightgbm_parameter": 7})` with a spy on
`lgbm.train` and got `forwarded: 7, calls: 1`. **A real ungated route already in
the tree**, on a user-facing config surface, with the inventory test green.

Confirmed independently with a semantic scan that resolves LightGBM from each
module's own imports (`scratchpad/pr1_scan_probe.py`): six of nine call sites
were invisible to the shipped scan.

### 2 — the inventory scanned the door, not its producers

A new in-package caller of `provider.build_estimator_factory(params={...})`
reaches `LGBMAdapter(params=final_params)` but produced `inventory detections:
[]`. So "the inventory test fails when a route appears" did not hold for
producers, and the argument for shipping no runtime backstop — *every producer
is already gated* — assumed what it was proving.

---

## Disposition

Both accepted. The remedy is in three parts, and the third is the one that
matters: the scan is now checked by execution rather than by reading.

### The calibration surface is gated

Change Gate first, because this adds a fourth `allow` surface. Measured over
the shipped suite at `1d7c4e2` with a pytest plugin recording every
`LizyMLConfig` the suite constructs:

```
configs recorded                : 875
  carrying a calibration block  : 94
  carrying calibration.params   : 3
    the gate would reject       : 0

Firing rate: 0/3 of configs carrying calibration.params
```

distinct keys `{num_boost_round, seed}`, both accepted; all 13
`_ISOTONIC_DEFAULTS` keys accepted; control `not_a_lightgbm_parameter` refused.
Recorded in H-0093 as decision 6 with its own firing-rate line.

Three things the model surface does not need:

- **only LightGBM-backed methods are checked.** `platt` / `beta` fit with numpy
  and scipy; judging their params against LightGBM's registry would refuse
  legitimate configs. Which methods qualify is **scanned**, not declared:
  `test_lgbm_backed_calibrators_matches_the_scan` reads each registered
  calibrator's own imports and fails in both directions.
- **the calibrator's own keys are declared beside the code that pops them**
  (`CALIBRATOR_OWN_PARAM_NAMES` in `isotonic.py`), passed to the shared gate as
  `extra_accepted`, so each surface's exceptions do not accumulate in the gate.
- **`seed` is not among them.** Writing the exception list, I included it
  because `__init__` pops it — and the test that asserts each declared name
  really fails to reach `lgbm.train` caught it: `merged["seed"]` puts it
  straight back, and LightGBM knows the name anyway. The exception would have
  been harmless and false. It is now pinned by a test in both directions.

`PlattCalibrator` ignores `params` entirely — a different defect on a different
route. Recorded in `DECISIONS-PENDING.md`, not changed here.

### The scan resolves aliases and sees producers

`tests/_ast_scan.py` (new) resolves LightGBM's local names from a module's own
imports, covering all three import forms, and keeps module aliases apart from
`from lightgbm import` names because they are called differently. Both scans use
it. Conventional names `{lgb, lightgbm}` remain in the union as a false-fail-only
net for a template fragment whose import lives in another constant.

`_estimator_routes_in_package` now also records
`build_estimator_factory(params=...)` call sites. The population went from a
declared 3 to a scanned **6**: adapter `train` / `Dataset`, the adapter
construction, the single factory producer in `model.py`, and the calibrator's
`train` / `Dataset`.

### The scan's own guarantee is now exercised

The claim "a new route fails this test on the commit that introduces it" was
prose, and it was false twice. `HOSTILE_ROUTE_SHAPES` feeds the scan ten
sources and asserts the exact detection set for each: aliased import,
`from lightgbm import`, renamed `from` import, a submodule-qualified
`lgb.basic.Dataset`, two factory-call shapes, an adapter construction, and three
negative controls (`not_lgb.train`, a factory call without `params=`, and
`self.lgb.train` whose root is not a module).

Writing those cases surfaced a third false-clean that review had not named: the
scan compared only the immediate receiver, so `lgb.basic.Dataset(...)` would
have been dropped with no report. It now walks the attribute chain to its root.
Nothing in `lizyml/` uses that shape today — the only submodule reference is
`lgb.basic.LightGBMError` in an `except` clause — so this is a closed hole, not
a fixed bug.

---

## What did not change

`fit(params=...)` remains inert and remains PR 2's subject (#264). The
extractor was not hardened against further synthetic template shapes, per the
round-3 scoping.
