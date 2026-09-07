# PR 1 — Codex review, round 1 (2026-09-07)

**Gate**: independent review of PR 1 (`fix/phase3-pr1-lgbm-parameter-names`,
base `origin/develop` = `1d7c4e2`). Merge gate for this run is Codex APPROVE +
CI green.

---

## Verdict

```
VERDICT: REQUEST_CHANGES
```

Four blocking findings, all reproduced independently before being accepted.

### 1 — `Model.load()` rejected legacy artifacts it promised to load

`_model_persistence.py:266` rebuilds with `cls(config)`, so the construction-time
gate ran on the load path. An artifact whose config carries
`not_a_lightgbm_parameter` produced `CONFIG_INVALID`, contradicting H-0093's own
decision 4.

### 2 — the gate missed every name that arrives after construction

Two routes, both measured:

- A caller keeps its reference to the `LizyMLConfig` (`model.py:130` stores it
  rather than copying). Construct `Model(cfg)`, then add a bad key to
  `cfg.model.params`, then fit: the spy recorded it in **all three**
  `lgb.train` calls.
- `best_model_params` restored from an artifact are installed *after*
  construction (`_model_persistence.py:280`). A legacy artifact followed by a
  re-fit forwarded its invalid restored parameter in three calls.

So the "two surfaces" population the gate claimed was false.

### 3 — the public Protocol specification was not updated

H-0093 declares a public Protocol change, but `BLUEPRINT.md` §14.4's canonical
listing still omitted `accepted_model_param_names()` and `smart_param_names()`.
DC3 on a public API, in the first-ranked document.

### 4 — two promised regression checks were false-clean

- `Booster.params` is an echo of the dict handed to `lgb.train`, not a report of
  what LightGBM parsed. Both `feature_weights` and an invented key are retained
  there. So asserting a value arrives proves nothing about parsing.
- The codegen check counted `lgb.*` calls and never read a name. Replacing a
  valid template key with an invented one still produced `1 passed`.

### Checked and clean (round 1)

- The five changed/new test files: **146 passed**.
- LightGBM 4.6.0 registry queried independently: 140 canonical / 307 with
  aliases; `feature_weights` absent, `feature_contri` present. The real alias
  `eta` was accepted.
- Reverting the emitted key in an isolated copy turned the behavioural E2E test
  and all three task-level name checks red. The fix changes learned behaviour
  and matches §5.3's semantics.
- `test_every_smart_parameter_has_a_case` closes its declared population.
- The 27-cell matrix is the declared cross product.
- The Facade call site respects the DAG: `config/` and `estimators/` are Layer-1
  siblings; Layer 4 may combine them.
- The two `0/N` firing rates are scoped to the right populations, and the
  rejection tests are live positive controls, so zero firings is not DC6.
- No alternate `Model.__new__` path exists.

---

## Disposition (main context)

All four upheld. Findings 1 and 2 have one cause and one repair.

### Findings 1 + 2 — the gate moved to the point of use

Validating at construction was the mistake. A config is mutable and still
referenced by its caller, and restored tuning params arrive later, so no
construction-time check can claim to cover the names that reach the estimator.

The gate is now called where the names are handed over:

| Call site | Covers |
|---|---|
| `_merge_params`, after the merge | config `model.params`, a config mutated post-construction, `best_model_params` restored from an artifact, the `fit()` override |
| `_model_tuning`, before the study | `category: model` search-space dimensions — trial params are drawn from these names, so covering the space covers them |

Both call the same function; this is one gate invoked twice, not two gates that
can disagree. `Model.load()` now succeeds on a legacy artifact and refuses only
when that artifact is used to *train*, which is the right asymmetry: a record
should be readable, but training must not start under a name that does nothing.

Three regression tests pin exactly Codex's probes — post-construction mutation,
restored `best_model_params`, and loading being unblocked — each asserting both
the `CONFIG_INVALID` and that the name never reached `lgb.train`.

### Finding 3 — `BLUEPRINT.md` §14.4

The two methods are added to the canonical listing, with the derivation rule
(derive from the library, never enumerate) and the point-of-use firing rule
stated as constraints rather than left in the Proposal.

### Finding 4 — both checks made real

Measured first, since the claim was about LightGBM's behaviour:

```
not_a_lightgbm_parameter     in Booster.params = True  -> 9
feature_weights              in Booster.params = True  -> [1.0, 1.0]
num_leaves                   in Booster.params = False -> None
```

`Booster.params` is worse than Codex said: it retains invented keys *and* omits
a parameter LightGBM defaulted. `test_param_reaches_booster` now asserts two
things — the value arrives, **and** the name is one LightGBM's registry defines.
The second is what makes the first meaningful, and it is not tautological: the
eight names in the parametrize list are hand-written, and that list is now
checked against the library.

The codegen check reads names instead of counting calls. All three `lgb.train`
sites pass params through a variable, so the extractor resolves a local bound
exactly once to a dict literal, walking each call site in its **innermost**
scope — attributing a call to the module scope judged a local name against the
module's bindings, where the same name is bound three times, and reported a
perfectly readable site as unreadable.

Each site is classified into three exhaustive outcomes, none of them a skip:

| Outcome | Meaning |
|---|---|
| readable | 11 train params + 3 Dataset keywords, checked against the authority |
| from `CFG[...]` | params come from the exported config, covered by the runtime gate |
| unreadable | **test failure** — the template grew a shape the scan cannot follow |

Codex's own probe now fails as it should:

```
AssertionError: the codegen templates hand LightGBM names it does not define:
[('train param', 'not_a_lightgbm_parameter')]
```

Gates after the repairs: `ruff check .`, `ruff format --check .`,
`mypy lizyml/` clean; full suite **2126 passed**.
