# PR 2 — Codex review round 12 (2026-09-08)

Unscoped, head `5fd6169`, full suite 2420 passed at the time of review.

```
VERDICT: REQUEST_CHANGES
```

Two blocking findings, both marked `[P2]`, both reproduced here before anything
was changed. A third was found afterwards by the seam enumeration the rounds
11-12 monitor is being asked to carry, and is recorded below with its
measurement.

## Finding 1 — equal sequences falsely refused (DC7)

`values_differ` converted an array with `tolist` and left a tuple alone, so
`np.array([1., 2.])` and `(1., 2.)` reached the printed-form step and differed.
Both are `feature_contri`/`feature_penalty`, one LightGBM parameter, so the
same-layer identity refusal fired on a caller who had written one value.

Reproduced at head, on both surfaces:

```
F1 model.params  ['feature_contri']                      -> trained
F1 model.params  ['feature_penalty']                     -> trained
F1 model.params  ['feature_contri', 'feature_penalty']   -> REFUSED  train_calls= 0
F1 fit(params=)  ['feature_contri', 'feature_penalty']   -> REFUSED  train_calls= 0
```

**Not a regression introduced by round 11.** Before the `tolist` step existed,
the pair also went to the printed forms and also differed. Round 11 narrowed
the class; this is the part it did not reach.

### The accepted decision it contradicts, and why the decision was wrong

`tests/test_core/test_value_equality.py` carried
`("a list and an equal tuple", [1.0, 2.0], (1.0, 2.0), True)` and a test calling
it *"a judgement, not an accident"*, reasoning from Python: `[1.0, 2.0] ==
(1.0, 2.0)` is `False`, and a caller who wrote both spellings wrote two things.

That reasoned about the wrong question. Nothing is chosen silently, because
there is nothing to choose between. Executed:

```
feature_contri        [1.0, 2.0] / (1.0, 2.0) / array([1., 2.]) / array([1, 2])
                      -> [feature_contri: 1,2]        identical trees: True
monotone_constraints  [1, 0] / (1, 0) / array([1, 0])
                      -> [monotone_constraints: 1,0]  identical trees: True
```

The decision is reversed, and the reversal is recorded in H-0094 decision 8
rather than left in a test file. It is the same argument that carried round 11's
third finding: a Python-level distinction refusing input the library treats as
one value.

**Fix.** A normalisation step of its own, before the comparison rather than
inside the `tolist` conversion — a non-text `Sequence` becomes a `list`. Placing
it in `_as_plain_python` would not have been enough: `[1.0, 2.0] == (1.0, 2.0)`
is a perfectly good `bool`, so the truth-value step answers first and the
conversion step is never reached. `str`, `bytes` and `bytearray` are excluded
and the exclusion is pinned by a case.

## Finding 2 — a calibration alias defeated by the default it overrides (DC1)

`IsotonicCalibrator.__init__` merges `{**_ISOTONIC_DEFAULTS, **user}` **by
spelling**, and the defaults are canonical. So `calibration.params = {"eta":
0.5}` passed the name check and the identity check — the caller wrote the
parameter once — and then arrived at `lgbm.train` beside `learning_rate: 0.03`,
which LightGBM preferred.

Reproduced at head:

```
F2 learning_rate  calibrator trained at {'learning_rate': 0.5}
F2 eta            calibrator trained at {'learning_rate': 0.03, 'eta': 0.5}
```

The reviewer stated the bound plainly: *"This is an existing downstream merge
left uncovered on the calibration path this change touches; I am not claiming
this PR introduced that merge."*

**Fix, and why it is where it is.** `lizyml/calibration/` may not import
`lizyml/estimators/`, so the calibrator cannot be taught about aliases.
`canonicalise_calibration_params` rewrites the names in the facade, where the
provider is already reachable, before the dict is handed over. The calibrator's
own keys are excluded and the exclusion is asserted: `num_boost_round` is a
LightGBM alias of `num_iterations`, and renaming it would take the key away from
the code that pops it.

`random_state` was the same defect against the seed the facade supplies, and is
closed by the same rewrite.

### The one behaviour the rewrite changed, and why it was closed rather than left

Executed before and after, against `HEAD` and against the working tree:

```
pre-fix   {'verbose': 1}   -> {'verbose': -1}                  the force held
pre-fix   {'verbosity': 1} -> {'verbosity': 1, 'verbose': -1}  the force lost
post-fix  both             -> {'verbosity': -1}                the force holds
```

The calibrator forces `merged["verbose"] = -1`, and **`verbose` is an alias**;
the canonical is `verbosity`. LightGBM prefers the canonical, so the force was
already defeatable before this PR — by writing the canonical spelling. The
canonicalisation would have made `verbose` behave like `verbosity`, turning an
inconsistency into a uniform hole, so the force was moved to the canonical name
(`merged["verbosity"] = -1`, other spellings popped). `monotone_constraints`
held all along because it was already canonical, which is the same fact read
from the other side.

## Finding 3 — the smart-managed refusal is wired to one surface (DC4 + DC1)

Not from the reviewer. Found by enumerating every place one parameter dict meets
another in `lizyml/`, which is the closed population this class has.

`check_smart_managed_overrides` refuses an override of a name an active smart
parameter is going to overwrite. Its own docstring states the bound: *"It
applies here to the `fit()` override only … the config surface is refused at
parse time for three of the five."* Executed over the whole surface — every
smart parameter, every native name it writes, every spelling LightGBM accepts:

```
population: 18 (smart parameter x native name x spelling)
DEFEATED   12/18   two spellings reach lgb.train; LightGBM keeps the canonical
REPLACED    3/18   the resolver overwrites the user's value in silence
REFUSED     3/18
```

The three refusals are the three canonical names the config schema denies. Every
alias of them passes, and `feature_weights` and `balanced` are not covered on
this surface at all — `scale_pos_weight: 10.0` with `balanced: true` trains at
`0.951`, and `fit(params={"scale_pos_weight": 10.0})` refuses the identical
input.

**Not implemented here, and not because it is small.** The defect is already
known, already recorded precisely in `BLUEPRINT.md` §14.4, and already filed as
**#280**, open for the maintainer's disposition — `config/` cannot import
`estimators/` under the layer rule, so *where* the refusal belongs is a design
decision, which is the substance of that issue. What this round adds to #280 is
the measurement above; what it fixes is the part that is a defect **in this
PR's own code**: the docstring's bound. "Refused at parse time for three of the
five" is true of the smart parameters and false of the surface — it counts the
three canonical names and not the fifteen spellings and targets that pass. A
declaration stated wider than it holds is DC5, and it is the shape this whole PR
is about.

The same class has one more instance on the calibration layer, recorded and not
fixed for the same reason: `calibration.params = {"min_data_in_leaf": 7}` trains
at `ceil(n × 0.01)` under **every** spelling, because `IsotonicCalibrator.fit`
writes that key unconditionally from its always-present
`min_data_in_leaf_ratio` default. Measured identical before and after this
change — the canonical spelling was already losing — so this PR neither
introduced nor moved it.

## The seam enumeration

24 dict-merge expressions over parameter-shaped dicts in `lizyml/`
(`instruments/` scan, AST over `{**a, **b}`, `.update`, `|`, and the two named
helpers). Of those, the ones where two parameter dicts from **different sources**
meet:

| site | resolved by |
|---|---|
| `calibration/isotonic.py:97` | canonicalised upstream — **finding 2** |
| `config/loader.py:108` | one user layer meeting itself. Executed: `model.params {learning_rate: 0.001}` + `model.lgbm.params {eta: 0.5}` is refused; the same value under both is trained |
| `core/_model_factories.py:583` | inside `overlay_params`, identity-aware |
| `core/_model_tuning.py:457,458` | `overlay_params` (round 11) |
| `core/_model_tuning.py:459` | smart layer; no smart name has an alias (pinned) |
| `core/model.py:469,472,503` | `overlay_params` |
| `core/model.py:481` | smart layer |
| `core/model.py:749` | `canonicalise_calibration_params` (new) |
| `estimators/lgbm/adapter.py:163` | the ratio resolver over user params — **finding 3**, which is **#280** |
| `estimators/lgbm/adapter.py:499` | identity-aware (rounds 1-2) |
| `estimators/lgbm/adapter.py:456,458` | `random_state`/`verbose`, subsumed by the dedup above |
| `estimators/lgbm/provider.py:265` | `{**_COMMON_DEFAULTS, **effective_params}`; the only key any resolver reads back is `max_depth`, which has **no alias** — measured, and pinned so it cannot go stale |

The remaining expressions are not parameter merges (a `frozenset` union, two row
builders).

## Firing rates

Measured by replaying every `LizyMLConfig` the suite builds, with the test that
built each hit recorded so this change's own regression tests can be told from
the pre-existing population.

```
Firing rate: 0/22 of pre-existing configs carrying calibration.params change
             the value the calibrator trains at
             (4/24 including this change's own regression tests; the 2
             pre-existing hits are round 11's two-spelling tests, where the
             collapsed alias carries the same value or is refused anyway)
Firing rate: 0/1009 of pre-existing configs carrying model.params were being
             refused by the refusal this change lifts
             (1/1010 including this change's own regression test)
```

## What this round establishes, and what it does not

- **Establishes**, by execution: the two reviewer findings are real and are
  fixed; a list, a tuple and an array of the same numbers are one value to
  LightGBM; the calibration alias now reaches the calibrator on every spelling
  of every default it carries; the calibrator's `verbosity` force holds where it
  never did; the parameter-merge population in `lizyml/` is 12 cross-source
  seams and each one has been executed.
- **Does not establish** that #280 is closed. It is not touched. What changed is
  that the bound in `check_smart_managed_overrides` now states the measured gap
  instead of a narrower one, and the measurement is in BLUEPRINT §14.4.
- **Does not establish** anything about `export` -> `load` -> re-tune beyond
  round 7's evidence plus round 12's own review of it.

Full suite **2445 passed**; `ruff check .`, `ruff format --check .`,
`mypy lizyml/` clean.

Blocking per round: **1, 1, 2, 2, 1, 3, 2, 4, 3, 2, 3, 2 (+1 found here).**
