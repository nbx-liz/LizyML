## D3 — specification clause ⇔ implementation contract, 1:1

**Population:** 527 rows across six sets, every count matching the plan —
94 config declarations, 95 contract-dataclass fields, 39 public-API members,
74 defaulted constructor knobs, 201 defaulted parameter declarations, 24 skill
files.

**Result:** 527 classified, **0 unclassified**.

| Set | OK | DEAD | UNDOC | DEAD-CANDIDATE | CANNOT-TELL |
|---|---|---|---|---|---|
| config declarations (94) | 93 | 1 | – | – | – |
| contract fields (95) | 95 | – | – | – | – |
| public API (39) | 37 | – | 2 | – | – |
| constructor knobs (74) | 49 | – | 25 | – | – |
| defaulted parameters (201) | 164 | – | – | 5 | 32 |
| skills (24) | 23 | 1 | – | – | – |

**Positive controls: 5 of 6 hit, the sixth disposed of rather than missed.**
`CategoricalEncoder.unseen_policy` and `NativeFeaturePipeline.unseen_policy` →
`UNDOC` (#260); `Model._merge_params(override)` → `DEAD-CANDIDATE` (#264);
`TimeHoldoutInnerValid.__init__(gap)` → `UNDOC` (K-07); `simulate` → spec-side
`DEAD` (#194). The sixth, `ModelTuningMixin._merge_params(override)`, is a
`...`-bodied stub under `if TYPE_CHECKING:` — declared so the mixin type-checks
against the facade method it delegates to, never executed by design. The plan
called the duplicate "a row to dispose of"; that is the disposition.

**Four instrument defects, all found by a control failing:**

- *Wrong population.* Enumerating pydantic `model_fields` at runtime gives 102,
  not 94, because inherited fields are re-counted on every subclass. Switched to
  the AST sweep the plan declares — otherwise the denominator the stop condition
  is checked against would have moved silently.
- *Bare-name reachability.* `TimeHoldoutInnerValid.__init__(gap)` came out `OK`
  because `TimeSeriesConfig.gap` exists — a different gap, on the outer split.
  Reachability is now asked of the knob's own config class, which is what makes
  it `UNDOC`: `TimeHoldoutInnerValidConfig` forbids the key.
- *Caller origin.* Counting all bindings made `Model._merge_params(override)`
  come out `OK`, because a unit test reaches in and passes it. Only bindings from
  a caller inside `lizyml/` count — for **private** declarations. Applying that
  to public ones marked `Model.fit(data)` dead, since a public entry point's
  callers are outside the package by definition.
- *Self-vouching protocol rule.* `KFoldSplitter.split(groups)` ignores `groups`
  and must still accept it, so a parameter another implementation binds is not
  dead. Built from all bindings, that rule let the test-direct call on one
  `_merge_params` vouch for the other — the control went `OK` again. It is now
  built from production bindings and keyed by declaration, so a declaration
  cannot vouch for itself.

### D3-F1 — 25 constructor knobs no Config field can reach

The library makes a policy choice on the user's behalf and provides no way to
change it. Grouped by what they decide:

| Knob | What the default silently decides |
|---|---|
| `CategoricalEncoder.unseen_policy`, `NativeFeaturePipeline.unseen_policy` | an unseen category is replaced by the training mode (**#260**) |
| `TimeHoldoutInnerValid.gap` | whether the early-stopping split gets the outer split's look-ahead guard (K-07; disposition waits on **D2-F1**) |
| `TimeSeriesSplitter`, `PurgedTimeSeriesSplitter`, `GroupTimeSeriesSplitter` × `max_train_size`, `max_test_size` | fold sizing for every time-series method |
| `LGBMAdapter.early_stopping_rounds`, `verbose_eval`, `num_class` | early stopping and logging at the estimator boundary |
| `ECE.n_bins`, `PrecisionAtK.k`, `HuberLoss.delta` | the metric's own parameter — `precision_at_k` cannot be given a `k` |
| `CVTrainer.n_classes`, `ratio_param_resolver`, `collect_raw_scores`; `RefitTrainer.ratio_param_resolver` | trainer wiring |
| `Tuner.progress_callback`, `storage`, `study_name` | tuning-run identity and persistence |
| `LizyMLError.debug_message`, `cause`, `context` | error payload (internal; no user-facing policy) |

`PrecisionAtK.k` is the sharpest after `unseen_policy`: `precision_at_k` is one
of the metrics D3b found mis-optimised, and its `k` is not settable from Config.

**Recommended:** decide per knob — expose it in Config, or state the default as a
written policy in BLUEPRINT. Not all 25 need exposing; all 25 need a decision.

### D3-F2 — five defaulted parameters no caller reaches

`Model._merge_params(override)` (**#264**), `Model.__init__(data)`,
`ModelPlotsMixin.importance_plot(top_n)`,
`ModelPlotsMixin.plot_learning_curve(metrics)`,
`search_space.detect_boundary(threshold)`.

The last three are documented public options that the whole 2051-item suite never
exercises — a test gap rather than a DC4. `Model.__init__(data)` likewise.
`_merge_params(override)` is the confirmed DC4 already filed.

### D3-F3 — one config field, two exports and one skill with no consumer

`EarlyStoppingConfig.inner_valid_explicit` is read nowhere outside
`lizyml/config/` — it is the round-trip explicitness flag H-0069/#203 added, and
its consumer is inside the config package, so `DEAD` here is a scope artefact
worth confirming rather than a defect. `lizyml.__version_tuple__` and
`lizyml.load_config` are exported and named in no specification document.
`.claude/skills/simulate/SKILL.md` has no implementing symbol at all (**#194**).

---

## D4 — task × stage input-propagation audit

**Population:** 13 stage entry points × 3 tasks × 9 inputs = **351 checks**, all
three axes derived from the repository and all three counts matching the plan.
Every `REACHES` is asserted against a real fit, not inferred from a signature.

**Result:** 351 classified — 27 `REACHES`, 42 `ABSENT-BY-DESIGN`,
222 `ABSENT-UNEXPLAINED`, 60 `UNCLASSIFIED`. "No code and no test" stays
`UNCLASSIFIED`; `ABSENT-BY-DESIGN` was given only with a citation.

**Positive control I-02 re-detected, and the task axis splits it correctly:**

| task | `CVTrainer.fit` | `RefitTrainer.fit` |
|---|---|---|
| regression | ABSENT-BY-DESIGN | ABSENT-BY-DESIGN |
| binary | ABSENT-BY-DESIGN | ABSENT-BY-DESIGN |
| **multiclass** | **REACHES** | **ABSENT-UNEXPLAINED** |

The citation is `estimators/lgbm/smart_params.py:79-99`, read rather than
assumed: `balanced` raises `UNSUPPORTED_TASK` for regression, resolves to the
native `scale_pos_weight` model parameter for binary — which travels through the
estimator factory, not as a weight vector — and produces a real `sample_weight`
array only for multiclass. A pass that marked all three alike did not run.

### D4-F1 — `RefitTrainer.fit` accepts four fewer inputs than `CVTrainer.fit`

```
CVTrainer.fit     X  y  groups  sample_weight  time_values  data_fingerprint  run_meta
RefitTrainer.fit  X  y  groups   —              —            —                 —
```

Of the 222 unexplained absences, **22 are material** — another stage in the same
role does accept the input. They concentrate on this pair:
`sample_weight` (I-02, multiclass), `time_values`, `data_fingerprint` and
`run_meta`.

`time_values` is the one to look at next to `sample_weight`: the refit trains the
final model on all data, and a trainer that cannot see the time column cannot
apply a time-ordered inner split to it.

**Recommended:** decide, for each of the four, whether the refit stage should
receive it or whether the absence is policy — and write the policy down. Three of
the four have no citation anywhere today.

---

## D6 — test hollowness sweep

**Population:** **1803** test functions in **150** files, both matching the plan.
Traced by running the whole suite under `sys.setprofile` (2051 items, all
passing) and intersecting each test's executed operations with the 444-member
schema.

**Result:** 1803 classified, **0 unclassified** — 210 `STRUCTURAL`,
385 `SOUND`, 184 `CANDIDATE-HOLLOW`, 1024 `CANNOT-TELL`. The `CANNOT-TELL` count
is large by construction: a claim matching no effect noun has no relation to the
operations that could produce its effect, and the plan declares that gap in §6
rather than absorbing it.

**Positive control re-detected**, exactly as measured during planning:
`test_feature_weights_changes_importance` traces 2 operations, both in
`smart_params`, and `lightgbm.train` is not among them.

**Two instrument corrections.** A missing measured population — the metric
registry D3b enumerates — sent every unit test of `Evaluator` or of a metric to
`CANDIDATE-HOLLOW`, because "metrics" resolved only to `Model`'s public members,
which such a test never calls. This is the same repair the plan already made once
when `D5.splitter_operations` was added. In the other direction, an earlier
version had replaced the plan's `inner_valid` and `early_stopping` nouns with
`inner`, `valid`, `early`, `stopping` on the ground that the word tokenizer can
never match an underscored token — true, and the plan's own choice. Widening them
pulled every config test mentioning "valid" into the split producer set and
manufactured 58 candidates. A noun that cannot match is a declared gap; replacing
it with looser ones is exactly the authorial latitude the effect table exists to
remove.

### D6-F1 — the gate for #261 and #262 stops one step short

`tests/test_estimators/test_param_behavioral_effect.py:107`
`TestBoosterParamPropagation::test_param_reaches_booster`:

```python
adapter = LGBMAdapter(task="regression", params={param_name: param_value})
params, *_ = adapter._build_params()
assert params[param_name] == param_value
```

It is named for the proposition "the param reaches the booster" and it never
builds a booster. It asserts the dict the adapter constructs — one step before
the boundary where #261 and #262 live. Traced: 2 operations, `lightgbm.train`
absent.

This is the gate that should have caught both filed defects.

### D6-F2 — the gate for #264 bypasses the entry point it is about

`tests/test_core/test_train_components.py:102`
`TestMergeParams::test_fit_args_override_tune_best`:

```python
model_params, _ = m._merge_params(_provider_for(m), override={"learning_rate": 0.2})
assert model_params["learning_rate"] == 0.2
```

It calls the private merge directly and asserts the merged dict. `fit()` — the
documented entry point whose `params` argument #264 shows is never forwarded — is
not exercised. The test passes on a build where `fit(params=...)` does nothing,
which is the build that shipped.

### D6-F3 — 184 candidates, of which 25 claim a training effect and never train

Per the defect-class process rule, a known class escaping a gate that hunts it is
a bug **in the gate**, filed separately from the code bug. The 25 that claim a
training effect and never invoke `lightgbm.train` are the K-01 shape and the
sharpest subset; D6-F1 and D6-F2 are the two whose confirmation is complete.

**The remaining candidates are not individually confirmed.** The plan's
confirmation step is to construct the input, run the named operation and observe
the claimed effect — and, where the claim is about a change the code should make,
to mutate that code and check the test still passes. That is 184 experiments and
it was not done. They are enumerated in `results/d6_rows.jsonl` with their traces;
treating them as confirmed would be the DC5 this plan exists to refuse.

### D3 / D4 / D6 dispositions

| # | Action | Carrying artifact |
|---|---|---|
| 1 | Decide each of the 25 unreachable constructor knobs: expose in Config or write the default down (D3-F1) | New Issue with the table |
| 2 | Cover `importance_plot(top_n)`, `plot_learning_curve(metrics)`, `Model.__init__(data)`, `detect_boundary(threshold)` (D3-F2) | Same Issue as 1 — one decision surface |
| 3 | Decide whether `RefitTrainer.fit` should receive `sample_weight`, `time_values`, `data_fingerprint`, `run_meta` (D4-F1) | New Issue; I-02 is the confirmed instance |
| 4 | Make `test_param_reaches_booster` assert against the booster (D6-F1) | New Issue against the gate; blocks nothing, but #261/#262 should not close without it |
| 5 | Make `test_fit_args_override_tune_best` exercise `fit(params=...)` (D6-F2) | Same Issue as 4 — one gate class |
| 6 | Confirm or downgrade the remaining 182 candidates (D6-F3) | This register; not started, and stated as not started |
