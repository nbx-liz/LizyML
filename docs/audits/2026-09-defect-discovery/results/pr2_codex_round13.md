# PR 2 — Codex review round 13 (2026-09-08)

Unscoped, head `363124e`, full suite 2447 passed at the time of review.

```
VERDICT: REQUEST_CHANGES
```

Three blocking findings, all production, all reproduced here before anything was
changed. Plus one non-blocking finding this round **invited**: the prompt asked
the reviewer to break the seam-enumeration claim, and it did.

**None of the three is in code round 12 wrote.** Findings 1 and 3 are
pre-existing readers this PR *exposed* by making aliases and forwarding work.
Finding 2 is in PR-authored code, but the case predates round 12's step — a
length of 2 against a length of 3 refused it before that step existed too.

## Finding 1 — `export_code` loses a custom metric written under an alias (DC1)

`_extract_feval_metadata` (`provider.py:473`) read `adapter.params.get("metric")`
by the **literal** spelling, while `_build_params` reads the same parameter with
`_pop_by_identity`. So the two disagreed about which parameter the caller named.

Reproduced at head:

```
metric         evaluated=['brier']  exported metric='None'  feval=['brier']
metrics        evaluated=['brier']  exported metric='None'  feval=[]
metric_types   evaluated=['brier']  exported metric='None'  feval=[]
```

The fit evaluates Brier correctly under every spelling; the export keeps the
evaluation function only under the canonical one. The reviewer then executed the
generated `train_lgbm` against the exported config and the alias case raised:

```
ValueError: For early stopping, at least one dataset and eval metric is required
```

So the generated code does not merely lose a metric — it does not run.

**Fix.** Read by identity, the same authority `_build_params` uses. Not popped:
this is a read of a dict the caller still owns.

**Its population, enumerated.** `grep` for literal reads of a parameter dict
across `estimators/`, `persistence/`, `codegen/`, `core/`, `training/` returns
four candidates on dicts that can still carry the caller's spelling.
`provider.py:473` was the live one; `adapter.py:234,507` read `params` *after*
`_pop_by_identity` has normalised it, and `smart_params.py:146` reads
`max_depth`, which has no alias. **This is a construct neither seam scan covers**
— not "one dict meets another" but "a user-spelled dict is read under one
spelling."

## Finding 2 — a sequence and its comma-separated text are refused as two (DC7)

Reproduced at head:

```
{feature_contri: [1, 2]}                          -> trained
{feature_penalty: "1,2"}                          -> trained
{feature_contri: [1, 2], feature_penalty: "1,2"}  -> CONFIG_INVALID
the two controls train identical boosters: True
```

The length step compared the **character count** of `"1,2"` with the **element
count** of `[1, 2]`.

**This is the third consecutive round finding the next equivalence class in one
function** — round 11: dtype; round 12: container; round 13: text grammar. That
is the open-grammar shape, and the fix is written to close rather than chase it.

**The authority is read, not guessed.** `lightgbm/basic.py::_param_dict_to_str`
writes `f"{key}={','.join(map(_to_string, val))}"` for **every** `list`, `tuple`,
`set` or 1-D ndarray, whatever the parameter is called, and passes a `str`
through unchanged. So the two forms are one value on the wire, for every
parameter — which is why the step is uniform rather than parameter-aware, and
the test executes the serialiser instead of citing it.

The comparison is **elementwise, not textual**, because the wire form is not
canonical: `[1.0, 2.0]` joins to `"1.0,2.0"` and `[1, 2]` to `"1,2"`, and
LightGBM parses both to the same doubles. Comparing joined strings would refuse
that pair — the same false refusal one formatting step along.

**Stated bound.** Nested grammar is **not** covered: `interaction_constraints`
accepts `[[0, 1], [2]]` and `"[0,1],[2]"`, and reading those needs a parser over
a grammar LightGBM may extend — the open-grammar shape DC1's own note warns
about. Those two are reported as differing, and the case table pins that.

Negative controls, executed: `"1,2"` vs `[5, 6]` and vs `[1, 2, 3]` still differ;
`"auc"` vs `["auc", "logloss"]` still differs. The fix does not reach the "the
same" floor, which would have admitted exactly the DC1 the gate exists for.

## Finding 3 — an accepted early-stopping override loses to the config (DC1)

`adapter.py:223` always adds an early-stopping callback built from
`training.early_stopping.rounds`. Reproduced at head:

```
config rounds= 2  reached lgb.train=[10, 10, 10]  iterations trained=3
config rounds=10  reached lgb.train=[10, 10, 10]  iterations trained=11
```

The override is present in every training call and the config still decides.

**The decisive execution, which chose the fix.** With the callback **off**, the
parameter is not inert either — LightGBM honours it itself, LizyML has built no
validation set, and the run dies with a `CONFIG_INVALID` blaming the metric:

```
callback=off  override={'early_stopping_round': 5}
  -> RAISED [CONFIG_INVALID] No valid eval metric for LightGBM...
```

So there is no reading under which the parameter is safely accepted. This is one
parameter under two names in two places, which decision 6 says to refuse.

**Its population, enumerated and executed.** Every `training.*` control that is
also a native LightGBM parameter, against every spelling:

```
training.early_stopping.rounds -> early_stopping_round
  early_stopping / early_stopping_round / early_stopping_rounds / n_iter_no_change
  all four accepted; the callback still decided
training.seed -> seed
  random_seed / random_state / seed
  all three accepted; the override beat training.seed, silently
```

`seed` fails in the **opposite direction** — the override wins — so a run's
reproducibility control was not the one the config appears to declare. Refusing
beats picking a winner precisely because the two directions disagree and nothing
in the config says which applies. Both are wired, on the merged dict with
`origins`, so the message names the input the caller has to change.

```
Firing rate: 0/916 of configs with early stopping enabled and model.params
Firing rate: 0/928 of configs with training.seed and model.params
```

## Non-blocking — the seam-enumeration claim, broken as invited

The prompt asked the reviewer to name a merge the widened scan still misses. It
named `config/loader.py:167`, `node[last] = _coerce_env_value(value)` — the
environment-override write, invisible because the cursor is called `node` and
the hint-word filter did not reach it. It executed `_apply_env_overrides` to
show the write is live.

Executed here before accepting the classification:

```
env eta=0.5 + file learning_rate=0.001  -> REFUSED (two spellings, two values)
env eta=0.5 + file learning_rate=0.5    -> trained
env learning_rate=0.5                   -> trained at 0.5
```

The reviewer's own bound was right: an enumeration gap, not a defect. The filter
now includes `node`, `cfg` and `config`; candidates 48 → **58**.

**And the claim is retitled.** "The population is enumerated" was asserted twice
and falsified twice within a round of being made — round 12's version omitted the
construct two of its own defects lived in, and this one missed `node`. The
instrument's docstring now says it generates **candidates**, that the table
asserts only what was *executed*, and that treating a scan over an open space as
a closure is the DC5 this run keeps finding in other people's declarations.

## What this round establishes, and what it does not

- **Establishes**, by execution: the three findings are real and fixed; the
  comma form is LightGBM's own uniform wire rule (serialiser executed); the
  `training.*` conflict is a two-instance class failing in opposite directions,
  both closed over every spelling; the env-override seam is covered.
- **Does not establish** that the seam population is closed. It is not, and the
  record now says so.
- **Does not establish** anything about nested sequence grammar; that bound is
  stated in the code and pinned by a case.
- **#280**, **#279**, **#277** remain untouched and open.

Full suite **2474 passed**; `ruff check .`, `ruff format --check .`,
`mypy lizyml/` clean.

Blocking per round: **1, 1, 2, 2, 1, 3, 2, 4, 3, 2, 3, 2, 3.**
