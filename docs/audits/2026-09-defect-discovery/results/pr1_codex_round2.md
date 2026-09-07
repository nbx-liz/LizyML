# PR 1 — Codex review, round 2 (2026-09-07)

Round 1 is in the sibling file. Before this round an absolute loop monitor
returned `DELIVERABLE-FOCUSED` / `continue`.

---

## Verdict

```
VERDICT: REQUEST_CHANGES
```

Four blocking findings. Each was reproduced before being accepted.

### 1 — `Model.fit(params=...)` is inert, so one coverage claim was vacuous

`model.py:202` calls `_merge_params(provider)` without forwarding `params`.
Measured: fitting with `{"not_a_lightgbm_parameter": 7, "num_leaves": 2}`
succeeded, three `lgb.train` calls occurred, and **neither** override reached
any of them — `num_leaves` stayed at its default of 32.

### 2 — a fourth route: `export_code()` on a legacy artifact

`_model_persistence.py:188` takes codegen params from the restored fitted
adapter; `provider.py:435` returns them unvalidated; the generated `train.py`
hands them to `lgb.train` at `templates.py:124`. A simulated legacy adapter
carrying `not_a_lightgbm_parameter=7` loaded successfully, and the value
appeared in the exported `config.json:lgbm_params`. So "`CFG[...]` is covered by
the runtime gate" was false for exactly the legacy-load path decision 5
deliberately supports.

### 3 — the codegen extractor still had silent-drop shapes

- `_template_sources` selected templates containing the substring `lgb.`, though
  `_is_lgb` claims to accept `lightgbm.*` as well.
- `lgb.Dataset(**kwargs)` was skipped rather than classified unreadable.

Two injected shapes — `lightgbm.train({"not_a_lightgbm_parameter": 1}, None)`
and `lgb.Dataset([], **{"not_a_dataset_keyword": 1})` — produced empty deltas in
**all three** categories, which reads exactly like a clean template.

### 4 — H-0093 still contradicted the implementation

Decisions 4 and 5 had been corrected, but the scope line, the compatibility
section and one acceptance criterion still described a gate firing at
`Model(...)` construction.

### Checked and clean (round 2)

- The config, post-construction-mutation, restored-tuning-result and
  tuning-space routes all reject before `lgb.train`.
- `Model.load()` itself succeeded on a legacy artifact.
- `_merge_params` is used by both training paths; the tuning check runs before
  study construction; no conflicting double validation.
- `BLUEPRINT.md` §14.4 lists both Protocol methods and the constraints.
- Finding 4a's repair is meaningful: propagation paired with an independently
  derived registry-membership assertion.
- The five focused test files: **155 passed**. `git diff --check` clean.
- DC2, DC6, DC7 showed no new blocking issue.

---

## Disposition (main context)

All four upheld; findings 1 and 2 verified by running them first.

### Finding 1 — the claim is wrong, not the code

`fit(params=...)` forwarding is **PR 2's deliverable** (#264): the plan's §4 PR 2
says so, and says its fix belongs on the dict `_merge_params` returns — which is
where this PR already put the check. So the repair here is to stop claiming a
route that does not exist yet.

H-0093 now states plainly that `fit(params=...)` reaches nothing today, quotes
the measurement, and records that PR 2 needs only to wire the argument for the
route to be covered. Fixing the forwarding here would have widened PR 1 past its
own Proposal and pre-empted a user-visible behaviour change that has its own
issue and its own gate.

### Finding 2 — a third call site

The export path now validates the params it is about to write, with its own
surface name (`exported lgbm_params`). This is the same function called a third
time, not a third gate.

It also completes decision 5's asymmetry: a legacy artifact **loads**, but
neither training from it nor exporting a script from it is allowed to proceed
under a name LightGBM would discard. A regression test tampers with the refit
adapter and asserts `export_code()` refuses.

### Finding 3 — the classification is now testable, and tested

Template selection is by **shape**: anything that parses as Python is admitted,
and whether it calls LightGBM is decided by reading it rather than by a
substring. `**kwargs` unpacking is classified unreadable.

More important than either fix: the extractor now takes its sources as an
argument, so its exhaustiveness claim can be exercised directly. Four hostile
shapes are parametrized — both of Codex's, plus params from a call and a name
rebound twice — each asserting the shape lands in a named bucket and that it
does not fall through all three. A negative control asserts a clean template
produces no findings, so an extractor that called everything unreadable would
not pass.

### Finding 4 — five statements corrected

The scope line, the two-call-site paragraph (now three, as a table), the
compatibility bullet, 案 B's rationale, and acceptance criterion (c). A sixth
acceptance criterion was added for the export route.

Gates after the repairs: `ruff check .`, `ruff format --check .`,
`mypy lizyml/` clean; full suite **2132 passed**.
