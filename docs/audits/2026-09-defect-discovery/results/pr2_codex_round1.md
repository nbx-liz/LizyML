# PR 2 — Codex review, round 1 (2026-09-07)

PR **#278**, `fix/phase3-pr2-fit-params-forwarding`, base `ccae32b`.
Proposal H-0094, issue #264.

---

## Verdict

```
VERDICT: REQUEST_CHANGES
```

One blocking finding. It is the same defect as the one the PR fixes, one level
down, and the maker did not see it.

### 1 — smart resolution replaces a `fit(params=)` override, silently

`lizyml/core/model.py`. Smart parameters resolve **downstream** of
`_merge_params` and the result wins:
`resolved_model = {**resolved_model, **smart_resolved}`. So forwarding the
override made `learning_rate` work and left the managed names exactly as broken
as before. Codex measured it:

```
fit(params={"scale_pos_weight": 10})  -> lgb.train received 0.9354838709677419
fit(params={"num_leaves": 12})        -> lgb.train received 32
fit(params={"min_data_in_leaf": 3})   -> lgb.train received 1
```

and confirmed the same values in the trained booster text. The docstring,
H-0094 and the CHANGELOG all claimed the override was the highest priority.
**Defect-class: DC5** — the declaration exceeded what shipped. (The original
defect is DC4.)

Codex also named the right precedent: `BLUEPRINT.md` §5.3 already requires
`CONFIG_INVALID` for the same collision in the config, so the new input has to
preserve that policy rather than invent a quieter one.

---

## The remedy

**Refuse, do not override.** Accepting a value that is then discarded is the
defect this PR exists to fix; making the override win over smart resolution
would be a different spec, and one nobody proposed.

- `SMART_PARAM_TARGETS` in `smart_params.py` declares the native names each
  smart parameter writes, and `smart_managed_names(smart, task)` returns the
  ones an *active* smart parameter will write. `balanced` claims
  `scale_pos_weight` for binary only: multiclass gets a sample weight, which is
  not a parameter name and cannot collide.
- `EstimatorProvider.smart_managed_param_names` exposes it — a **public
  Protocol change**, recorded in H-0094 and in `BLUEPRINT.md` §14.4.
- `check_smart_managed_overrides` in `_model_factories.py` refuses such a name
  from `fit(params=)`, naming both the parameter and the smart parameter that
  manages it, and how to switch that off.

**Closed against the code, not against a reading of it.** A test walks the
`resolved[<name>] = ...` assignments in `resolve_smart_params` and
`resolve_ratio_params` and fails if the table and the assignments disagree, in
either direction. A second test asserts every smart parameter the provider
declares is classified as writing native names or writing none.

**Both directions executed.** For each of the six managed names: with the smart
parameter active the override is refused with zero Boosters trained; with it
disabled the same override reaches `lgb.train` unchanged. Without the second
half the table could name anything and every refusal would still pass.

RED verified three ways: removing the gate turns all six refusal cells red;
declaring a name the resolvers never write, and writing one the table does not
declare, each turn the scan test red and nothing else.

---

## What this PR does *not* close, and why

The same collision inside a `category: model` tuning space is still silently
replaced. Measured over the 912 configs the shipped suite constructs:

```
Firing rate: 0/824 of configs carrying model.params
Firing rate: 54/67 of configs carrying a category:model tuning space
Firing rate: 0/0 of shipped calls passing fit(params=...) -- no call site exists
```

Executed, not inferred: a four-trial study over a `num_leaves` model dimension
sampled 37/40/47, `lgb.train` received **32** every time, and
`best_model_params` recorded `{'num_leaves': 37}` — a value that never trained
anything.

**54 of 67** model-category search spaces in this repository's own suite are in
that state. Closing it here would fail all 54 and the direction is a maintainer
decision, so it is filed as **#279** with the measurements and the execution
transcript, and `BLUEPRINT.md` §5.3 now carries a per-entry-point table saying
which surfaces the collision rule reaches. H-0094 states the inconsistency
rather than implying the surface is closed.

---

## Checked and clean (round 1)

Reported by the reviewer, each with what was run:

- 134 passed across the four override, parameter-name, search-space and
  calibration-name files.
- **Regression sensitivity**: reverting the forwarding in memory turned the
  override file to 7 failed / 4 passed, including identical booster text.
- **Re-entry and origins**: a real one-trial study followed by two fits, plus
  same-name collisions across all four origins — precedence and restoration
  correct, each winning origin reported, zero training calls on refusal.
- **The calibration declaration**: paired fits returned identical metrics for
  `platt` and `beta`, so the BLUEPRINT 12.2 statement is true of the code.
- **Change Gate**: proposal fields present, proposal committed before the
  implementation, the conditional-evidence exemption justified *for the
  forwarding*. (The remedy above adds a gate, so H-0094 now carries measured
  firing rates as well.)
- No further DC1–DC7 blocker in the diff; `git diff --check` clean; the
  reviewer changed nothing.

`gh` was unauthenticated in the sandbox, so CI was not verified from inside the
review. CI is green on #278 independently (12/12).
