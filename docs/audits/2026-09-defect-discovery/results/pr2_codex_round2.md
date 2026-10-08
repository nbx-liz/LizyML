# PR 2 — Codex review, round 2 (2026-09-07)

Round 1 is in the sibling file. Before this round an **absolute** monitor
returned `DELIVERABLE-FOCUSED` / `continue`
(`results/pr2_monitor_round12.md`), and corrected one fact I had given it:
#277 shipped with the original PR, not with the round-1 remedy.

---

## Verdict

```
VERDICT: REQUEST_CHANGES
```

One blocking finding, inside round 1's remedy, and it is the same silence one
step further out.

### 1 — an accepted alias walks past the refusal

`lizyml/core/_model_factories.py`. The refusal compared **literal names**, and
LightGBM resolves aliases: `max_leaves` *is* `num_leaves`. So `max_leaves`
passed the H-0093 name check (it is an accepted name) and passed the new
managed-name refusal (it is not the literal string `num_leaves`), reached
`lgb.train` beside the smart-resolved `num_leaves`, and LightGBM used the
canonical one. Codex measured it:

```
True  3 [(12, 32), (12, 32), (12, 32)]   ['[num_leaves: 32]']
False 3 [(12, None), (12, None), (12, None)]  ['[num_leaves: 12]']
```

With `auto_num_leaves` active the override was accepted and had no effect —
which is exactly the defect the round-1 remedy was written to stop.
**Defect-class: DC2** (a matcher accepting the same target under another
spelling), presenting as DC1 (the gate reported clean because it did not look at
the right name).

---

## The remedy

**Canonicalise from the library, not from a list.**

- `param_names.py` gains `LGBM_CANONICAL_NAME` (every accepted spelling → the
  canonical parameter) and `accepted_spellings(canonical)`, both derived from
  the same `LGBM_DumpParamAliases` dump H-0093 already treats as the authority.
  `accepted_spellings` raises rather than returning an empty set for a name
  LightGBM does not define: an empty authority makes every check against it pass
  vacuously.
- `SMART_PARAM_TARGETS` keeps declaring **canonical** names, and
  `smart_managed_names` expands each to every spelling the library accepts. Its
  return type becomes `{spelling: (canonical, smart parameter)}`, so the refusal
  can say *which* parameter an alias names.
- The message now reads `'max_leaves', which names 'num_leaves', is resolved
  from the smart parameter 'auto_num_leaves'…`, and the machine-readable context
  carries `name`, `canonical` and `smart_param`.

Measured: the six managed names have **18 accepted spellings** — four aliases
each for `num_leaves`, `min_data_in_leaf` and `feature_contri`, none for
`min_data_in_bin`, `scale_pos_weight` or `feature_pre_filter`.

**The tests are parametrized over the registry, not over a list.** Both
directions, all 18 spellings: refused with zero Boosters when the smart
parameter is active, reaching `lgb.train` unchanged when it is off. A separate
test asserts the population really contains aliases and equals
`accepted_spellings` per name — without it, an `accepted_spellings` that
returned only the canonical name would leave every cell green and the hole
back.

RED verified: replacing the expansion with the literal mapping turns **exactly
the 12 alias cells** red plus the named regression test, and leaves all six
canonical cells green — the shape of the miss itself.

Codex's own probe, re-run after the fix:

```
auto_num_leaves=True   REFUSED: 'max_leaves', which names 'num_leaves', is
                       resolved from the smart parameter 'auto_num_leaves' ...
auto_num_leaves=False  boosters=3  [(12, None), ...]  ['[num_leaves: 12]']
```

---

## Checked and clean (round 2)

Reported by the reviewer, each with what was run:

- 27 passed on the override file (before this remedy; 53 after).
- **Active predicates**: a cross-product comparing `smart_managed_names` against
  both resolvers' actual output — **192 valid combinations matched**, across all
  three tasks, including empty feature weights.
- **AST sensitivity**: injecting `resolved["review_probe"] = 1` into each
  resolver separately — the shipped scan rejected both.
- **Re-entry**: canonical `num_leaves` refused after tuning, on re-fit, and
  after config mutation, each with zero training calls.
- **The deferral**: the same one-trial probe recorded
  `best_model_params={'num_leaves': 39}` while training received 32, and the
  declarations acknowledge the tuning-space gap rather than implying it is
  closed.
- No further DC1–DC7 blocker; `git diff --check` clean; the reviewer changed
  nothing.

## State

Blocking findings per round: **1, 1**. Both were live and both were inside the
thing the previous stage had just built — round 1 in the forwarding, round 2 in
round 1's refusal. Full suite **2246 passed**; `ruff check .`,
`ruff format --check .`, `mypy lizyml/` clean.
