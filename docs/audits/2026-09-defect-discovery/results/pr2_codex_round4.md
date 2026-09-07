# PR 2 — Codex review, round 4 (2026-09-07)

Rounds 1-3 are in the sibling files. Before this round a **relational** monitor
over rounds 2-3 returned `CONVERGING` / `continue`
(`results/pr2_monitor_round23.md`) and supplied a tripwire, which the main
context adopted as a pre-registration **before the verdict was seen**: a round-4
finding producing no production change on the fit-params path takes the stop
condition.

---

## Verdict

```
VERDICT: REQUEST_CHANGES
```

Two blocking findings, both on `_build_params`, **both introduced by the round-3
remedy**. The pre-registration is therefore not triggered: these are defects in
the new production code on the fit-params path, not adjacent-surface findings.

### 1 — an objective alias skipped the task-compatibility check

`lizyml/estimators/lgbm/adapter.py`. `_check_objective_compatible` sees only the
literal `objective` key. Round 3's shadow-drop removed the task's default
objective when an alias was present, so the alias became the value that trained
— unvalidated. On a binary task:

```
{'objective': 'regression'}   -> LizyMLError [CONFIG_INVALID]
{'application': 'regression'} -> TRAINED ['[objective: regression]']
```

**Defect-class: DC2 → DC1.** The check was reachable, correct and green; it was
looking at one spelling of a parameter the library resolves by identity.

### 2 — two equal spellings of the objective crashed

`{'objective': 'binary', 'application': 'binary'}` → `KeyError: 'objective'`.
The adapter pops and validates `objective` into `params`; the shadow-drop then
mistook that processed value for a default, deleted it because `application` was
still in the user dict, and the H-0079 invariant read the deleted key.
**Defect-class: DC2.**

Codex attributed both to the stage by reverting only the shadow-drop in memory
and watching them disappear.

---

## The remedy

**The adapter's special handling matches by identity too.** `_pop_by_identity`
removes every spelling of one parameter and returns its single value; it is used
for the three parameters `_build_params` treats specially — `objective`
(validated), `metric` (resolved into native and feval lists) and the boosting
round count (turned into a call argument). Two spellings with the same value are
accepted; **different values are refused** with `CONFIG_INVALID`, because
deciding which applies by dictionary order is the defect this change exists to
remove.

Finding 2 disappears as a consequence rather than by a special case: once every
spelling of `objective` is popped, the shadow-drop cannot see one and cannot
mistake the processed value for a default.

**The same refusal also lives in the facade** (`check_duplicate_identities` on
the `fit(params=)` surface), and that is not redundant — measured. The adapter's
refusal only covers the three specially handled names; an ordinary parameter is
popped by nothing, so both spellings would survive and the estimator would pick
one. RED-verified: removing the facade check leaves the `objective` case passing
and fails **only** the ordinary-parameter case. Without that test the facade
check would have been inert wiring (DC4) sitting inside a PR about inert wiring.

**A consequence worth naming:** the boosting round count now honours any
spelling. Only the literal `n_estimators` was extracted before, so
`num_iterations` stayed in the params dict and reached `lgb.train` beside a
different `num_boost_round` argument. Verified: `n_estimators`,
`num_iterations` and `num_round` all hand `7` to `lgb.train`, and the default
hands `3`.

**The population is derived, not listed.** A test scans `adapter.py` for the
names passed to `_pop_by_identity` and fails if the cases here do not cover
exactly that set — a parameter the adapter starts treating specially, matched
by one literal name, is this defect again.

Codex's own probe after the fix:

```
{'objective': 'regression'}                            CONFIG_INVALID (task mismatch)
{'application': 'regression'}                          CONFIG_INVALID (task mismatch)
{'objective': 'binary', 'application': 'binary'}       TRAINED [objective: binary]
{'objective': 'binary', 'application': 'cross_entropy'} CONFIG_INVALID (ambiguous)
{'application': 'binary'}                              TRAINED [objective: binary]
```

---

## Checked and clean (round 4)

- 59 passed on the override file before this remedy; **70 after**.
- **Merge probes**: `overlay_params` and `_merge_params` over **181 spelling
  pairs**, **27 three-layer combinations**, empty overlay, input preservation,
  unknown-name retention, and provider-fixed → tune → fit metric precedence.
- **Re-fit probe**: `fit(params={"eta": 0.5})` then `fit()` — 0.5, then the
  config's 0.07.
- **Mutation probes**: reverting the facade seam failed three tests, reverting
  the adapter seam failed five; the shipped code passed all eight.
- `git diff --check 31a25a6..HEAD` clean; no repository changes.
- The reviewer's sandbox has no writable temporary directory, so the export test
  could not set up; reported as a setup limitation rather than a finding.

## State

Blocking findings per round: **1, 1, 2, 2**. Rounds 3 and 4 were both on the
production path the deliverable runs through, and round 4's were defects the
round-3 remedy introduced. Full suite **2263 passed**; `ruff check .`,
`ruff format --check .`, `mypy lizyml/` clean.
