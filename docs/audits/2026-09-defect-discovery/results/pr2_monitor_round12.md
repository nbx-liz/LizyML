# PR 2 — loop monitors (2026-09-07)

Both monitors ran read-only in fresh contexts, inheriting neither the maker's
rationale nor the reviewer's conclusions. Archived here because the round
records cited a scratchpad path that does not exist inside the repository — the
relational monitor caught that itself.

---

## Absolute monitor, before round 2 (observed round 1)

```
VERDICT: DELIVERABLE-FOCUSED
RECOMMENDATION: continue
```

**Correction it made to the capsule.** I told it "#277 and #279 were filed
during the remedy". Wrong: the stage boundary is `0396c08`, the fourth
*original* commit, which already carries BLUEPRINT §12.2 and **#277**. Only
**#279** is a round-1 deferral.

**Classification.** The refusal, `SMART_PARAM_TARGETS`, the provider Protocol
method, the AST scan and the two-direction execution tests are all deliverable.
The Protocol method is "not asked for by the reviewer; forced by layering" —
`core/model.py` must stay estimator-agnostic (BLUEPRINT §14.4, H-0051/52/53).

The AST scan is deliverable **narrowly**, for a reason worth keeping: the
execution tests are parametrized *over the table*, so they cannot see a name the
resolvers write that the table omits, and a name the table wrongly declares
passes both execution directions. Both halves of `declared == written` are
uncovered by anything else.

One item flagged **apparatus-leaning**: the classification test and its
one-entry `SMART_PARAMS_THAT_WRITE_NOTHING`. Kept deliberately — it is the
partition shape PR 1's round 5 required of `TRAINING_ENTRY_POINTS`, where a
hand-written set checked against nothing was itself the finding.

**Proportion.** Proportionate: "the 2-line forwarding fix did not deliver the
declared behaviour for 6 of the parameter names; the remedy is where the
deliverable actually landed." It weighed, unprompted, that the 912-config
firing-rate instrument is **not in the diffstat** — the meter was used and
discarded rather than shipped.

---

## Relational monitor, before round 3 (observed rounds 1-2)

```
VERDICT: CONVERGING
RECOMMENDATION: continue
```

**It measured the stages rather than counting findings.** "The flat count
(1, 1) is not the metric."

| Stage | prod | test | spec docs | loop records |
|---|---|---|---|---|
| A original `ccae32b..0396c08` | +25/-3 | +348/-36 | +105/-2 | 0 |
| B round-1 remedy `..48b3780` | +173/-4 | +193/-1 | +53/-3 | +122 |
| C round-2 remedy `..HEAD` | +99/-24 | +82/-12 | +14/-2 | +127 |

Stage C adds **no new module and no new test file**; it touches only files
already in A/B plus `param_names.py`, the pre-existing H-0093 authority. Finding
breadth narrowed too: 6 canonical names → 12 alias cells of 3 of them.

**Why it is not a remedy-of-remedy chain.** The argument is about the *kind* of
fix, not its size:

> Round 1 replaced "nothing checked" with a table closed against the code in
> both directions; round 2 replaced a literal comparison with derivation from
> the library registry, where `accepted_spellings` raises rather than returning
> empty. Each round closed an axis **by derivation**, not by adding the instance
> that was missed. After round 2 there is no "next spelling" to find, because
> the set is no longer enumerated. That is the structural difference from PR 1's
> rounds 5-6, where each fix added one more entry point to a hand-written set.

It also checked that round 2's finding stayed **on** the declared deliverable —
"without reopening the parameter-name boundary H-0093 closed" is exactly what an
alias walking past the refusal reopens — and that the declarations gained
precision without renarrowing: the scope sentence 「適用範囲は `fit(params=)`
のみ」 is unchanged from stage B, and acceptance criterion 4 widened from 6
names × 2 directions to 18 spellings × 2 directions.

**Periphery it weighed and rejected as decisive.** Loop bookkeeping (249 lines)
is nearing parity with total production (297), but those are records under
`docs/audits/` following PR 1's convention, not apparatus in the shipped
artifact. `SMART_PARAMS_THAT_WRITE_NOTHING` did not grow, #279 was pushed out
rather than absorbed, and the firing-rate instrument is still absent from the
diffstat.

---

## Main context's disposition

`continue` for both, and the dangling-path finding is fixed by this file: the
round records now cite `results/pr2_monitor_round12.md` rather than a scratchpad
path that exists only on the machine that ran the loop.
