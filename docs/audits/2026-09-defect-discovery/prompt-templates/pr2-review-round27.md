Review-kind: review
Review-round: 27
Monitor-mode: relational
Monitor-verdict: CONVERGING
Monitor-carrier: read-only fresh context, inheriting neither maker rationale nor reviewer conclusions
Monitor-evidence: docs/audits/2026-09-defect-discovery/results/pr2_monitor_round2627.md
Monitor-disposition: escalate
Monitor-rationale: The monitor found the loop converging but escalated on a record gap, not on the code: H-0096 cited D13 as its decision while D13 ended with four paths and no decision line, so nothing distinguished a maker narrowing the spec to escape its own loop from the principal narrowing it after root cause. The main context wrote that decision into D13, including the two remands that preceded it, and opens this round as a finite question fenced by acceptance criterion 6.

# PR 2, round 27 — a rule that does not read the values

You are reviewing a change to a Python machine-learning library. Read the
working tree at the head below. You have read-only access; change nothing.

## What changed since round 26

Rounds 1 to 26 reviewed a rule that refused one parameter written under two
spellings **when the two values differed**, and allowed the pair through when
they were equal. Deciding "equal" for whatever a caller had written is a
question over an unbounded domain, and it was the subject of the findings in
rounds 18 to 26.

An accepted proposal, **H-0096**, removes that question rather than bounding it
further: a duplicate spelling is now refused whatever the values are. The
comparison module is deleted.

The commits are `3fba763..9d33737` — a proposal, a documentation pass that
re-anchors what referred to the deleted module, and the implementation.

```
git log --oneline 3fba763~1..9d33737
git diff 3fba763~1..9d33737
```

## The contract this round checks

**R1. One parameter, one spelling.** If a single layer names one parameter
under two or more spellings that the estimator resolves to the same parameter,
that layer is refused with `CONFIG_INVALID` before anything trains. The values
are not read. This holds at five places: `model.params`, `fit(params=)`,
`calibration.params`, a restored `tuning best_model_params`, and
`_pop_by_identity` in the estimator adapter.

**R2. A single spelling is untouched.** A parameter written once reaches
`lgb.train` exactly as it did before, alias or not — including the shapes the
deleted comparison used to have opinions about: a sequence, a comma-separated
text form, a numpy scalar, a 1-D array, a tuple, an empty list, `None`.

**R3. The accepted-value set is unchanged.** `lizyml/core/param_domain.py` keeps
the behaviour it had at round 26. Its stated purpose changed — the comparison
was one of its consumers and is gone — but the values it admits and refuses did
not. `tests/test_core/test_param_domain.py` is unchanged apart from prose and
one test that called the deleted module.

**R4. Nothing still declares the deleted module.** No production module imports
or names `lizyml.core.value_equality`.

### Two things that are deliberate, and are not findings

- **A refusal of a duplicate spelling is the rule, not a false refusal** —
  including when the two values are indistinguishable to LightGBM.
  `feature_contri: [1, 2]` beside `feature_penalty: "1,2"` reaches LightGBM as
  the same bytes and is refused anyway. Round 13 filed that pair as a false
  refusal under the previous rule; under R1 it is the intended answer.
  A false refusal on a **single** spelling is a finding, and R2 is where to look
  for one.
- **`param_domain.py` states one boundary in three structural walks**
  (`normalise_value`, `_holds_a_mapping`, `_is_unchanged`) with nothing keeping
  them in step. This is known, is recorded in the proposal, and is tracked as
  **issue #284**. Collapsing them is a separate proposal. Please do not spend
  the round on it.

## The question

**Does the shipped code satisfy R1 to R4?**

Two forms are useful, and both are finite:

1. For every value in the declared accepted population, does each consumer
   requirement in the H-0095 contract still hold, now that one consumer has been
   removed?
2. Is every input outside that population still refused, at the same places?

And for R1 specifically: is there a **layer** — one of the five — where a
duplicate spelling reaches training, or where the refusal reads a value in order
to decide? Reading a value is what R1 removes, so a path that still reads one is
the defect this round is looking for.

For R2: is there a **single-spelling** value that trained before this change and
does not now? The converted tests carry the population that the previous rule
was exercised over; a shape missing from it is worth naming.

## The change under review

- Repository: `/home/rem/repos/LizyML`
- Branch: `fix/phase3-pr2-fit-params-forwarding`, PR **#278** (draft)
- **Head: `9d33737`**, which is the current `HEAD` and matches `origin`.
- Proposals: **H-0094**, **H-0095** (the section titled 契約の確定 is the
  settled contract) and **H-0096** in `HISTORY.md`. `BLUEPRINT.md` §14.4 carries
  the amended rule.
- The evidence H-0096 rests on, each reproducible from
  `docs/audits/2026-09-defect-discovery/instruments/`: the tolerated branch
  fires nowhere in pre-existing code; LightGBM warns on a duplicate whether or
  not the values agree and resolves it by a precedence independent of dictionary
  order; and of nine surveyed systems only the C preprocessor branches on
  agreement, comparing token sequences rather than values.
- Full suite **7707 passed, 230 skipped**; `ruff check .`,
  `ruff format --check .` and `mypy lizyml/` clean. The count moved from 9731
  because the deleted test module collected 2116 cases from a cross-product
  parametrize; 92 tests are net new.

## Please tag each finding

- `deliverable-path` — in `lizyml/core/_model_factories.py`,
  `lizyml/estimators/lgbm/adapter.py`, or `lizyml/core/param_domain.py`;
- `contract` — in H-0096, in the settled-contract section of H-0095, in
  `BLUEPRINT.md` §14.4, or in `CHANGELOG.md`;
- `periphery` — in the tests.

And say whether the finding is **in code these commits wrote** or in code that
predates them.

Return `APPROVE`, or `REQUEST_CHANGES` with any counterexample you have run,
together with the snippet that runs it and its output.

## Bounds

State plainly what you did not check, and do not describe a scan as complete
unless you enumerated its population. A clean result supports "the contract
above holds at the head named above"; it is not a claim about every value a
caller could write.
