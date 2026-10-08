Review-kind: review
Review-round: 29
Monitor-mode: relational
Monitor-verdict: CONVERGING
Monitor-carrier: read-only fresh context, inheriting neither maker rationale nor reviewer conclusions
Monitor-evidence: docs/audits/2026-09-defect-discovery/results/pr2_monitor_round2829.md
Monitor-disposition: take-stop-condition
Monitor-rationale: The deliverable improved in round 27 and stayed stable through round 28; the production freeze was confirmed. take-stop-condition is adopted as written -- automatic adversarial rounds stay stopped, and this round is not one, because D14 authorised one acceptance review against criteria written and approved first. The monitor also found that criteria section 8 pre-decided its own reconciliation; section 8 is now conditional.

# PR 2, round 29 — an acceptance review against criteria written in advance

You are reviewing a pull request in a Python machine-learning library. Read the
working tree at the head below. You have **read-only** access; change nothing.

**This is not a defect hunt over the diff.** Twenty-eight adversarial rounds
already ran; every finding was reproduced and fixed. The maintainer has stopped
automatic rounds and authorised **one acceptance review** against completion
criteria that were **written and approved before this prompt existed**.

**Read the criteria document first, and treat it as the contract for this round:**

```
docs/audits/2026-09-defect-discovery/results/pr2_acceptance_criteria.md
```

## Exact head

- Repository: `/home/rem/repos/LizyML`
- Branch `fix/phase3-pr2-fit-params-forwarding`, PR **#278**, draft.
- **Production code is frozen at `6b14b99`.** `git diff 6b14b99 HEAD -- lizyml/`
  is empty. Later commits change documents and rename one test without changing
  an assertion.
- Full suite at the current head: **7710 passed, 230 skipped**. `ruff check`,
  `ruff format --check` and `mypy lizyml/` clean. CI green.

## Authoritative contracts

- `HISTORY.md` **H-0094** (decisions 1-8 and its acceptance criteria),
  **H-0095** — the authoritative part is the section 「契約の確定」, **not** the
  original proposal section that precedes it — and **H-0096**.
- `BLUEPRINT.md` sections 5.3 and 14.4.
- `CLAUDE.md` sections 2 and 3.
- The criteria document above, which maps each acceptance criterion to the test
  that is claimed to establish it.

## The three questions, and nothing else

### (a) The last repair

Does **`6b14b99`** do what it claims? Round 28 found that a regression test built
an unprintable value out of `sys.get_int_max_str_digits()`, which
`PYTHONINTMAXSTRDIGITS=0` disables, so the test could not reach either refusal
helper. The repair replaces it with an object whose `__str__`, `__repr__` and
`__format__` all raise.

```
git show 6b14b99
```

Does the repair remove the dependence on an interpreter setting **without
introducing a dependence of its own**, and does the test still assert what its
name says?

### (b) The criteria-to-evidence table

Section 2 of the criteria document lists every acceptance criterion of H-0094,
H-0095 and H-0096, and names the test that is claimed to establish each one.

**For each row: does the named test actually assert that criterion, at this
head?** Not whether a test with that name exists — whether it asserts the thing
the row claims. Name every row where it does not.

Pay particular attention to:

- rows marked **superseded** — H-0096 revised part of H-0094 decision 6, and the
  37 tolerance cases were **rewritten into refusal cases, not deleted**. Is that
  actually so? A deleted case that the table calls rewritten is a finding.
- H-0096 criterion **1**, which claims all **five** positions refuse a duplicate
  spelling whatever the values are — the four surfaces plus the adapter helper.
- H-0096 criterion **5**, which claims the values leave both the message **and**
  the error context.
- H-0095 criterion **2**, which claims the accepted set is **derived** from
  LightGBM rather than transcribed.

### (c) One bounded interaction check

The three proposals were accepted at different times and reviewed at different
heads, so the accumulated evidence may not cover their **interaction**. Check it
once, end to end:

**For each of the four surfaces** — `model.params`, `fit(params=)`,
`calibration.params`, `tuning.optuna.space` — does the composition hold?

1. **H-0094 forwarding** — a value written at that surface reaches training, and
   outranks the layers it is declared to outrank.
2. **H-0095 normalisation** — it is normalised once, at the entry, and the bytes
   the estimator receives are unchanged by normalising.
3. **H-0096 refusal** — a second spelling of the same parameter at the same
   surface is refused with `CONFIG_INVALID` before anything trains, whatever the
   values are.

Look for a seam where two of the three disagree: an order in which normalisation
runs after the duplicate check and changes what a duplicate means, a surface
where forwarding happens after the gate, a layer where the refusal fires but a
value has already reached a Booster.

## What is out of scope

Section 3 of the criteria document disposes of three known items **explicitly**,
and the maintainer has accepted shipping with them open:

- **#284** — `param_domain` states one boundary in three structural walks (DC3).
- **#285** — a sixth site resolves `seed` / `verbosity` duplicates by picking
  one, where five others refuse (DC4). Unreachable from every surface; measured
  and pinned by a test.
- **#286** — the adapter refusal message does not name the surface, while the
  facade one does.

**Another instance of any of these three is not a finding for this round.** Say
which issue it belongs to and move on.

## Output

Return at most **1200 words**.

1. **(a)**: `satisfied` or `not-satisfied`, with the reason.
2. **(b)**: the rows you checked and, for each, `satisfied` or `not-satisfied`.
   If a row is `not-satisfied`, name the criterion, the test, and what the test
   actually asserts instead.
3. **(c)**: for each of the four surfaces, `holds` or the seam you found.
4. **Anything else you saw** that does not fit (a), (b) or (c) — as a separate,
   clearly labelled section, with the bucket you think it belongs to.

**Do not return `APPROVE` or `REQUEST_CHANGES`.** Acceptance is the maintainer
declaration, not yours. Return per-criterion judgements and let them decide.

**State your bounds** at the end: what you read, what you did not read, what you
executed, and what you did not verify.
