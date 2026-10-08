Review-kind: review
Review-round: 26
Monitor-mode: relational
Monitor-verdict: CONVERGING
Monitor-carrier: read-only fresh context, inheriting neither maker rationale nor reviewer conclusions
Monitor-evidence: docs/audits/2026-09-defect-discovery/results/pr2_monitor_round2426.md
Monitor-disposition: continue
Monitor-rationale: Blocking counts and authorship recursion are both flat, and the round-25 fix is a class-level repair rather than three patches: the predicates now call the normaliser instead of restating the boundary, and the type axis reads the module. The monitor also returned a defect in its own criterion, which three monitors had written three ways, so one wording is now pinned in D11 and inherited from here. This round is the scoped one D11 chose.

# PR 2, round 26 — the fix for round 25, scoped

You are reviewing **one commit** in a Python machine-learning library. Read the
working tree at the head below. You have read-only access; change nothing.

## What to review

**Commit `65dfdbe` only**, and whether it does what it claims. The rest of the
pull request was reviewed in rounds 1 to 25 and is not the surface here.

```
git show 65dfdbe
git diff 65dfdbe~1..65dfdbe
```

It answers three findings from round 25, recorded in
`docs/audits/2026-09-defect-discovery/results/pr2_codex_round25.md`.

### Finding 1 — both consumers encode UTF-8, and the requirements did not ask

`lightgbm.basic._c_str` encodes UTF-8, and `codegen/artifact_writer.py` opens
`config.json` with `encoding="utf-8"`. A lone surrogate is a `str` of an
accepted type; it passed normalisation, the assertion before training and the
json oracle, and then raised `UnicodeEncodeError` inside each consumer.

**The fix** checks the characters where they are produced:
`_encodable_or_refused` in `lizyml/core/param_domain.py`, reached from
`_written_or_refused` for every scalar and element, from the path conversion,
and for mapping keys.

### Finding 2 — the predicates restated the accepted set instead of asking it

`is_plain` and `is_accepted` accepted any exact `int`, so when round 24 narrowed
the normaliser to values the serialiser can turn into characters, they were left
behind: `10 ** 5000` was accepted by both predicates and by the exit assertion,
and refused by normalisation.

**The fix** defines `is_accepted` through `normalise_value` -- accepted means
normalisation succeeds and the result is what went in -- and `is_plain` as that
minus a mapping. The agreement test now walks the **refused** population too.

### Finding 3 — the population did not cover the type set it claimed

The fixture's numpy type list called itself derived and was twelve types typed
out; the module admits five more on this machine.

**The fix** reads the axis from `NUMPY_SCALAR_TYPES`, builds atoms by dtype kind
rather than by naming types, classifies each refusal **per position** because the
same type can be refused in one and accepted in the other (`longdouble`), and
records in the contract that the accepted population is a finite **sample** of an
infinite domain rather than the domain.

## The question

**Does `65dfdbe` do what it claims, without introducing a defect of its own?**

Specifically:

1. Is each of the three findings actually closed, in the general case rather
   than for the value that was reported?
2. Does the UTF-8 check reach **every** position a character can be produced at?
3. Is `is_accepted` defined through the normaliser correct at the edges --
   `nan`, `-0.0`, `bool` against `int`, a value normalisation leaves alone
   versus one it converts?
4. Do the new refusal reasons classify the whole refused sample, both
   directions, without a reason that nothing reaches or a refusal with no
   reason?
5. Did the fix break anything the earlier rounds bought? The regression cases
   from rounds 1 to 24 are in the same files.

## The change under review

- Repository: `/home/rem/repos/LizyML`
- Branch: `fix/phase3-pr2-fit-params-forwarding`, PR **#278** (draft)
- **Head: the current `HEAD`**; the commit under review is `65dfdbe`.
- Proposals: **H-0094** and **H-0095** in `HISTORY.md`; within H-0095 the section
  titled 契約の確定 is the settled contract, amended by this commit.
- Full suite **9710 passed, 278 skipped**; `ruff check .`,
  `ruff format --check .`, `mypy lizyml/` clean; CI green on thirteen lanes.

## Please tag each finding

- `deliverable-path` -- in `lizyml/core/param_domain.py`;
- `contract` -- in the settled-contract section of H-0095 or in BLUEPRINT §14.4;
- `periphery` -- in the tests.

And say whether the finding is **in code this commit wrote** or in code that
predates it.

Return `APPROVE`, or `REQUEST_CHANGES` with any counterexample you have run,
together with the snippet that runs it and its output.

## Bounds

State plainly what you did not check, and do not describe a scan as complete
unless you enumerated its population.
