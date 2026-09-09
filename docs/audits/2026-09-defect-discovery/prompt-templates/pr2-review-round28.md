Review-kind: review
Review-round: 28
Monitor-mode: relational
Monitor-verdict: CONVERGING
Monitor-carrier: read-only fresh context, inheriting neither maker rationale nor reviewer conclusions
Monitor-evidence: docs/audits/2026-09-defect-discovery/results/pr2_monitor_round2728.md
Monitor-disposition: continue
Monitor-rationale: The repair is 33 production lines inside the two files the proposal names, adds no module, instrument or document, and leaves param_domain untouched, so the monitor read it as a class-level repair rather than a widening. It set one condition before this round: HISTORY still said the refusal message names the values, which the round-27 finding had made false. That line is corrected. The monitor also warns that the authorship reset is spent, and that a defect in code 5715ee2 wrote would fire condition C with no reset left.

# PR 2, round 28 — the fix for round 27, scoped

You are reviewing **one commit** in a Python machine-learning library. Read the
working tree at the head below. You have read-only access; change nothing.

## What to review

**Commit `5715ee2` only**, and whether it does what it claims. The rest of the
pull request was reviewed in rounds 1 to 27 and is not the surface here.

```
git show 5715ee2
git diff 5715ee2~1..5715ee2
```

## What round 27 found, and what the commit claims

Round 27 returned one finding. The rule introduced by H-0096 decides a refusal
by counting how many spellings of one parameter a layer carries, and never reads
a value — but **reporting** the refusal did read them, by formatting the
supplied dict into the message. A Python `int` above
`sys.get_int_max_str_digits()` digits has no decimal text, so `str()` of one
raises, and the promised `CONFIG_INVALID` became a bare `ValueError`.

The round reported this at `_pop_by_identity` in
`lizyml/estimators/lgbm/adapter.py`, and stated its bound plainly: this is not a
bypass of a shipped surface, because every surface normalises first and
`param_domain` already refuses a value it cannot turn into characters. The
reproduction calls the helper directly.

**The commit claims three things:**

**C1.** Neither refusal depends on a value being printable. Both
`_pop_by_identity` and `check_duplicate_identities` raise `CONFIG_INVALID` for
a duplicate spelling whatever the values are, including a value whose `str()`
raises.

**C2.** The same defect was present at `check_duplicate_identities`, which round
27 did not name, and both are corrected in one change rather than at the
position where it was reported.

**C3.** The values leave the error **context** as well as the message.
`LizyMLError.__repr__` renders its context with `!r`, so a value left there is
one something downstream still tries to print. What remains in message and
context is the list of spellings, which are `str` keys.

## The question

**Does `5715ee2` establish C1 to C3, without introducing a defect of its own?**

Specifically:

1. Is C1 true in the general case rather than for the integer that was
   reported? A value whose rendering raises can be built in more than one way.
2. Is there a **fourth** position where a caller-written parameter value is
   still rendered on the refusal path — in either helper, in what they raise, or
   in anything that handles it?
3. Does removing values from the context break a consumer that read them? The
   context is a public part of `LizyMLError`.
4. Are the refusal messages still specific enough to act on — do they name the
   parameter, the spellings written, and the surface?
5. Did the fix break anything the earlier rounds bought? The regression cases
   from rounds 1 to 27 are in the same files.

## The change under review

- Repository: `/home/rem/repos/LizyML`
- Branch: `fix/phase3-pr2-fit-params-forwarding`, PR **#278** (draft)
- **Head: `259c0ac`**, the current `HEAD`, matching `origin`. The commit
  under review is `5715ee2`; `259c0ac` follows it with a documentation
  correction, a comment, and a test, and is not the surface here.
- Proposals: **H-0094**, **H-0095** (the section titled 契約の確定 is the
  settled contract) and **H-0096** in `HISTORY.md`; `BLUEPRINT.md` §14.4 carries
  the rule. The round-27 record is
  `docs/audits/2026-09-defect-discovery/results/pr2_codex_round27.md`.
- Full suite **7710 passed, 230 skipped**; `ruff check .`,
  `ruff format --check .` and `mypy lizyml/` clean.

## Two things that are settled, and are not findings

- **A refusal of a duplicate spelling is the rule**, including when the two
  values are indistinguishable to LightGBM. A false refusal on a **single**
  spelling is a finding.
- **`param_domain.py` states one boundary in three structural walks** with
  nothing keeping them in step. This is known, recorded, and tracked as issue
  **#284**. Please do not spend the round on it.

## Please tag each finding

- `deliverable-path` — in `lizyml/core/_model_factories.py` or
  `lizyml/estimators/lgbm/adapter.py`;
- `contract` — in H-0096, in the settled-contract section of H-0095, in
  `BLUEPRINT.md` §14.4, or in `CHANGELOG.md`;
- `periphery` — in the tests.

And say whether the finding is **in code this commit wrote** or in code that
predates it.

Return `APPROVE`, or `REQUEST_CHANGES` with any counterexample you have run,
together with the snippet that runs it and its output.

## Bounds

State plainly what you did not check, and do not describe a scan as complete
unless you enumerated its population.
