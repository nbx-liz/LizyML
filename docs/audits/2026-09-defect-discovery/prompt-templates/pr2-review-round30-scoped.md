Review-kind: review
Review-round: 30
Monitor-mode: relational
Monitor-verdict: CONVERGING
Monitor-carrier: read-only fresh context, inheriting neither maker rationale nor reviewer conclusions
Monitor-evidence: docs/audits/2026-09-defect-discovery/results/pr2_monitor_round2930.md
Monitor-disposition: continue
Monitor-rationale: The monitor found the deliverable stable across rounds 28-29 with its evidence improved, the production delta from the reviewed head still one file, and the excluded issues still excluded. It corrected one thing: section 6 prescribes filing a non-blocking B3 and not re-reviewing a B4, so fixing them in-PR is an authorised exception to that route, not its execution. Recorded. Its condition is that this round stay inside the elected commits and generate no further round automatically.

# PR 2, round 30 — the two repairs the maintainer elected, scoped

You are reviewing **three commits** in a Python machine-learning library. Read the
working tree at the head below. You have **read-only** access; change nothing.

**Scope is these commits and nothing else.** The rest of the pull request was
reviewed in rounds 1 to 29, and round 29 was an acceptance review against criteria
written in advance. Do not review the wider diff, and do not reopen anything those
rounds settled.

```
git show 22b11b3     # tests only, no production change
git show 6928cbf     # HISTORY, BLUEPRINT, instruments, measurement
git show 77e02a8     # the only production change
git show 9267c9a     # its tests
```

## Exact head

- Repository: `/home/rem/repos/LizyML`
- Branch `fix/phase3-pr2-fit-params-forwarding` at **`9267c9a`**. Working tree clean.
- `git diff 6b14b99 9267c9a -- lizyml/` is **one file**: `lizyml/tuning/search_space.py`.
- Full suite **7721 passed, 230 skipped**. `ruff check`, `ruff format --check` and
  `mypy lizyml/` clean.

## Why these commits exist

Round 29 returned two findings that the completion criteria classified as
**non-blocking**. The maintainer directed that both be fixed inside this pull request
anyway. That is an authorised exception, recorded in
`docs/audits/2026-09-defect-discovery/results/pr2_acceptance_criteria.md` section 1
and in `docs/audits/2026-09-defect-discovery/DECISIONS-PENDING.md` under D14.

## What the commits claim

**C1 — `22b11b3`, one test per identity merge seam (#288).** H-0094 decision 5 makes
merging resolve by identity at four seams and decision 7 adds the trial overlay as a
fifth. Round 29 found one test covering two of them. The claim is that each seam now
has a test writing an alias at that layer, and that **the seam population is derived
from the syntax tree** rather than listed, so a sixth seam cannot be silently
uncovered.

One bound is asserted inside the tests: the provider fixed seam asserts on the merged
dict rather than on the recorded `lgb.train` call, because that layer carries only
`metric` and `first_metric_only` and the adapter pops every spelling of `metric` by
identity and writes the canonical one back.

**C2 — `6928cbf` + `77e02a8` + `9267c9a`, the choices gate (#287).** A value sampled
from a categorical search dimension is overlaid onto the trial params **without
passing an entrance normaliser**, so it reached the exit assertion in the adapter as
the type it was written as. The gate admitted `(NoneType, bool, int, float, str)` by
`isinstance`, and exactly two numpy scalar types subclass a Python scalar:
`np.float64` and `np.str_`. Those two failed every trial, and the user was told
`TUNING_FAILED: All tuning trials failed. Check parameter ranges.`

The claim is that judging the type by **identity** closes that and nothing else: the
space still refuses every numpy scalar, the three types already refused stay refused
in the same shape, and a plain choice still tunes.

The proposal explicitly does **not** claim to make the search space consistent with
the four normalisation surfaces. That question is left open as #287.

## The questions

1. **Do C1 and C2 establish what they claim, at this head?**

2. **Does either introduce a defect of its own?** In particular:
   - Is the seam derivation actually closed? It reads `overlay_params` calls from two
     named modules. What would it miss — an overlay written another way, a third
     module, an expression rather than a name?
   - Is `any(type(val) is allowed for allowed in _ALLOWED_CHOICE_TYPES)` the right
     test, and is the accompanying claim true that `type(val) in (...)` would be a
     hash-and-equality search a caller can influence?
   - `_ALLOWED_CHOICE_TYPES` and `param_domain`'s accepted set are two declarations
     that overlap. The docstring claims they guard **different** boundaries and are
     therefore not one boundary declared twice (BLUEPRINT 14.4, round 25). Is that
     claim true, or is this a second copy that will drift?
   - The firing rate is recorded as 0/54 over the shipped suite. Is that the right
     population for the claim it supports, and is the claim stated within its bound?

3. **Does the refusal message now say something false?** It was changed because
   "Each choice must be a scalar (str, int, float, bool, or None)" contradicted itself
   for a value that *is* a `float`.

## What is out of scope

- The wider pull request, and anything rounds 1-29 settled.
- **#284, #285, #286** — dispositioned as out of scope with the maintainer accepting
  them open.
- **#287's remaining question** — whether `choices` should accept numpy the way the
  four surfaces do. Deliberately open. Saying the inconsistency exists is not a
  finding; it is recorded in `HISTORY.md` and `BLUEPRINT.md` section 11.3.

## Output

Return at most **900 words**.

1. **C1**: `established` or `not-established`, with the reason.
2. **C2**: `established` or `not-established`, with the reason.
3. **Defects introduced**, if any, each with a reproduction.
4. **Bounds**: what you read, what you executed, what you did not verify.

Return `APPROVE` or `REQUEST_CHANGES` for **these commits only** — this verdict is
about the elected repairs, not about the pull request, whose acceptance is the
maintainer's declaration.
