# PR 2 — Codex review, round 5 (2026-09-07)

Rounds 1-4 are in the sibling files. Before this round a **relational** monitor
over rounds 3-4 returned `CONVERGING` / `continue`
(`results/pr2_monitor_round34.md`) and supplied a stop condition, which the main
context adopted **verbatim and before this verdict was seen**:

> a blocking defect in code round 4 itself wrote — `_pop_by_identity` or
> `check_duplicate_identities` — takes the stop condition, and PR 2 goes to the
> maintainer rather than to a round 6. A finding that is unmasked pre-existing
> behaviour is progress and does not trip it.

---

## Verdict

```
VERDICT: REQUEST_CHANGES
```

One blocking finding, and it landed exactly where the pre-registration said the
loop ends.

### 1 — equal numeric values were refused as conflicting

`lizyml/core/_model_factories.py`. `check_duplicate_identities` compared
`repr(value)`, so `1` and `1.0` read as two different values and a call meaning
one thing twice was refused:

```
{'learning_rate': 1}                  TRAINED
{'eta': 1.0}                          TRAINED
{'learning_rate': 1, 'eta': 1.0}      LizyMLError [CONFIG_INVALID]
only the round-4 check disabled:      TRAINED
```

A gate refusing valid input — the **DC7 direction** — and it disagreed with
`_pop_by_identity`, which compares by equality, so the same call was accepted or
refused depending on which parameter it named.

**Authorship: round-4-authored**, and the reviewer showed its work rather than
asserting it: `git diff 45f0da1..6737ca3 -- lizyml/core/_model_factories.py`
shows the whole function was added in round 4, and disabling only that check
leaves the adapter and the alias merge in place, so no older default masks
either value. That is the distinction round 4 taught the loop to make, and it
was made.

---

## The remedy, and the stop

**Compare by equality, not by printed form.** The same notion `_pop_by_identity`
uses, so the two refusals cannot disagree about what "the same value" means, and
it works for the unhashable values a parameter can take — `feature_contri` is a
list, which a set of values would have raised on.

Tests: equal values under two spellings accepted through a real fit (`1`/`1.0`,
`0.5`/`0.5`); the two refusals asserted to agree on the same inputs, including
`True`/`1`, which is checked at the helper because LightGBM cannot parse a bool
as a learning rate; unhashable equal and unequal lists. RED-verified — restoring
the `repr` comparison fails exactly the two equality tests.

**The loop stops here.** The reviewer said so too, unprompted:

> The adopted stop condition is triggered: fix this defect, then hand PR 2 to
> the maintainer rather than proceeding to round 6.

The finding was fixed rather than left standing — handing over a PR carrying a
reviewer-confirmed defect is worse than handing over a fixed one, and fixing a
confirmed defect is not continuing the loop. Opening round 6 would have been.

---

## Checked and clean (round 5)

- 70 passed on the override file (74 after this remedy).
- **19 special-parameter spellings** passed adapter, export-parameter and
  pickle-roundtrip checks; the default round count matched `_COMMON_DEFAULTS`.
- `_pop_by_identity` probed with `None`, falsy values, equal unhashable values,
  and a parameter with no aliases; `check_duplicate_identities` with an empty
  override and an unknown name; `seed` / `verbosity` single-spelling
  normalisation; **the registry contains no cross-identity spelling overlaps**.
- The deferrals are still explicit in the tree — verified by searching for
  `#277`, `#279`, `#280`, `54/67` and `0/824` across BLUEPRINT, HISTORY and the
  CHANGELOG.
- DC1–DC7 reviewed; no additional blocking finding. `git diff --check` clean;
  the reviewer changed nothing.
- The sandbox refused temporary-file creation, so filesystem export was not
  exercised; reported as a limitation rather than a finding.

## State handed to the maintainer

Blocking findings per round: **1, 1, 2, 2, 1**. Seven findings, every one
reproduced before it was accepted, and every one on the path
`fit(params=)` → `lgb.train`:

| Round | Finding | Authorship |
|---|---|---|
| 1 | Smart resolution overwrote the forwarded override | shipped |
| 2 | The refusal compared literal names | round-1 |
| 3 | The merge kept two spellings; the estimator picked one | shipped |
| 3 | The config surface refuses literal names only | shipped, deferred as #280 |
| 4 | An objective alias skipped the task check | shipped, unmasked by round 3 |
| 4 | Two equal spellings crashed | round-3 |
| 5 | Equal numeric values refused as conflicting | round-4 |

Full suite **2267 passed**; `ruff check .`, `ruff format --check .`,
`mypy lizyml/` clean. The decision is `DECISIONS-PENDING.md` **D7**.
