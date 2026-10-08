Review-kind: review
Review-round: 24
Monitor-mode: relational
Monitor-verdict: CONVERGING
Monitor-carrier: read-only fresh context, inheriting neither maker rationale nor reviewer conclusions
Monitor-evidence: docs/audits/2026-09-defect-discovery/results/pr2_monitor_round2022.md
Monitor-disposition: take-stop-condition
Monitor-rationale: Both pre-registered falsifiers fired in round 22, and the maintainer chose a scoped round rather than stopping. That round ran through the fresh-checker route because the usual runner was cut off three times, and it returned three findings, all fixed and RED-verified. This round returns to the whole deliverable, and its request is worded as contract verification rather than as an adversarial exercise, because the earlier wording was what the runner would not complete. Adopted in full.

# PR 2, round 24 — the whole change

You are reviewing a pull request in a Python machine-learning library. Read the
working tree at the head below. You have read-only access; change nothing.

## What the change does

Two parts, reviewed together.

1. **`Model.fit(params=...)` reaches the trained model** (issue #264, proposal
   H-0094). The argument was documented and never forwarded. Parameter layers
   -- config defaults, a tuning result, the `fit()` argument, calibration
   parameters -- are now merged by **parameter identity** rather than by
   spelling, because LightGBM resolves aliases (`eta` is `learning_rate`) and
   would otherwise silently keep one of two spellings.

2. **Parameter values are validated and normalised at the four surfaces they
   are written on** (proposal H-0095, `lizyml/core/param_domain.py`). Each
   value is converted once, to a small set of plain Python types, and a value
   outside that set is rejected with `CONFIG_INVALID` before training starts.

## The contract to verify

> **A value the module accepts serialises to the same characters after
> normalisation as before it.** A value for which that cannot be guaranteed is
> rejected at the surface, naming the parameter and the surface.

`lightgbm.basic._param_dict_to_str` is what "serialises to" means here, and the
tests read it directly rather than reimplementing it.

**Is that contract kept, and is the pull request correct and complete as a
whole?**

Return `APPROVE`, or `REQUEST_CHANGES` with any counterexample you have run,
together with the snippet that runs it and its output.

## The change under review

- Repository: `/home/rem/repos/LizyML`
- Branch: `fix/phase3-pr2-fit-params-forwarding`, PR **#278** (draft)
- **Head: the current `HEAD`** — read the working tree.
- Base: `origin/develop`
- Proposals: **H-0094** and **H-0095** in `HISTORY.md`. Issue **#264**.
- Full suite **7439 passed, 256 skipped**; `ruff check .`,
  `ruff format --check .`, `mypy lizyml/` clean; CI green on twelve lanes.

```
git diff origin/develop...HEAD
```

## The nine acceptance criteria — please execute each

1. **Serialisation is preserved** across normalisation, for every value the
   module accepts.
2. **The accepted set is derived** from LightGBM and from numpy rather than
   written down, so that an upgrade widening either is visible here.
3. **A rejection happens at the surface, before training**, and names the
   parameter, the surface, and what is accepted.
4. **All four surfaces normalise and then use the result.** A surface that
   validated and then passed the original value on would be a defect the
   validation cannot see.
5. **The regression cases from earlier review rounds still hold.** They were
   rewritten rather than removed when the design changed, so check that the
   rewrite kept the claim.
6. **`values_differ` answers for every value the surface accepts** and raises
   on none of them.
7. **Issue #283 is deferred explicitly**, with the reason recorded, rather than
   quietly.
8. **The assertion before the training call is reached from every place that
   trains**, with its population derived from the source.
9. **A metric written as a mapping is accepted at the surface and rejected at
   the training call.** `metric={"precision_at_k": {"k": 15}}` is a library
   form the adapter consumes before serialisation, so the two ends have
   different accepted sets deliberately.

## Notes that will save you time

- **The accepted set is narrower than LightGBM's**, on purpose, and each
  narrowing is recorded with a measurement: a `set` (sequence parameters are
  positional and a set has no order), a list nested more than two deep,
  `numpy.timedelta64` and `numpy.datetime64`, and subclasses of numpy scalar
  types and of `ndarray`. **A rejection of a value the library ought to accept
  is a finding worth reporting** — please say when that is what you found,
  since it is a different class from a changed serialisation.
- **Type checks compare with `is` rather than `in`.** Membership in a set or a
  tuple consults `__hash__` and `__eq__`, which for a class come from its
  metaclass; `is` does not.
- **Membership in the numpy set is decided by `numpy.dtype(kind).type is
  kind`**, not by reading a namespace, since a module dictionary is writable.
  The stated bound is that the domain is closed against parameter values, and
  not against a caller who has already replaced part of numpy in the same
  process.
- **The precision boundary depends on the numpy version** — which floats print
  in a form no plain Python number prints is numpy's choice — so the tests
  derive it rather than assert a literal.
- **Deliberately not in this pull request**, each filed with measurements:
  #277, #279, #280, #281, #282, #283.

## Please tag each finding

- `deliverable-path` — in `lizyml/core/model.py`,
  `lizyml/core/_model_factories.py`, `lizyml/core/param_domain.py`,
  `lizyml/core/value_equality.py`, `lizyml/estimators/lgbm/adapter.py`, or
  `lizyml/calibration/isotonic.py`;
- `periphery` — in tests, instruments, or documentation.

And say whether it is a **changed serialisation** or a **rejection of a value
the library ought to accept**.

## Bounds

State plainly what you did not check, and do not describe a scan as complete
unless you enumerated its population.
