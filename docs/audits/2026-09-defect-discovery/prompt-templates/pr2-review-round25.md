Review-kind: review
Review-round: 25
Monitor-mode: relational
Monitor-verdict: CONVERGING
Monitor-carrier: read-only fresh context, inheriting neither maker rationale nor reviewer conclusions
Monitor-evidence: docs/audits/2026-09-defect-discovery/results/pr2_monitor_round2425.md
Monitor-disposition: continue
Monitor-rationale: The monitor found the contract rewrite deliverable-directed rather than apparatus: no production change, and the instrument it added produced a fact the hand-written note had missed. It asked that this request say plainly that the contract itself may be rejected, since otherwise the author of a failing artifact would have rewritten its own acceptance criteria. Adopted in full.

# PR 2, round 25 — the settled contract

You are reviewing a pull request in a Python machine-learning library. Read the
working tree at the head below. You have read-only access; change nothing.

## What the change does

Two parts, reviewed together.

1. **`Model.fit(params=...)` reaches the trained model** (issue #264, proposal
   H-0094). The argument was documented and never forwarded. Parameter layers
   -- config defaults, a tuning result, the `fit()` argument, calibration
   parameters -- are merged by **parameter identity** rather than by spelling,
   because LightGBM resolves aliases (`eta` is `learning_rate`) and would
   otherwise silently keep one of two spellings.

2. **Parameter values are normalised at the four surfaces they are written on**
   (proposal H-0095, `lizyml/core/param_domain.py`). Each value is converted
   once, to a small set of plain Python types, and a value outside that set is
   refused with `CONFIG_INVALID` before training starts.

## What is different about this round

The proposal has been **rewritten**, and the question this review answers has
changed with it. Read `HISTORY.md`, proposal **H-0095**, the final section
titled **契約の確定** (the settled contract). It supersedes the proposal table
above it, which is kept only as a record.

That section states three things:

1. **The accepted set**, as position x exact type, with the test that pins each
   row. The block is **generated** from `lizyml/core/param_domain.py` by
   `docs/audits/2026-09-defect-discovery/instruments/param_domain_contract.py`,
   and `--check` compares it against the module.
2. **Every consumer** of a normalised value, and the one requirement each
   brings, with the oracle that runs that requirement over the **whole accepted
   population**.
3. **The declared bounds** -- what this is explicitly not.

## The question

**Under the accepted set in section 1 and the bounds in section 3, does each
requirement in section 2 hold over the whole accepted population?**

And, separately and equally: **is the contract itself right?** You are free to
reject it, and the following are findings, not out of scope:

- a row of the accepted set that no test pins;
- a consumer missing from section 2, or a requirement stated for one that is
  not the requirement that consumer actually has;
- a bound in section 3 that cannot be satisfied, or that claims more than the
  code can deliver;
- a row of section 1 that the generated block contradicts;
- a value the library ought to accept that this refuses.

Please say which of these a finding is.

## The change under review

- Repository: `/home/rem/repos/LizyML`
- Branch: `fix/phase3-pr2-fit-params-forwarding`, PR **#278** (draft)
- **Head: the current `HEAD`** -- read the working tree.
- Base: `origin/develop`
- Proposals: **H-0094** and **H-0095** in `HISTORY.md`. Issue **#264**.
- Full suite **8239 passed, 256 skipped**; `ruff check .`,
  `ruff format --check .`, `mypy lizyml/` clean; CI green on thirteen lanes.

```
git diff origin/develop...HEAD
```

## The requirements to execute

Each of these runs over the accepted population, which the tests build rather
than list (`tests/test_core/test_param_domain.py`, `ACCEPTED_POPULATION`).

1. **Serialisation is preserved** across normalisation. The oracle is
   `lightgbm.basic._param_dict_to_str`, read rather than reimplemented, because
   the text it writes is the whole of what the trainer sees.
2. **The normalised value lands inside the closed set**, and the predicates
   agree with the function that produces it.
3. **Normalising twice is normalising once.** `calibration.params` is
   normalised at two places, so the wiring depends on this.
4. **The normalised value can be written as json.** `export_code` writes the
   same parameters into `config.json`; a path trained happily and made that
   raise `TypeError` afterwards, which is why this is asserted for the whole
   population now rather than for the type that was reported.
5. **`values_differ` answers for every value in the set and raises on none.**
6. **The assertion before the training call is reached from every place that
   trains**, with its population derived from the source rather than listed.
7. **The accepted set is derived** from LightGBM and from numpy rather than
   written down, so that an upgrade widening either is visible.
8. **A refusal happens at the surface, before training**, naming the parameter,
   the surface, and what is accepted.
9. **A metric written as a mapping is accepted at the surface and refused at
   the training call.** `metric={"precision_at_k": {"k": 15}}` is a library
   form the adapter consumes before serialisation, so the two ends have
   different accepted sets deliberately.

## Notes that will save you time

- **The accepted set is narrower than LightGBM's**, on purpose, and each
  narrowing carries a measurement: a `set` (sequence parameters are positional
  and a set has no order), a list nested more than two deep, and subclasses of
  numpy scalar types, of `ndarray` and of `str`.
- **The type set is wider than the value set**, and by position. The generated
  block records which admitted numpy types have no accepted probe value, and
  `longdouble` differs between the two positions.
- **Type checks compare with `is` rather than `in`.** Membership in a set or a
  tuple consults `__hash__` and `__eq__`, which for a class come from its
  metaclass; `is` does not.
- **Membership in the numpy set is decided by `numpy.dtype(kind).type is
  kind`**, not by reading a namespace, since a module dictionary is writable.
- **The generated block names the numpy version and the platform** because
  `longdouble`, `longlong` and `ulonglong` are aliases for C types that resolve
  differently elsewhere.
- **Deliberately not in this pull request**, each filed with measurements:
  #277, #279, #280, #281, #282, #283.

## Please tag each finding

- `deliverable-path` -- in `lizyml/core/model.py`,
  `lizyml/core/_model_factories.py`, `lizyml/core/param_domain.py`,
  `lizyml/core/value_equality.py`, `lizyml/estimators/lgbm/adapter.py`, or
  `lizyml/calibration/isotonic.py`;
- `contract` -- in the settled-contract section itself;
- `periphery` -- in tests, instruments, or other documentation.

Return `APPROVE`, or `REQUEST_CHANGES` with any counterexample you have run,
together with the snippet that runs it and its output.

## Bounds

State plainly what you did not check, and do not describe a scan as complete
unless you enumerated its population.
