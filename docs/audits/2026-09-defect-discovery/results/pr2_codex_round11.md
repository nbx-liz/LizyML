# PR 2 — Codex review, round 11 (unscoped, 2026-09-08)

**The first round since round 7 that was not narrowed to the previous round's
remedies**, and the round the whole run turned on. Scoped by the rounds 9-10
relational monitor (`results/pr2_monitor_round910.md`, **`DRIFTING` /
`redirect`**), which refused to answer whether round 10's clean production
result meant the deliverable was finished, and said why:

> Rounds 6, 8, 9 and 10 were each scoped to the previous round's remedies. The
> one exception is round 7 — **and it found production defects.** A round aimed
> at freshly written apparatus finds apparatus defects at whatever rate new code
> carries them; it yields "no production defect" whether or not the deliverable
> is finished. **Round 10's result is mechanically produced by its scope.**

It named the one observation that would settle it: one round over the deliverable
path, unrestricted, asked the deliverable question. This is that round.

## Verdict

```
VERDICT: REQUEST_CHANGES
```

**Three findings, all in production.** Two of them on the merge path itself —
the code the monitor had noted was *"reviewed once since it last changed and not
flagged, which is not the same as declared clean."* All three reproduced here
before being fixed.

### 1 — tuning evaluated different parameters from the ones it selected (DC1)

`_model_tuning.py`, the trial merge. Decision 5 made three seams identity-aware
and left the fourth spelling-based. With a config setting `learning_rate` and a
search dimension named `eta`:

```
best: {'eta': 0.5}
tune: ['[learning_rate: 0.001]', '[learning_rate: 0.001]']
fit:  ['[learning_rate: 0.5]',   '[learning_rate: 0.5]', ...]
```

The trials trained at the **config's** value, the study recorded the **trial's**,
and the fit afterwards used the recorded one. **Tuning selected a model it had
never evaluated, and the best score it reported belonged to a different model.**

### 2 — the same-layer refusal had no caller for `model.params` (DC1, DC4)

Decision 6 declares that one parameter written twice under two spellings with
different values is refused. The call existed for `fit(params=)` only, so a
config carrying both `learning_rate: 0.001` and `eta: 0.5` sent **both** to
`lgb.train`, which kept the canonical one in silence:

```
[(0.001, 0.5), (0.001, 0.5), (0.001, 0.5)]   # learning_rate, eta
[learning_rate: 0.001]                        # what the booster trained on
```

Declared, implemented, and unwired for the layer most callers use.

### 3 — equal arrays under two spellings were refused (DC7)

Round 8 removed the elementwise step and **wrote the cost into the docstring**:
arrays with equal numbers under different dtypes print differently and are
reported as differing. Round 11 executed that cost on the production entrypoint:

```
integer array          trained
float array            trained
equal arrays together  ErrorCode.CONFIG_INVALID
```

Two values LightGBM accepts individually, naming one parameter twice, refused.
**Acknowledging a cost in a docstring does not satisfy the requirement not to
refuse valid input.**

---

## The remedy

**Finding 1 — the fourth seam.** `overlay_params` applied in the same order
(base → provider fixed → trial). The regression test asserts all three at once:
the trials train at the recorded value, `best_model_params` records it, and the
subsequent fit reproduces it. This touches `_model_tuning.py`, **outside the
PR's diff** — but the PR created the asymmetry by making one side identity-aware,
so it is this PR's to close. It is not #279, which is the smart-parameter
overwrite; this is an ordinary alias collision.

**Finding 2 — one call, in the facade.** `check_duplicate_identities(provider,
model_params, surface="model.params")` before the overlays. It lives there rather
than in the schema because the alias table is in `estimators/`, which `config/`
may not import. This changes what an existing config does, so the Change Gate's
measurement was taken rather than estimated:

```
Firing rate: 0/811 of pre-existing configs carrying model.params
(observed at the check's call site over every config this repository's suite
constructs; 1 of 812 fires, and that one is the regression test added with it)
```

Both directions are tested: two spellings with different values are refused
before any Booster is trained; two spellings of the *same* value still train.

**Finding 3 — convert, then ask the same question again.** A step between the
truth conversion and the printed forms: if both sides have `tolist`, convert and
re-run the **same guarded** `bool(a == b)`. That is a documented conversion
followed by the ordinary question — not a rule about what iterating an arbitrary
object yields, which is the hypothesis round 8 deleted after three refutations.

It also fixed a false *acceptance* nobody had asked about: two arrays whose
printed forms summarised the difference away were reported as the same, and now
differ. Both stated-cost cases in the table flipped, and a `DataFrame` pair was
added as the case that still reaches the printed-form fallback — the remaining
cost, with something reaching it.

RED verified per finding, with the reviewer's own scripts.

## Checked and clean (round 11, from the reviewer)

Substantial, because a clean report on the shipped path is what this round
existed to obtain:

- **Layer priority**: all four combinations executed across regression, binary
  and multiclass with real one-trial tuning. Config `0.01`, tuning `0.2`, fit
  alias override `0.5` produced the expected value in both CV folds and the
  refit. Input config dictionaries unchanged.
- **Calibration**: `platt`, `beta` and `isotonic` executed; all base-model
  training calls retained the fit override, and isotonic's three calibrator calls
  retained their own learning rate.
- **Aliases and boundaries**: all **307** installed-registry spellings checked
  against their canonical identity, and **614** prefixed/suffixed unknown names
  refused. Real fits under every spelling of objective, metric, boosting rounds,
  seed, verbosity and learning rate.
- **Refusal controls**: unknown names, conflicting fit overrides, equal scalar
  duplicates, active smart-managed refusals, and acceptance after disabling the
  smart parameter.
- **The original wiring defect**: Booster comparisons and `lgb.train`
  observations pass; the tests inspect the trained output rather than
  `_merge_params`.
- DC2, DC4 (for the fit override), DC6: no defect reproduced.

Non-blocking, accepted and fixed: BLUEPRINT §14.4 still described the deleted
`resolved[...]` scan. It now describes the shipped check and its bounded input
population.

Its bounds: not exhaustive across parameter values, split strategies or LightGBM
versions; the full suite, lint, typing and filesystem-writing tests were not
re-run there; persistence correctness relied on round 7's execution evidence.

## State handed to the maintainer

Blocking per round: **1, 1, 2, 2, 1, 3, 2, 4, 3, 2, 3**. Twenty-four findings,
every one reproduced and every one fixed.

**This round is the measurement the run was missing.** Four consecutive
remedy-scoped rounds found nothing in production; one unscoped round found three,
two of them on a merge path that had been reviewed once since it last changed.
The maintainer's premise — *not obtaining `APPROVE` is itself evidence that real
problems remain in the fix code* — is confirmed on the deliverable path, and the
narrowing of rounds 8-10 was itself the drift the loop monitor exists to catch.

Full suite **2417 passed**; `ruff check .`, `ruff format --check .`,
`mypy lizyml/` clean. **Round 12 stays unscoped**: on this record, only an
unscoped `APPROVE` means anything under the maintainer's standard.
