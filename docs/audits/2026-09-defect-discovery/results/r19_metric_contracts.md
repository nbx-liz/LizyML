# Per-metric input contracts, from the authoritative definitions

Answering: *can `METRIC_REQUIRES_PROBA`'s condition be defined per metric from
the specialist documentation, instead of by a shape heuristic?*

**Yes — and doing it exposes that `needs_proba` is one flag over two different
requirements.** Sources are the packed scikit-learn documentation
(`~/local-docs/packed/scikit-learn-docs.md`), which is what each of these metrics
actually calls.

## What scikit-learn itself says

The docs split classification metrics into exactly two classes, and say so in
one sentence (`:22973-22979`):

> If the scoring function **only accepts probability estimates** (e.g.
> `metrics.log_loss`), then one needs to set the parameter
> `response_method="predict_proba"`. Some scoring functions **do not necessarily
> require probability estimates but rather non-thresholded decision values**
> (e.g. `metrics.roc_auc_score`).

The scorer table (`:22829-22845`) marks the same division: `neg_brier_score`,
`neg_log_loss` and `d2_log_loss_score` carry *requires `predict_proba` support*;
`average_precision` and every `roc_auc*` variant do not.

And on `roc_auc_score` directly (`:24190-24192`):

> This function requires the true binary value and the target scores, which can
> either be **probability estimates of the positive class, confidence values, or
> binary decisions**.

Log loss (`:23950`): *"defined on probability estimates"*, with the multiclass
form summing over a matrix `P̂` of per-class probabilities (`:23970-23976`) — so
the row-simplex requirement is in the definition, not an implementation detail.

## The contract table

| metric | calls | requires | range | shape (multiclass) |
|---|---|---|---|---|
| `logloss` | `log_loss` | **probabilities** | `[0,1]` | 2-D, rows sum to 1 |
| `brier` | `brier_score_loss` | **probabilities** | `[0,1]` | 2-D per-class (OvR) |
| `ece` | own | **calibrated probabilities** | `[0,1]` | 2-D per-class |
| `auc` | `roc_auc_score` | *scores* — any monotone real | unbounded | 2-D per-class (OvR) |
| `auc_pr` | `average_precision_score` | *scores* | unbounded | 2-D per-class (OvR) |
| `precision_at_k` | own (ranking) | *scores* | unbounded | n/a (binary only) |

## The finding this produces

**`needs_proba=True` on `auc`, `auc_pr` and `precision_at_k` is wrong against the
definition those metrics come from.** All three accept non-thresholded decision
values; a raw logit is a valid input to `roc_auc_score`.

This is **not a live bug.** `_pred_for_metric`
(`evaluation/evaluator.py:55-64`) uses the flag only to choose between passing
the score through and binarising / argmax-ing it. Passing the score through is
the *correct* behaviour for AUC, so the outcome is right and the flag's name is
wrong — the same shape as #273's `embargo`, where the implementation is
defensible and the term of art is not.

**It matters the moment the guard is built.** A guard keyed on `needs_proba`
would demand `[0,1]` for `auc` and reject a perfectly valid input. The condition
has to key on the *probability* requirement, which is a strict subset.

## What the guard becomes

Instead of "multiclass predictions that are not 2-D", per metric:

| condition | detected for |
|---|---|
| values outside `[0,1]` | `logloss`, `brier`, `ece` — **binary and multiclass** |
| multiclass rows not summing to 1 | `logloss` |
| multiclass array not 2-D, or column count ≠ class count | `logloss`, `brier`, `ece`, `auc`, `auc_pr` |

**This retracts what I told you about the binary blind spot.** I said binary was
*intrinsically undetectable*. That is true only for **hard labels** — `{0,1}` is
a subset of `[0,1]`, and a perfectly confident model is legitimate, so no test
separates them. It is **false for raw scores**: logits fall outside `[0,1]` and
the range check catches them in binary as well as multiclass. Since H-0030
introduced `predict_raw` returning logits, that is the realistic confusion, and
it is detectable.

Corrected statement of the limit: **binary hard labels remain undetectable;
binary raw scores are detectable for the three `[0,1]` metrics.**

## Reachability, stated honestly

`oof_pred` holds probabilities and `oof_raw_scores` holds logits as a **separate
field** (`core/types/fit_result.py:63`, `:76`; filled at
`training/cv_trainer.py:309`). The metric path reads `oof_pred`, so no shipped
path puts logits there.

So the guard is a **public-API boundary check** on a caller-supplied `FitResult`
through the exported `Evaluator`, not a detector of a failure the library
produces. That is a legitimate thing to build — `_compute_metrics`
(`evaluation/evaluator.py:66-76`) has **no `try`/`except` at all**, so every
metric failure escapes as a raw sklearn `ValueError` with no code and no
context — but the `Firing rate:` line must say so rather than implying real
inputs trip it.

## What defining the contracts surfaced: `auc` and `auc_pr` disagree

`AUC.needs_simplex = True` (`classification.py:148-149`), `AUCPR.needs_simplex =
False`. Both compute a **per-class OvR macro** through the same helper
(`_multiclass_ovr_macro`), calling **binary** `roc_auc_score` /
`average_precision_score` per class — neither goes through sklearn's
`multi_class="ovr"` mode. Both are rank-based. Yet on a `multiclassova` artifact
one is row-normalised before the metric sees it and the other is not.

Measured (`r21_auc_simplex.py`, 400 rows, 3 classes, independent sigmoids with
row sums `0.570`–`2.509`):

```
auc      needs_simplex=True   raw=0.797399  after dispatch=0.847941   DIFFERS
auc_pr   needs_simplex=False  raw=0.669219  after dispatch=0.669219   SAME
logloss  needs_simplex=True   raw=0.366686  after dispatch=0.876616   DIFFERS
brier    needs_simplex=False  raw=0.226767  after dispatch=0.226767   SAME

softmax control (rows already sum to 1): all SAME
row normalisation preserves within-class sample order: False
```

**A 0.05 swing in reported AUC**, and row normalisation is **not** monotone
within a column — it divides each *row* by its sum, so it reorders samples
inside a class, which is precisely what a rank-based metric measures.

**`logloss`'s `needs_simplex=True` is correct** — `log_loss` is defined on a
probability distribution, and sklearn itself warns *"The y_prob values do not sum
to one"* on the un-normalised input. H-0049's stated rationale ("necessary for
`multiclassova`") holds there.

**For `auc` it is a judgement, not a clear error, and the honest reading is that
the two siblings are inconsistent.** Normalising first makes `auc` behave like
sklearn's `roc_auc_score(multi_class="ovr")`, which *does* require probabilities
summing to 1 — so it is defensible. But `average_precision_score` has no
multiclass mode imposing that, `auc_pr` does not normalise, and the two metrics
are otherwise computed identically. On `multiclassova` they are therefore scored
on **different inputs**, and nothing states which is intended.

This is a live numerical question, not a documentation gap, and it is not covered
by any filed issue.

## Consequences for the plan

1. **A fourth option exists for #263** that neither the plan nor the review
   considered: define the requirement per metric from the definition, replacing
   `needs_proba` as the guard's key.
2. **A new defect, not in any filed issue:** three metrics declare
   `needs_proba=True` against their own definitions. `BLUEPRINT.md:1024`'s
   `needs_proba / greater_is_better / supports_task` enumeration is the place it
   would be stated — the same line H-0049 already puts in the 18 for omitting
   `needs_simplex`.
3. The distinction is already half-present in the codebase: `needs_simplex`
   (H-0049) separates "wants a simplex" from "wants a probability". A third
   property, or a single three-valued one, would close it.
