# 2026-09-04 — Codex check on #263's three `ErrorCode` dispositions

Prompt `prompts/errorcode-review-263-r1.md`, schema `errorcode-263-schema.json`,
verdict `reviews/ec263-r1.json`. Read-only over `/home/rem/repos/LizyML` at
`5712f41`; repository left clean.

**`VERDICT: DISPOSITIONS_HOLD`** — all three survive, but PR 6's justification
for two of them is wrong in ways that would have shipped a defect.

## The mechanical claim holds

20 members, 17 raised, 3 unraised — reproduced by an AST scan over `ast.Raise`
subtrees, not just a literal grep. No dynamic construction anywhere
(`ErrorCode[...]`, `ErrorCode(value)`, `getattr`, enum iteration): **0 sites**.
The documentation rows at `BLUEPRINT.md:1351/1356/1359` and
`docs/api.md:474/479` exist and say what the issue says.

The reviewer's caveat is fair: a literal grep is not *by itself* a sound
population proof, because a member can be produced without its name appearing.
It happens to be correct at this head. PR 6 already ships the AST scan.

## `DATA_FINGERPRINT_MISMATCH` — drop holds, the stated reason does not

Executed, not argued:

| | measured |
|---|---|
| `file_hash` | `DataFingerprint(row_count=80, ..., file_hash=None)` — `None` on the public `fit(df)` path, confirmed by execution |
| reordered columns | predicts successfully, no warning, though `column_hash` changes |
| missing column | `LizyMLError(DATA_SCHEMA_INVALID)`, context `{'missing_columns': ['feat_b']}` |
| extra column | predicts, warns `Extra columns ignored during transform: ['extra']` |
| `float64` → `float32` | predicts successfully, but an order-insensitive `(name, dtype)` set comparison **would reject it** |
| `float64` → `str` | reaches the raw dtype failure — but that is `INCOMPATIBLE_COLUMNS`'s condition, not this one |

**Minor finding, and it is a real one:** PR 6 says `row_count` "differs on every
valid batch, by construction". False — predicting on all 40 training rows returns
`fit_rows=40 predict_rows=40 status=OK`. The correct reason is that batch size is
unconstrained, so equality or inequality *carries no compatibility meaning*.
The drop stands; the sentence justifying it was overstated in a way a reviewer
could have refuted, which is how a disposition gets reopened later.

## `INCOMPATIBLE_COLUMNS` — keep, and the reproduction is real

Fit regression on 80 rows, then `predict` after `X.astype({'feat_a': 'str'})`:

```
EXC type=ValueError code=None context=None
    pandas dtypes must be int, float or bool.
    Fields with bad pandas dtypes: feat_a: str
```

The name-only pipeline check passes and reorders at
`features/pipelines_native.py:104-125`; `_model_predict.py:43-56` then hands it to
the estimator. **No LizyML guard intercepts it.** `DATA_SCHEMA_INVALID` covers
absent names, not changed dtypes, so this is a distinct condition.

## `METRIC_REQUIRES_PROBA` — keep, but the plan's stated condition is unreachable

**My pre-review measurement was half right and half wrong, and the wrong half
mattered.**

Right: the **task half is closed by construction**. Zero `needs_proba` metrics
are permitted on regression; all six (`auc`, `auc_pr`, `brier`, `ece`, `logloss`,
`precision_at_k`) trip `UNSUPPORTED_METRIC` at `metrics/registry.py:166` first,
and `Model.evaluate` gates through the same call at `core/model.py:320`, so the
post-`load()` path is gated too.

Wrong: I concluded from `_pred_for_metric` having no raise statement that no
condition could be constructed. That confuses "nothing raises here today" with
"nothing could". The reviewer constructed it, and **I re-ran the construction**:

```
real multiclass fit  : oof_pred.ndim = 2, shape (120, 3)   <- never 1-D
forged 1-D oof_pred, through the exported Evaluator:
  auc     -> ValueError code=None : multi_class must be in ('ovo', 'ovr')
  logloss -> ValueError code=None : y_prob contains values greater than 1: 2.0
```

`Evaluator` is public (`lizyml/evaluation/__init__.py:3-8`) and takes a
caller-supplied `FitResult`. A malformed one escapes as a bare sklearn
`ValueError` with **no `ErrorCode` and no context** — which is exactly what the
library's own exception policy exists to prevent. The member is implementable.

**But PR 6's contract for it must be narrowed, or it ships a defect:**

- **DC4** — a task-side branch after `get_metrics_for_task` is unreachable,
  because `UNSUPPORTED_METRIC` fires first.
- **DC7** — presenting task incompatibility as this member's condition is an
  unsatisfiable declaration.

The condition that *is* reachable: for an allowed classification metric with
`needs_proba=True`, the supplied prediction artifact cannot represent
probabilities — e.g. multiclass predictions that are not 2-D — with metric, task,
expected shape and actual shape in the context.

**One limit worth recording**, raised by the reviewer against its own proposal:
binary hard labels and binary probabilities are **both 1-D**, so a dimensionality
check cannot distinguish them. The guard covers **multiclass only**. A binary
`FitResult` carrying hard labels stays undetected.

## What changes in the plan

1. PR 6's `METRIC_REQUIRES_PROBA` paragraph — replace "asked for a task or
   artifact that has no probabilities" with the artifact-shape condition, and say
   the task half is already `UNSUPPORTED_METRIC`.
2. PR 6's `row_count` row — replace "differs on every valid batch, by
   construction" with the unconstrained-batch-size reason.
3. PR 6's RED test for this member — spell out the fixture: `dataclasses.replace`
   on a valid multiclass `FitResult`, prediction arrays swapped for 1-D hard
   labels, asserting `code` and the shape context. Without it a test author
   targets regression and merely reorders `UNSUPPORTED_METRIC` precedence.
4. Record the binary blind spot as a stated limit of the check.

The population stays **19** and "two of three implemented" stands.

## A caveat on this review's own provenance

The first invocation could not authenticate `gh` inside its sandbox and fell back
to a web search; the second was given the issue body as a local file. The verdict
file's summary still mentions the authentication failure, so it may be the first
run's late-written output. Every load-bearing claim in it was re-executed here
before being adopted, so the substance is verified independently of which run
produced it — but the provenance is not clean and is recorded rather than hidden.
