# `METRIC_REQUIRES_PROBA` — independent probe, before the reviewer returns

The plan's PR 6 keeps this member and implements it, on one clause with no
evidence behind it:

> raised from metric dispatch when a `needs_proba` metric is asked for a task or
> artifact that has no probabilities.

Two halves. Measured here at `5712f41`, read-only.

## The task half is closed by construction

`r17_proba_probe.py`:

```
regression   huber mae mape r2 rmse rmsle smape wape
binary       accuracy auc* auc_pr* brier* ece* f1 logloss* precision_at_k*
multiclass   accuracy auc* auc_pr* brier* f1 logloss*
  * = needs_proba

needs_proba metrics permitted on regression : none
auc / auc_pr / brier / ece / logloss / precision_at_k on regression
                                            : UNSUPPORTED_METRIC raised first (6/6)
```

`_TASK_METRICS` (`metrics/registry.py:24-32`) is a per-task allowlist and
`get_metrics_for_task` raises `UNSUPPORTED_METRIC` at `:166` for anything outside
it. **No `needs_proba` metric is reachable on a task without probabilities.**

## The artifact half has no detection site

- `Model.evaluate(metrics=[...])` calls `get_metrics_for_task` at
  `core/model.py:320` *before* anything else, so the post-`load()` path is gated
  by the same allowlist.
- `_pred_for_metric` (`evaluation/evaluator.py:39-64`) is where dispatch decides
  what a `needs_proba` metric receives. It **has no failure branch at all**:
  `if metric.needs_proba:` returns `raw_pred`, normalised for the multiclass
  simplex case. There is nowhere for the code to be raised from.

## Consequence, pending the reviewer

On the shipped estimator, the condition appears **not constructible** — the same
status the plan measured for `DATA_FINGERPRINT_MISMATCH` and used to justify
dropping it. If that holds, PR 6's "two of three implemented" is "one of three",
and the post-PR population is 18, not 19.

The one argument for keeping it: a third-party `EstimatorProvider` whose
`predict_proba` returned non-probabilities would violate the contract, and this
code could name that. But LightGBM is the only shipped provider, so such a check
would be a condition no real input satisfies — DC6 rather than DC7, and the
Change Gate would want a measured firing rate before it ships.
