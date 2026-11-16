Review-kind: review
Review-round: 1

# Review: the implementation of H-0147 (#406) — feature names a saved LightGBM model cannot hold intact, and `reason` on the two category-save refusals

Review the implementation at this head against Proposal H-0147 (HISTORY.md `## H-0147`, Accepted after `h0147-proposal-review` round 2 APPROVE at `6fcc8d8`). The implementation is `git diff 6fcc8d8 HEAD`. Answer as findings with `file:line` locations and executed or quoted evidence. `APPROVE` when the implementation does what the Proposal says and nothing it forbids; `CHANGES_REQUESTED` otherwise.

## Amendment to check (made during implementation, recorded in H-0147 実装の記録)

Acceptance 5 (the rule against the installed LightGBM) found that `"=b"` as the first feature reads back as `b`. The rule gained a third case, `equals_sign_at_feature_names_edge`: a `=` at the start of the first name or the end of the last name when the joined names split at `=` into a single non-empty piece. It is derived from LightGBM 4.6.0's reader (`src/boosting/gbdt_model_text.cpp`, the header loop with `Common::Split(cur_line, '=')`, the `strs.size() == 2` / `> 2` branches and `feature_names_ = Common::Split(..., ' ')`; `include/LightGBM/utils/common.h` `GetLine` and `Split`, which drops empty pieces). Evidence: `docs/audits/2026-11-i406/equals-readback.out`. Is the rule in `lizyml/estimators/lgbm/feature_names.py` exactly the reader's behaviour for every list of booster names (booster names never hold a space or NUL)? Is there a name list it misjudges — refused but intact, or admitted but altered?

## Checks (run what your environment allows)

1. **Rule** (`lizyml/estimators/lgbm/feature_names.py`, `LGBMAdapter.unsaveable_feature_names`): the three reasons, position order, an unfitted adapter answers `[]`; the names judged are the booster's own `feature_name()`.
2. **Where it runs** (`lizyml/persistence/exporter.py`): `check_feature_names` runs first inside `check_category_values`, so `Model.export`, `lizyml.persistence.export` and `export_code` refuse before writing anything (and before `_resolve_export_path`); `_saved_adapters` lists `refit` and every `fit_result.models` entry regardless of `pipeline_state_per_fold`; `export_code` keeps its H-0141 `empty_alias` refusal first. The context keys are exactly `column`, `position`, `state`, `reason`, `surface`.
3. **Fit warning** (`warn_unsaveable_feature_names`, called in `lizyml/core/model.py` after `warn_saved_state_changes`): same enumeration and answer as the refusal, one warning per (position, reason), raised after the fit result is committed.
4. **`reason` on the category refusals**: `_refuse_value` has `unsaveable_category_value` and the adapter branch `lightgbm_unwritable_category`; the other keys and the messages are unchanged.
5. **IF** (`lizyml/estimators/base.py`): `UnsaveableFeatureName` and the `[]` default; `persistence/` still imports no concrete estimator.
6. **Tests** (`tests/test_persistence/test_h0147_feature_names.py`, the one updated assertion in `tests/test_persistence/test_unwritable_categories.py`): does each Proposal acceptance criterion have a test that fails without the change? Does acceptance 5 compare against the real LightGBM read-back for every case rather than a fixture?
7. **Docs**: BLUEPRINT §14.1 and §15.3, CHANGELOG `[Unreleased]`, `docs/proposal_dispositions.toml`, and the H-0147 amendment and Firing rate (15/42) against the `.out` files.

## Environment

The workspace is a digest-fixed, read-only copy of this head. Run single test files or the probes with the workspace's Python if it has the dependencies; never the full suite. The full suite with slow tests ran locally on this head. Say what could not run.

## Stop criterion

The Proposal's own レビューの停止基準 applies: findings must be contract defects (a model that does not read back intact passing export, an intact model refused, the three surfaces disagreeing, the warning and the refusal judged by different rules, a missing `reason`) or false statements in the diff. #409's names, rescuing artifacts saved by 0.21.1, and estimators other than LightGBM are not findings.
