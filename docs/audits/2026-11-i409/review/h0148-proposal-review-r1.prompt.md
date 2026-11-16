Review-kind: review
Review-round: 1

# Review: Proposal H-0148 (HISTORY.md `## H-0148`) — refuse feature names LightGBM refuses, and duplicate column names, before fit and tune train (#409)

Review the Proposal text at this head, not an implementation (there is none yet). Answer as findings with `file:line` locations and executed or quoted evidence. `APPROVE` when the Proposal is correct, complete and implementable as written; `CHANGES_REQUESTED` otherwise. Also check the facts stated in issue #409 and in the Proposal's 実測 against the code at this head and `docs/audits/2026-11-i409/`.

## Decided (do not reopen)

The owner's rulings in 管理者の判断 (2026-11-16): `DATA_SCHEMA_INVALID` with a snake_case `reason`; both `Model.fit` and `Model.tune` refuse before any training and tune runs no trial; names with spaces that do not collide stay as they are, with no new warning; v0.21.3 follows.

## Checks (run what your environment allows)

1. **LightGBM's rule.** Is the rule in 対応方針 2 exactly LightGBM 4.6.0's `Dataset::set_feature_names` and `Common::CheckAllowedJSON` (`include/LightGBM/dataset.h:888-913`, `include/LightGBM/utils/common.h:886`) as reached from Python's `lgb.Dataset` with a pandas frame (the names `str()`-ed, passed as C strings, so cut at NUL)? Check the order (JSON check before the space rewrite and the repeat check, name by name), and whether any other path in `lgb.Dataset.construct` or `lgb.train` refuses a feature name (for example a categorical feature named by a refused or colliding name, or the validation set built with `reference=`).
2. **Where the check sits.** Do `Model.fit` (`lizyml/core/model.py`) and `Model.tune` (`lizyml/core/_model_tuning.py`) both reach a point after `_prepare_training_data` and `get_provider` and before any training, study creation or trial? Is the claim true that the X columns reach the adapter's `lgb.Dataset` unchanged (`lizyml/features/pipelines_native.py`, the adapter's `fit`)? Are there other training entry points (refit-only paths, `Model.tune` resume, inner-valid splits, calibration, `export_code`'s generated `train.py`) that would still reach LightGBM with such a name, and does the Proposal need to name them?
3. **Duplicate column names** (対応方針 4): is `dataframe_builder.build` the right place, does "appears more than once and not in `features.exclude`" refuse every frame that crashes today and admit every frame that fits today (the `excluded-duplicate-and-storage.out` case), and what about a duplicate target, `time_col` or `group_col` name?
4. **Capability shape** (対応方針 1): is a `runtime_checkable` Protocol beside `ResourceWarningProvider` consistent with BLUEPRINT §14.4 and the layering of §2.1 (core imports no concrete estimator)?
5. **Context and compatibility**: are the context keys well defined for both reasons (non-string column names such as `0` and `1.5`)? Are the 互換性 statements true, including that every refused input fails today and that tune with storage would no longer write a study?
6. **Change Gate fields**: the `Firing rate:` line (12/16) matches `fit-names.out`; `Gate-category: none` is admissible.
7. **Acceptance criteria**: is each testable, does criterion 1 fail on this head, and does criterion 4 compare against the real LightGBM rather than a fixture (DC7)?

## Environment

The workspace is a digest-fixed, read-only copy of this head. Run single test files or the probes with the workspace's Python if it has the dependencies; never the full suite. Say what could not run.

## Stop criterion

The Proposal's own レビューの停止基準 applies: findings must be contract defects (a name LightGBM refuses still reaching training, a name it accepts refused, fit and tune disagreeing, tune running a trial, a missing `reason` or context) or false factual claims. Non-colliding space names, H-0147's save check, predict frames and estimators other than LightGBM are not findings.
