# H-0122 PR 1 design: tune resume / storage reattach compatibility

I have not changed any files. This plan is ready for a maker to implement. It covers every PR 1 criterion in `pr-split.md`.

**Decisions that differ from the brief or the spec text** (each has a line in §8):
1. **The Tuner keeps its enqueue parameter.** `test_direction_reconciliation.py:111-115` calls `Tuner.tune(enqueue_params=...)` directly, and criterion 21 forbids changing Tuner-direct tests. Removing the enqueue therefore happens on the Model side only: `Model.tune` stops passing it (R1).
2. **More conditions count as "inconsistent" (state X)** than the spec's closed table lists: an orphan `:end`, undecodable records, bad `start_trial`, and trial params with no route (R2). LizyML's own writes can't produce any of them. Record this in the PR 1 decision record.
3. **`data.path` is not part of the identity**, because the file content is hashed instead (R4).
4. **A different `boundary_threshold` on an open round is not refused**, because U2 doesn't cover it; it is logged at INFO (R6). Check with the maintainer if a refusal is wanted.

Verified by running code:
- **In memory only:** a `BaseException` subclass raised from the objective leaves the trial RUNNING, and `study.tell(n, FAIL)` then marks it FAIL. In-memory `user_attrs` keep tuples as written.
- **SQLite:** the same is expected to hold, because it follows the same `_run_trial` path (`optuna/study/_optimize.py:211`) and matches the dead-process fact in `probes/optuna_facts.py`. I did not execute it against SQLite.

Section numbers like §7.4 refer to `docs/audits/2026-10-tuning-resume/resume-spec-draft.md`, whose text matches HISTORY.md from line 12053. Worktree `fix/h0122-resume-compatibility` @ `a0afbf5`.

PR 1 criteria:
- In full: 1, 2, 3, 5, 8, 8a, 9, 10, 11, 13, 16, 18, 21, 23, 24, 25, 26, 27, 28, 31, 32, 33.
- In part:
  - 8b: `n_trials`, `expand_boundary` and the config `n_trials` warning.
  - 12: everything except category change.
  - 14: no `space=` rows.
  - 15: no `space=`.
  - 19: no `space=`.
  - 22: PR 1 rows.
  - 30: bullets 2 and 3.

## 1. Module layout

### 1.1 Why

- **`_model_tuning.py` is 708 lines.** Orchestration has to grow, so the mixin keeps only the phase skeleton, the objective closure and the commit.
- **BLUEPRINT.md:448 and spec §9.2.** Tuner only creates/loads studies and reads/writes attributes. The *content* of identity and lineage is built on the Model side.
- **Layering.** `lizyml/tuning/` is Layer 2 (BLUEPRINT.md:46-59). It can't import `LizyMLConfig` or concrete providers. Its lineage code must be pure so PR 2's `lizyml.tuning.read_study_lineage` (§7.6) can wrap it.
- **No new refusals in `_model_factories.py`.** `tests/test_core/test_refusal_matrix.py` asserts every refusal there has a column in its grid.

### 1.2 Files

| File | Layer | Status | Budget | Responsibility |
|---|---|---|---|---|
| `lizyml/tuning/lineage_codec.py` | 2 | new | ~170 | See below. |
| `lizyml/tuning/lineage.py` | 2 | new | ~380 | See below. |
| `lizyml/tuning/rounds.py` | 2 | rewrite | ~140 | `assemble_tuning_result(lineage, trials, best_trial, *, metric_name, direction, current_round)`. It builds trial rounds, `RoundSummary`, `boundary_report` and the routed best from lineage. Replaces `assemble_round_result` (`rounds.py:18-92`), whose only caller is `_model_tuning.py:678`. |
| `lizyml/tuning/tuner.py` | 2 | extend | 323 → ~400 | New module-level study I/O: `load_stored_study`, `create_study`, `write_study_attr`, `fail_running_trials`, `check_study_direction`. New `Tuner.run(...)` does only the optimize step. `Tuner.tune(...)` keeps its signature and combines these for direct callers. |
| `lizyml/core/_tuning_identity.py` | 4 | new | ~230 | §4.1 objective identity: `summarise_frame(df)`, `config_parts(cfg, fixed, metric_entry, direction)`, `build_identity`, `mismatched_elements`, `library_versions(provider)`, `version_differences`. Pure, no Model state. |
| `lizyml/core/_tuning_resume.py` | 4 | new | ~300 | The §3 call matrix and all pre-write checks: `precheck_study_args`, `resolve_call_row`, `same_study`, `locate_study -> StudyTarget`, `refuse_state` (L/S/X), `check_against_record` (direction, §5.1, §4.3, version warning). Builds the `lizyml:study` payload. |
| `lizyml/core/_tuning_plan.py` | 4 | new | ~380 | `resolve_config_space(cfg, provider) -> ResolvedSpace`, moved from `_model_tuning.py:413-474`. Frozen `RoundPlan`. `plan_first_round`, `plan_next_round`, `plan_open_round` (with U2 and the config `n_trials` warning). `route_entries(dims, provider)`. Boundary expansion, moved from `_maybe_expand_boundary` (`_model_tuning.py:476-525`). |
| `lizyml/core/_model_tuning.py` | 4 | rewire | 708 → ~600 | `tune()` runs phases A–F (§3). `_build_tune_objective` unchanged. `_run_tune_round` **keeps its name**, because `test_fit_params_override.py:3090` and `test_space_merge.py:369` patch it. New `_close_round` and the commit block. `_resolve_search_space(*, provider)` becomes a thin wrapper. `_maybe_expand_boundary` and `_assemble_tuning_result` are removed (moved). The `tune()` docstring (`:153-180`, in particular `:163-169` on `load_if_exists` reattach and `:178-179` "resume without prior tune()") is updated to the new rules. |
| `lizyml/core/model.py` | 4 | +4 lines | | Two fields in `__init__` next to `:179-185`: `_tuning_storage: str \| BaseStorage \| None = None` and `_tuning_study_name: str \| None = None` (the remembered "same study" pair, §3). No other change; `_merge_params` is reused with `include_tuning_result=False`. |

**`lineage_codec.py`:**
- Tagged scalar codec, plus `SearchDim` and `BoundaryReport` encode/decode.
- `to_attr_value(payload)` = `json.loads(json.dumps(payload, allow_nan=False))`. This is a strict JSON round trip, so in-memory and RDB studies hold identical list-based values. In-memory storage keeps the object as written (`optuna/storages/_in_memory.py:102-106`); RDB uses `json.dumps` (`_rdb/storage.py:321`).
- Raises `LineageCodecError(ValueError)` on bad input.

**`lineage.py`:**
- Key names and `SCHEMA_VERSION = 1`.
- Frozen dataclasses `StoredStudy`, `Route`, `StoredRound`, `StoredRoundEnd`, `Lineage`.
- Pure `classify_study(user_attrs, trials) -> Lineage`, implementing the §7.4 table.
- `round_of`, `merge_routes`, `split_by_routes`, `remaining_trials`, `remaining_timeout`.
- No Optuna import. Trials are duck-typed through a `TrialLike` Protocol: `number`, `state.name`, `datetime_start`, `datetime_complete`, `params`, `value`.

### 1.3 What moves or changes

- **`_resolve_search_space` → `_tuning_plan.resolve_config_space`.**
  - The resumed-space branch is removed (`_model_tuning.py:437-440`: `resume and self._space`).
  - The wrapper keeps the 4-tuple `(attach_bounds(declared), used_default, fixed, default_dims)` for `test_space_merge.py:36, 134, 170`. Those three calls just drop the `resume=False` keyword.
- **`_maybe_expand_boundary` → `_tuning_plan._expand_for_next_round`.** This also takes the logs at `:499, :509, :515-524`.
- **`_assemble_tuning_result` and `assemble_round_result` → `rounds.assemble_tuning_result`.**
- **`_validate_tune_inputs`** (`:379-411`) loses the block at `:393-401`, which moves to phase A5.
- **`Tuner.tune` body** (`tuner.py:166-323`) is split:
  - Create/load → `create_study` / `load_stored_study`.
  - Direction check (`:181-198`) → `check_study_direction`.
  - Optimize with callbacks (`:204-280`) → `Tuner.run`. Progress uses a new `progress_total` (default `self.n_trials`) instead of `n_total = self.n_trials` (`:210`).
  - The "no COMPLETE" raise (`:282-290`) and the result build (`:299-323`) stay in `Tuner.tune` for direct callers only.
- **`search_space.split_by_category`** (`:200-225`, model fallback at `:218`) is unchanged. `Tuner.tune` still uses it at `:313`. Model routes through `lineage.split_by_routes` instead (§4.3).

## 2. Attribute schema (schema version 1)

**General rules**
- Every payload goes through `to_attr_value` **while planning (phase D)**. A non-serialisable value therefore fails before any write (INV-R4).
- **Tagged scalars.** Every value slot that isn't a plain flag or name is `[tag, value]`:
  - Tags: `"bool"`, `"int"` (exact beyond 2**53), `"float"` (finite number, or the string `"nan"` / `"inf"` / `"-inf"`), `"str"`, `"none"` (value `null`).
  - The tag comes from `type(v) is ...`, mirroring `_ALLOWED_CHOICE_TYPES` (`search_space.py:37`).
  - Any other type, including numpy scalars (already refused by `parse_space`, `search_space.py:105-119`), raises `LineageCodecError`.
  - The decoder checks the tag against the JSON type; `["int", true]` is malformed.
  - This keeps `True`, `1`, `1.0` and NaN distinct and strict-JSON-safe, as PR 2's §5.2-3 equality needs.
- A codec error in `lizyml:study` gives state S. In a round or `:end` record it gives state X.

**Encoded `SearchDim`** (exact round trip of `core/types/search_dim.py:20-61`):
```json
{"kind": "float", "name": "learning_rate", "category": "model",
 "low": ["float", 0.01], "high": ["float", 0.3], "log": true,
 "min_allowed": ["float", 1e-08], "max_allowed": ["float", 1.0]}
{"kind": "int", "name": "num_leaves", "category": "model",
 "low": ["int", 8], "high": ["int", 32], "log": false,
 "min_allowed": ["none", null], "max_allowed": ["none", null]}
{"kind": "categorical", "name": "boosting", "category": "model",
 "choices": [["str", "gbdt"], ["bool", true], ["int", 1], ["float", 1.0], ["float", "nan"], ["none", null]]}
```
A `FloatDim` with an int literal `low` keeps `["int", 0]`, so `decode(encode(d)) == d` for every dim. Decoded `choices` is a tuple.

**Encoded `BoundaryReport`** (`core/types/tuning_result.py:16-57`):
```json
{"dims": [{"name": "learning_rate", "best_value": ["float", 0.29], "low": ["float", 0.01],
           "high": ["float", 0.3], "position_pct": ["float", 0.9655], "edge": "upper",
           "expanded": true, "new_low": ["none", null], "new_high": ["float", 0.9],
           "clamped_to_bound": false}],
 "expanded_names": ["learning_rate"]}
```

**`lizyml:study`** — written once, at study creation or when initialising an E-state study:
```json
{"schema": 1,
 "objective": {"digest": "<sha256 hex over sorted (element, digest) pairs>",
               "elements": {"data.columns": "<hex>", "data.dtypes": "<hex>", "data.categories": "<hex>",
                            "data.values": "<hex>", "data.categorical_codes": "<hex>", "data.object_types": "<hex>",
                            "config.task": "<hex>", "config.data": "<hex>", "config.features": "<hex>",
                            "config.model": "<hex>", "config.fixed": "<hex>", "config.training": "<hex>",
                            "config.split": "<hex>", "metric": "<hex>"}},
 "direction": "minimize",
 "versions": {"lizyml": "...", "lightgbm": "...", "optuna": "...", "pandas": "...", "scikit-learn": "..."},
 "space_mode": "merge",
 "fixed": {"first_metric_only": true, "metric": ["huber", "mae", "mape"]},
 "origin_space": ["<encoded dims of parse_space(cfg.tuning.optuna.space), in dict order>"],
 "created_at": "2026-10-11T12:00:00+00:00"}
```

**Identity element contents** (`_tuning_identity`)

Data elements, all computed from the **raw loaded frame**:
- `data.columns`: column names with their type names, plus index dtype and index names.
- `data.dtypes`: `str(dtype)` per column.
- `data.categories`: per categorical column, the `ordered` flag and `[type name, str(v)]` for each category.
- `data.values`: sha256 of `pd.util.hash_pandas_object(df, index=True)` bytes.
- `data.categorical_codes`: per categorical column (iterated by position, so duplicate names are safe), sha256 of `hash_pandas_object(col.cat.codes, index=False)`.
- `data.object_types`: per object column, a hash of `col.map(lambda v: type(v).__qualname__)`. If hashing raises `TypeError` (unhashable cells), fall back to `col.map(repr)` (R9).

Config elements: each is sha256 of `json.dumps(value, sort_keys=True, separators=(",", ":"), default=<numpy .item() / repr fallback>)` over `cfg.model_dump(mode="python")` subtrees:
- `config.task`
- `config.data` without `path` (R4)
- `config.features`
- `config.model`: whole model config, including `params`, smart values, `feature_weights`, `balanced`.
- `config.fixed`: the current effective fixed params.
- `config.training`: whole, including the computed `inner_valid_explicit` (`config/schema.py:518-529`), which steers the factories.
- `config.split`
- `metric`: first metric name and kwargs from `parse_metric_entry((cfg.evaluation.metrics or _DEFAULT_METRICS[task])[0])`, plus the derived direction.

Excluded: `tuning.*`, `calibration`, `evaluation.metrics[1:]`, `output_dir`, `config_version`.

`fixed` is plain JSON; provider values are plain (`estimators/lgbm/defaults.py:138-153`).

`origin_space` is the parsed config declaration without bounds. Parsing normalises an omitted `log`, and encoding keeps NaN choices strict-JSON.

**`lizyml:round:<k>`** — written once, after all planning, before any trial:
```json
{"round": 2,
 "declaration": {"source": "previous",
                 "space_arg": null,
                 "provenance": [["model", "learning_rate", "default"], ["model", "num_iterations", "user"]]},
 "declared": ["<encoded dims, no min/max_allowed>"],
 "effective": ["<encoded dims, bounds attached, expansion applied>"],
 "expand_boundary": true,
 "boundary_threshold": ["float", 0.05],
 "boundary_report": null,
 "budget": {"n_trials": 3, "timeout": ["none", null]},
 "start_trial": 5,
 "best_before": ["float", 1.234],
 "routes_added": [{"name": "eta", "canonical": "learning_rate", "category": "model"}]}
```

Field notes:
- **`declaration.source`:** `"config"` for round 1 and `"previous"` after that. PR 2 adds `"space_arg"`.
- **`declaration.space_arg`:** always `null` in PR 1. It is the slot for PR 2's `space=` argument.
- **`provenance`:** `[category, raw name, "user" | "default"]` per declared dim, from the `overrides` / `default_dims` logic now at `_model_tuning.py:463-470`.
- **`expand_boundary`:** the boolean resolved by the H-0102 rule, recorded even for round 1.
- **`boundary_report`:** encoded, or `null` when there is no expansion or no previous best.
- **`routes_added`:**
  - `name` is the raw dim name, which is also the Optuna param key.
  - `canonical` is `provider.canonical_param_names(...)` for `model` dims only. For other categories it equals `name`. Canonicalising the training dim `early_stopping_rounds` would wrongly give LightGBM's `early_stopping_round`.
  - It lists only names not already routed (routes are append-only, §4.1b). Round 1 lists all declared dims; in PR 1, later rounds list `[]`.
- **Expanded dims:** `boundary_report.expanded_names` or `()`; there is no separate field.

**`lizyml:round:<k>:end`:**
```json
{"status": "completed", "best_after": ["float", 1.111], "ended_at": "2026-10-11T12:03:00+00:00"}
```
For `failed`, `best_after` is `["none", null]`.

**Schema policy**
- A study is readable when `type(schema) is int and 1 <= schema <= SCHEMA_VERSION`. Anything else, including missing keys or a malformed payload, is state S (`CONFIG_INVALID`, §7.1).
- Round and `:end` records are versioned by `lizyml:study.schema`. Unknown keys inside payloads are ignored; any change in meaning bumps the version.
- PR 1 and PR 2 both ship before v0.19.0, so PR 2 may still adjust schema 1 without a migration. The shape above already has `space_arg`, `provenance`, `declared` separate from `effective`, routes, threshold and the full report, so PR 2 should need no change.
- `lizyml:*` keys other than `lizyml:study` and `^lizyml:round:\d+(:end)?$` are ignored. A key that starts with `lizyml:round:` but doesn't match the pattern is X.

**Naming:** the internal types are `StoredStudy`, `StoredRound`, `StoredRoundEnd` and `Lineage`. PR 2's public `RoundRecord` and `StudyLineage` (§7.6) can then be built from them without a name clash.

## 3. `tune()` control flow (PR 1)

Phases A–D only read. All study writes are in E; all Model-state writes are in F.

### Phase A — arguments (no I/O)
- **A1.** `cfg.tuning is None` → `CONFIG_INVALID` (existing, `_model_tuning.py:383-391`).
- **A2.** §3 pre-checks in `precheck_study_args(storage, study_name)`, regardless of `resume`:
  - `storage` set and `study_name` None → `CONFIG_INVALID`. Today `Tuner.__init__` (`tuner.py:104-112`) raises this only after the `TUNING_FAILED` check; this fixes the order.
  - `study_name` set and `storage` None → `CONFIG_INVALID` (new; today it is ignored without notice).
- **A3.** `boundary_threshold` range → `CONFIG_INVALID` (existing, `:403-411`).
- **A4.** `direction = resolve_tuning_direction(cfg)` → `CONFIG_INVALID` on contradiction (`_tuning_validation.py:24-47`). `test_direction_reconciliation.py:56-66` requires this to happen before any `optuna.create_study`.
- **A5.** Call-matrix rows that need no I/O (`resolve_call_row`):
  - `resume and self._study is None and storage is None` → `TUNING_FAILED` (row 8). This is the relocated `:393-401`. Context `{"resume": True, "round_number": self._round_number, "storage_given": False}`.
  - `resume and self._study is not None and storage is not None and not same_study(...)` → `CONFIG_INVALID` (row 5).
- **`same_study` rule:**
  - `study_name == remembered_name`, and either both storages are `str` and equal, or the remembered storage is not a `str` and `storage is remembered`.
  - The remembered pair is read with `getattr(self, "_tuning_storage", None)` / `getattr(self, "_tuning_study_name", None)`, so whole-Model pickles from before this change reach state L instead of raising `AttributeError` (R10).
  - An in-memory `_study` remembers `(None, None)`, so any `storage` is refused.

### Phase B — data (no study I/O; per-call transients only)
- **B6.** `raw = self._load_data(data)` (`model.py:733-753`).
  - Then **immediately** `current_data = summarise_frame(raw)`, before anything else touches the frame.
  - Then `X, y, groups, _ = self._prepare_training_data(raw)`. With `data` not None, `_load_data` returns it unchanged, so there is one read and no `model.py` change.
  - The summary covers all columns, including `features.exclude`, in read order, before the builder casts or the time-series sort (`model.py:779, 805`).
- **B7.** `provider = get_provider(cfg.model)` as a **local**; the `self._provider` assignment moves to E0. Splitter as today (`:205-210`).

### Phase C — locate and judge the study (reads only)
- **C9.** `locate_study` → `StudyTarget`:
  - `resume and self._study is not None` → the remembered study (row 4).
  - Else if `storage` is set → `load_stored_study(storage, study_name, seed)`. This calls `stored_study_exists` (`tuner.py:58-69`) then `optuna.load_study(..., sampler=TPESampler(seed))`. Loading writes nothing (verified fact).
    - Found and `not resume` → `CONFIG_INVALID` (row 3, any state): "study already exists; use `resume=True` to continue or another `study_name` to start a new one".
    - Not found → "create in storage". If `resume` is set (row 7, U3), also `announce_create=True`.
  - Else (`resume=False`, no storage) → "create in memory" (row 1). The current `self._study` is replaced only at commit.
- **C10.** For an existing study: `lineage = classify_study(study.user_attrs, study.trials)`.
  - `legacy`, `unreadable`, `inconsistent` → `CONFIG_INVALID`, each with its own message (D4 / §7.1 / §7.4 row X).
- **C11.** For an existing study in any remaining state, including `uninitialized`: `check_study_direction(study, direction)` → `CONFIG_INVALID`. This is H-0099 decision 2, moved from `tuner.py:181-198` into a shared function; it is the identity check's position per §4.3.
- **C12.** `current = build_identity(current_data, config_parts(cfg, current_fixed, metric_entry, direction))`.
  - `current_fixed = provider.default_fixed_params(task)` if `space_mode == "merge"`, else `{}`. It is recomputed every call (criterion 28).
- **C13.** For `no_round`, `open`, `completed`, `failed` (`check_against_record`):
  - **a. §5.1.** If config `space_mode != study.space_mode` → `CONFIG_INVALID`. Then if `parse_space(cfg space)` differs from `decode(study.origin_space)` → `CONFIG_INVALID`. The space comparison is a set keyed by `(category, name)` with exact dim equality, order-insensitive. Message: "the space a study was created with cannot be changed through the config; start a new study". It runs before identity so a `space_mode` edit, which also changes `config.fixed`, gets this specific message.
  - **b. §4.3.** `mismatched = mismatched_elements(study.objective, current)`. If non-empty → `CONFIG_INVALID` with context `{"surface": "tuning.study", "mismatched": [sorted names], "study_name": name or "<in-memory>"}`. An element present on one side only counts as mismatched.
  - **c. Versions.** If `version_differences(study.versions, library_versions(provider))` is non-empty → `warnings.warn(..., UserWarning)` naming each package old→new, and continue (D4). This changes no state.
  - `uninitialized` (E) skips C13; its record is written in E2 from the current values.

### Phase D — plan (pure; the remaining refusals)
- **D14. `open` (O), round k** — `plan_open_round`:
  - **U2.** If `n_trials is not None and n_trials != record.budget.n_trials` → `CONFIG_INVALID`: "round k was started with n_trials=X; finish it without n_trials or with the same value". Same rule for `expand_boundary` against `record.expand_boundary`.
  - **Config warning.** If `n_trials is None` and the config's `n_trials != record.budget.n_trials` → `warnings.warn(UserWarning)` that the config value is ignored for this open round.
  - Effective space, report and threshold come from the record; analysis is not redone.
  - `running_to_fail` = numbers of RUNNING trials in round k.
  - `run_n_trials = max(0, record.n_trials - |{t in round k: state in {COMPLETE, FAIL, PRUNED, RUNNING}}|)`. RUNNING counts because it is FAILed in E3.
  - `run_timeout = remaining_timeout(record, round_trials)`.
  - `start_trial = record.start_trial`.
  - `fixed = study.fixed`.
- **D15. `completed` / `failed` (C/F), new round k = last + 1** — `plan_next_round`:
  - `declared` and `provenance` come from the last record; `prev_effective` from its `effective`; `fixed = study.fixed`.
  - `used_default = study.space_mode == "merge" and no "user" provenance`. Model fields are never read.
  - `should = expand_boundary if expand_boundary is not None else used_default`.
  - `best = study.best_trial` if any trial is COMPLETE, else `None`. State F always gets `None`. This is §5.3 bullet 3: the previous best comes from the study, never from `self._tuning_result`.
  - If `should` and `best` exists: `report = detect_boundary(prev_effective, {routed names from best.params}, boundary_threshold)` and `effective = expand_dims(prev_effective, report)` when anything expanded. Otherwise `report = None` and `effective = prev_effective`.
  - Budget is `n_trials` or the config value; timeout is the config value.
  - `start_trial = len(study.trials)`, `best_before = best.value` or `None`, `routes_added = []`.
- **D16. New study / `uninitialized` (E) / `no_round` (N), round 1** — `plan_first_round`:
  - `resolve_config_space` gives `declared`, `provenance`, `used_default`, `fixed` and `default_dims`. For N, `fixed` is `study.fixed` (identity has proved it equal to the recomputed value); otherwise it is the recomputed one.
  - **H-0121.** `admit_smart_dimensions(provider, declared, base_smart, default_dims)` drops or refuses (§5).
  - `effective = attach_bounds(declared_after_drop, provider.parameter_bounds(task))`; `report = None`; `should` is resolved and recorded.
  - Budget from the argument or config; `start_trial = len(trials)` (that is, 0); `best_before = None`.
  - `routes_added = route_entries(declared_after_drop, provider)`.
  - Builds `study_payload` for brand-new and E studies.
- **D17. All plan kinds.** There is **one** `_merge_params(provider, tuning_fixed_params=plan_fixed, include_tuning_result=False)` call per tune. Its `base_smart` feeds D16's admission. Then the existing checks run on `plan.effective`:
  - config-level checks `check_param_names`, `check_duplicate_space_dimensions`, `check_training_managed_space`, `check_calibration_param_names` (today at `:217-237`; moved after C13a so an edited config space gets the §5.1 message first);
  - `validate_tuning_dimensions` (`:270`);
  - the `training_claims` / `check_training_managed_overrides` pair (`:276-297`).
- **D18.** `to_attr_value` serialises `study_payload` and `round_payload`.

The INV-R4 boundary sits here: nothing above writes, and nothing below refuses with `CONFIG_INVALID`.

### Phase E — writes, in §7.1 order
- **E0** (no study write):
  - `self._ensure_run_dir(generate_run_id())`, moved from `:195` so a refused tune creates no run dir or `FileHandler` (`core/logging.py:45-70`).
  - `self._provider = provider`, moved from `:203`; it must precede `_build_run_meta`, which reads it (`model.py:828`).
  - Build run_meta and the objective (`_build_tune_objective` unchanged; it already applies `fixed` and then the trial values, `:569-571`).
  - The `tune.start` log (`:188-193`) moves here and logs `round`, plan kind and `plan.run_n_trials`.
  - Construct **exactly one** `Tuner(dims=effective, n_trials=plan.run_n_trials, direction, timeout=plan.run_timeout, seed, progress_callback, storage, study_name)`. `test_knob_reachability.py:539-556` asserts one `Tuner` construction carrying `storage`/`study_name`; `:459-462` maps `Tuner.n_trials`/`timeout` to config values for a fresh tune, where `run_n_trials == cfg n_trials` and `run_timeout == cfg timeout`.
  - If `announce_create` is set: `INFO event='tune.study_created' reason=resume_without_stored_study study_name=...` (U3).
- **E1 (write position a)**, new study only: `study = create_study(direction, seed, storage, study_name)`. It calls `_optuna.create_study(..., load_if_exists=False)` through the module attribute, so `patch("optuna.create_study")` guards keep working.
- **E2 (position b)**, new or E: `write_study_attr(study, "lizyml:study", plan.study_payload)`.
- **E3**, O only: `fail_running_trials(study, plan.running_to_fail)`, i.e. `study.tell(n, state=TrialState.FAIL)`.
- **E4 (position 2)**, not O: `write_study_attr(study, f"lizyml:round:{k}", plan.round_payload)`.
- **E5 (position 3)**, if `run_n_trials > 0` and (`run_timeout is None` or `> 0`): `self._run_tune_round(tuner, study, objective, ...)`, which calls `tuner.run(study, objective, round_number=k, round_start=plan.start_trial, progress_total=plan.budget_n_trials, expanded_dims=plan.expanded_names)`.
  - `Tuner.run` wraps an `Exception` from `optimize` as `TUNING_FAILED`, as `tuner.py:274-280` does today.
  - Any exception or `BaseException` propagates **without `:end`**. The round stays open (O) and the Model is untouched.
- **E6 (position 4)**, `_close_round`:
  - `complete = any(t.state is COMPLETE for t in study.trials)`.
  - Write `lizyml:round:{k}:end` as `completed` with `best_after = study.best_value`, or `failed` with `["none", null]`.
  - If not complete → `TUNING_FAILED` with context `{"round": k, "n_trials": plan.budget_n_trials, "status": "failed"}`. This is the old `tuner.py:285-290` raise, moved to the Model side so `:end` is written first. The Model is untouched, and an in-memory study created by this call is dropped (§7.2, last bullet).

### Phase F — assemble from lineage, then commit
- **F19.** `final = classify_study(study.user_attrs, study.trials)`. It must be `completed` (assert).
- **F20.** `result, rounds = assemble_tuning_result(final, study.trials, study.best_trial, metric_name=..., direction=..., current_round=k)`. One function serves in-process continuation and reattach, so INV-R5 holds by construction.
- **F21.** One commit block (today's pattern at `:361-368`):
  - `_X`, `_y`, `_tuning_result = result`, `_study = study`, `_round_number = k`, `_rounds = list(rounds)`, `_space = list(plan.effective)`, `_used_default_space = plan.used_default`, `_tuning_fixed_params = dict(plan.fixed)`.
  - The remembered pair: a storage target sets the arguments passed; row 4 continuation without arguments keeps the remembered pair; a new in-memory study resets it to `(None, None)`.

### INV-R4 proof table

INV-R4 holds when the last refusal phase comes before the first write phase.

| Site | Phase | Kind | Study effect | Model-state effect |
|---|---|---|---|---|
| A1–A4, A5 (rows 5, 8) | A | refusal (`CONFIG_INVALID` / `TUNING_FAILED`) | none | none |
| B6 data load/prepare | B | no refusal | none | per-call transients only (R8) |
| C9 row 3, C10 L/S/X, C11 direction, C13a §5.1, C13b identity | C | refusal | none (reads only) | none |
| C13c versions | C | warning | none | none |
| D14 U2, D16 H-0121, D17 checks, D18 codec | D | refusal | none | none |
| E0 run dir, `_provider`, Tuner | E | — | none | `_run_dir`, `_provider` |
| E1 create, E2 `lizyml:study`, E3 RUNNING→FAIL, E4 round record | E | write | written | none |
| E5 exception during optimize | E | **failure**, not a refusal | trials added, round left open (O) | none |
| E6 no COMPLETE | E | **failure** (`:end failed` + `TUNING_FAILED`) | `:end` written | none |
| F commit | F | write | none | all tune-owned fields |

The highest refusal phase is D, and the lowest study or tune-owned-state write phase is E. Both failure rows leave tune-owned Model state unchanged, as H-0094 decision 14 requires.

### 3.1 Pure lineage algorithms (`lineage.py`)

**`classify_study`** checks the conditions below in order, first match wins (§7.4). `n = len(trials)`.

| Order | Condition | State |
|---|---|---|
| 1 | No `lizyml:study` and `n > 0` | `legacy` (L) |
| 2 | No `lizyml:study` and `n == 0` | `uninitialized` (E) |
| 3 | `lizyml:study` unparsable or bad schema | `unreadable` (S); checked before N, so a study that is both S and N is S (criterion 33) |
| 4 | No round keys, no end keys, `n == 0` | `no_round` (N) |
| 5 | Any X condition below | `inconsistent` (X) |
| 6 | Last round has no `:end` | `open`, `open_round = K` (O) |
| 7 | Otherwise | `completed` / `failed` from the last `:end` |

X conditions:
- Round records missing while `n > 0`.
- Round numbers not exactly `{1..K}`.
- An `:end` without its round.
- A non-last round without `:end`.
- An undecodable round or `:end` payload.
- `start_trial` values not non-decreasing, or the last one greater than `n`.
- `n > 0` and round 1 `start_trial > 0`. Optuna numbers trials 0..n−1, so this is exactly the spec's "trial number < round 1 start".
- Recommended extension (R2): a trial param name not in the merged routes, or `merge_routes` reporting a conflict.

Other functions:
- `round_of(number)` returns the largest k with `start_trial(k) <= number`.
- `merge_routes(existing, added)` raises `RouteConflict` when the same canonical name appears with a different category. It is wired in PR 1, where it never fires; PR 2 uses it for the §4.1b refusal.
- `split_by_routes(params, routes)` routes by `routes[name].category`. An unknown name raises, which is unreachable after the X check.
- `remaining_timeout`:
  - `None` if the recorded timeout is None.
  - Otherwise `timeout − Σ(datetime_complete − datetime_start)` over COMPLETE and PRUNED trials in the round.
  - FAIL trials are excluded, including trials FAILed on reattach, whose `datetime_complete` covers the downtime (§7.2).

## 4. Changes to existing behaviour

1. **Enqueue removed (§6.2).**
   - `_run_tune_round` no longer builds `enqueue` (`_model_tuning.py:631-637, 653`).
   - `Tuner.tune` keeps `enqueue_params` and `tuner.py:200-202` for direct callers (R1).
   - The log line `tune.resume: enqueued previous best` does **not** exist in `lizyml/` (grep). It appears only in HISTORY H-0068 (:5269), so nothing in code needs removing.
2. **Base is `config ⊕ fixed` (§6.1).**
   - Every round calls `_merge_params(..., include_tuning_result=False)`. Today it passes `include_tuning_result=resume` (`:250-252`), which applies the previous best and smart values via `model.py:532-575`.
   - With `False`, `model.py:576-580` checks the fixed layer and `:640` passes `training_overrides=None`. The objective then applies `fixed` and the trial values (`:569-571`).
   - The `fit` path (`include_tuning_result=True`, the default) is unchanged.
3. **Best routing via lineage (§6.4).** `assemble_tuning_result` splits `best_trial.params` with `split_by_routes(params, lineage.routes)`. Routes are append-only, so a removed smart dim still goes to smart. Removing a dim needs `space=`, so criteria 6/6b are proven in PR 1 with synthetic lineage unit tests (pr-split.md:21).
4. **Previous best and `fixed` from the study (§5.3, bullet 3).**
   - D15 reads `study.best_trial`. `self._tuning_result` is no longer read anywhere in `tune()` (today: `:491, :503, :634-637`).
   - `fixed` is `study.fixed` for every existing study. `self._tuning_fixed_params` is only written, at commit.
   - The study attribute wins over the Model field.
5. **Round numbering from lineage.** `round_number = plan.round_number` replaces `self._round_number + 1` (`:327`). A `resume=False` new study starts again at round 1 (§6.5, last bullet). Today rounds pile up via `prior_rounds=self._rounds` (`rounds.py:57`).
6. **Trial rounds and `RoundSummary`.**
   - `TrialResult.round = lineage.round_of(number)` replaces the cumulative `n_trials` scan (`rounds.py:63-79`) and the sentinel `round=1` (`tuner.py:306-308`).
   - `RoundSummary(round=k, n_trials=<actual count of trials in k, all states>, best_score_before=record.best_before, best_score_after=end.best_after, expanded_dims=report.expanded_names or (), space_snapshot=tuple(record.effective))` (criterion 31).
   - `failed` rounds are omitted (option a). The round after one has `best_score_before = None`, because no COMPLETE trial existed when it started.
   - `boundary_report` comes from the current round's record, so an open round finished on reattach returns the report from its start (§6.5).
7. **Progress (§6.5).**
   - `current_trial = trial.number - round_start + 1`, the existing formula at `tuner.py:223`. With `round_start = record.start_trial` it yields K+1 for an open round; this also fixes P4b.
   - `total_trials` = `progress_total`, the recorded `n_trials`.
   - `cumulative_trials = trial.number + 1` (unchanged); `round = k`.
8. **Timeout (§7.2).** A new round records the config timeout and runs with it. An open round runs with `remaining_timeout`, or skips `optimize` when it is ≤ 0. The `optimize` call itself is unchanged (`tuner.py:266-273`).
9. **Failed close (§7.1).** E6: `:end failed`, then `TUNING_FAILED`, Model untouched, round omitted from `TuningResult.rounds`, next round's `best_score_before` is `None`. An exception escaping `optimize` leaves state O instead.
10. **`resume=False` onto an existing stored study is refused** (row 3). Today `Tuner` reattaches via `load_if_exists=True` (`tuner.py:171-177`); that behaviour remains only on `Tuner.tune`'s direct path.

## 5. Interaction with H-0121

- **Today:** `new_study = not resume and (storage is None or not stored_study_exists(...))` (`_model_tuning.py:244-246`) gates admission at `:260-268`.
- **New:** the gate is `plan.is_new_study`, which is true exactly for `plan_first_round`: a brand-new study (rows 1, 2, 7), state E and state N. This matches §7.4 rows E and N, "treated as a new study".
- **Refusals** (decision 3) for E and N run in D16, before E2 and E4. An E study therefore never gets `lizyml:study`, and an N study never gets a round record (criterion 19, second paragraph).
- **Drops** (decision 4) shape round 1's `declared`, as §5.2-4 says: "applied while resolving the declared space".
  - The drop is recorded and later rounds inherit it.
  - Smart values can't change between rounds (`config.model` is in the identity), so a reattach sees the same space.
- **C, F and O rounds** do not re-run `admit_smart_dimensions`. `validate_tuning_dimensions` still runs on the effective space (D17).
- **Docstring:** the module docstring at `test_smart_dimension_admission.py:14-15` and the pin at `:178-193` are rewritten (§7).

## 6. Test plan (PR 1 criteria)

### 6.1 Shared helpers — `tests/test_tuning/_resume_helpers.py`

**Configs and data**
- `fast_config(task="regression", n_trials=2, mode="replace")`:
  - built on `make_config(task, n_estimators=3, n_splits=2, tuning_n_trials=n_trials)` (`tests/_helpers.py:172-243`);
  - `cfg["model"]["auto_num_leaves"] = False`, as in `test_retune.py:779-796`;
  - space: `num_leaves` int 4–8 and `learning_rate` float 0.05–0.3 with `log`;
  - merge mode overrides `n_estimators` to int 3–5 (the `_SMALL_TREES` pattern, `test_smart_dimension_admission.py:43`).
- `make_regression_df(n=80)`. `url(tmp_path)` returns `f"sqlite:///{tmp_path/'s.db'}"`.

**Crash simulation**
- `class ProcessDeath(BaseException)`. It is neither an `Exception` nor `KeyboardInterrupt`, so Optuna's `_run_trial` doesn't catch it (`optuna/study/_optimize.py:211`) and the trial stays RUNNING. Verified in memory; for SQLite this is inferred from the same code path.
- `die_in_objective(monkeypatch, model, at_call)` wraps `model._build_train_components`, which runs once per trial (the instance-patch pattern from `test_space_merge.py:256-260`).
- `die_at_attr(monkeypatch, key)` wraps `lizyml.tuning.tuner.write_study_attr` and raises `ProcessDeath` **before** writing a matching key. This simulates crashes between write positions (a) and (b), before `lizyml:round:k`, and before `:end`.
- `run_child_and_kill(script)`: one real subprocess crash using `os._exit(1)` in the objective, following the pattern in `probes/optuna_facts.py:18-27`. Run it with `cwd` set to the repo root so `tests._helpers` imports.

**State inspection**
- `sqlite_exec(url, sql, params)` uses stdlib `sqlite3` on table `study_user_attributes` (`optuna/storages/_rdb/models.py:109`, columns `key` and `value_json`). It deletes or renames `lizyml:` keys to build L and X states from real studies. Optuna has no attribute-delete API; `set_user_attr` can only overwrite.
- `snapshot_study(study_or_url, name)` returns `[(t.number, t.state.name, t.values, t.params)]` plus the `lizyml:*` user attrs. Storage studies are reloaded with `optuna.load_study`.
- `snapshot_model(model)`:
  - Identity (`is`) for `_study`, `_tuning_result`, `_X`, `_y`, `_fit_result`.
  - Copy equality for `_space`, `_round_number`, `_rounds`, `_used_default_space`, `_tuning_fixed_params`, `_tuning_storage`, `_tuning_study_name`, `_run_dir`.
  - `_block_values`, `_input_categories` and `_provider` are excluded on purpose. They are per-call transients (`model.py:172, 177-178`) that `fit()` recomputes before reading (`model.py:236, 272`).
- **Templates:** a module-scoped fixture builds one C study with 2 rounds in a SQLite file. Each test `shutil.copy`s it into its own `tmp_path`, so rows stay isolated and cheap.

### 6.2 Unit tests (pure)
- **`tests/test_tuning/test_lineage_codec.py`:**
  - round trip for every dim kind (bool/int/float/NaN/±inf/None/str choices, `IntDim` beyond 2**53, `FloatDim` with an int `low`, bounds set/unset);
  - `True`, `1` and `1.0` stay distinct;
  - `BoundaryReport` round trip;
  - a numpy scalar raises `LineageCodecError`;
  - `to_attr_value` turns tuples into lists.
- **`tests/test_tuning/test_lineage_classify.py`:**
  - every state from synthetic attrs and `TrialLike` stand-ins: L, E, S (bad and too-new schema), N, each X variant, O, C, F;
  - S∧N → `unreadable` (criterion 33);
  - `round_of`; `remaining_trials` counting RUNNING;
  - `remaining_timeout`: COMPLETE and PRUNED counted, FAIL excluded, None timeout → None, exhausted → ≤ 0 (criterion 16);
  - `merge_routes` conflict;
  - `split_by_routes` with a removed smart dim → smart and a removed training dim → training (criteria 6/6b, per pr-split.md:21).
- **`tests/test_tuning/test_objective_identity.py`:**
  - `summarise_frame` changes only the expected element (criteria 13 and 26 at element level);
  - config parts are isolated from each other (criterion 27);
  - the `fixed` monkeypatch changes `config.fixed` (criterion 28).

### 6.3 Criterion map

| # | File :: test(s) | How |
|---|---|---|
| 1 | `test_resume_identity.py::test_objective_element_change_refused[...]`, `::test_non_identity_change_passes[...]` | Start from the storage C template and build a new `Model` with one change each: a data value, row order reversed, `split.n_splits`, `training.seed`, a `model.params` value, `num_leaves_ratio`, the `features.exclude` list, or the first metric rmse→mae. Each → `CONFIG_INVALID`, with trial count and `snapshot_model` unchanged. Pass group, on a binary fast config: `n_trials` argument, config timeout, `progress_callback`, `expand_boundary`, `boundary_threshold`, an appended second metric, `calibration` added, `output_dir` → round 3 runs. |
| 2 | `::test_legacy_study_refused[sqlite_deleted, raw_optuna, in_memory]` | (a) `sqlite_exec` deletes `lizyml:%` from a C study; (b) a raw `optuna.create_study` + `optimize` with no attrs; (c) `model._study` set to an attr-less study. All → `CONFIG_INVALID`. |
| 3 | `::test_version_difference_warns_and_continues` | Overwrite `lizyml:study.versions` with `set_user_attr` → `pytest.warns(UserWarning, match="lightgbm")`, and the round runs. |
| 5 | `test_resume_lineage.py::test_unchanged_declaration_inherits_expanded_range[memory, reattach]` | Round 2 with `expand_boundary=True` expands; round 3's record starts from round 2's effective space. |
| 8 | `test_resume_open_round.py::test_crash_mid_round2_reattach_finishes_round2[in_process, new_model, subprocess]` | Round 2 `n_trials=3`, die at the 2nd trial; new Model `resume=True`. Round 2 ends with 3 trials, the RUNNING trial is FAIL, `rounds` and trial rounds match an uninterrupted run, `boundary_report == decode(record)`, no `tune.expand` log. In-process variant: same Model, `_run_tune_round` raises. |
| 8 | `::test_crash_before_end_closes_without_trials` | `die_at_attr("lizyml:round:2:end")`; reattach does not call `Tuner.run` and writes `:end`. |
| 8a | `::test_single_running_trial_round1_fails_then_round2` | Round 1 `n_trials=1`, die at its only trial. Reattach → `TUNING_FAILED` and `:end failed`. `resume=True` again → round 2 completes; `rounds == (round 2,)`, `best_score_before is None`, trial 0 is round 1 / fail, `boundary_report is None`. |
| 8b (part) | `::test_u2_same_args_pass[n_trials, expand_boundary, omitted, none]`, `::test_u2_different_args_refused[n_trials, expand_boundary]`, `::test_config_n_trials_warns`, `::test_after_closed_round_n_trials_free` | Run on state O with a RUNNING trial. Refusals leave that trial RUNNING and the attrs unchanged. The warning matches `"n_trials"` and the round still uses the recorded budget. |
| 9 | `test_resume_call_matrix.py::test_resume_false_onto_existing_refused[completed, uninitialized]` | Plus the rewritten H-0121 pin (§7). |
| 10 | `::test_resume_other_storage_refused` | |
| 11 | `test_resume_identity.py::test_config_space_change_refused[space, space_mode][memory, reattach]` | In memory, mutate `model._cfg`; for reattach, build a new Model from an edited config. |
| 12 (part) | `test_resume_call_matrix.py::test_storage_only_or_name_only_refused[resume F/T]`, `::test_unreadable_schema_refused` | Set `lizyml:study` to `{"schema": 999}`. |
| 13 | `test_resume_identity.py::test_data_identity_refusals[...]` | Rename a column; reorder categories; change categorical `[1, "1"]` to `["1", 1]` (put the column in `features.exclude` if the builder rejects mixed categories); pass data through the constructor; rewrite the `data.path` CSV in place. |
| 14 (part) | `test_resume_call_matrix.py` (see list below) | |
| 15 (part) | `test_resume_lineage.py::test_resume_round_has_no_enqueued_trial` | No round-2 trial carries `"fixed_params"` in `system_attrs` (Optuna's enqueue marker); the round-2 count equals `n_trials`; best equals the study best. |
| 15 (part) | `::test_stop_before_round_record_is_plain_resume`, `::test_stop_after_round_record_is_open` | Stopping before the `lizyml:round:2` write leaves state C, and the next resume runs round 2. Stopping after it leaves state O. |
| 16 | `test_resume_open_round.py::test_timeout_remaining_passed`, `::test_timeout_exhausted_closes_without_trials`, `::test_second_crash_excludes_failed_time` | Config timeout 3600; a spy on `Tuner.__init__` sees `timeout` ≈ 3600 − Σ(COMPLETE + PRUNED spans). For exhaustion, `sqlite_exec` sets the round-2 budget timeout to `["float", 1e-6]`: no optimize, `:end` is written. For a second crash, die again on the first resumed trial; the second reattach's timeout excludes the FAIL trials. |
| 18 | `::test_progress_continues_k_plus_1[memory, storage]` | `current_trial` starts at K+1 and `total_trials` equals the recorded `n_trials`, the same with and without storage. |
| 19 (part) | `test_resume_lineage.py` (see list below) | |
| 21 | rewrites in §7 | |
| 22 | `test_resume_refusal_invariance.py::test_refusal_changes_nothing[row]` | Table in §6.4. |
| 23 | `test_resume_lineage.py::test_resume_false_new_study_restarts_lineage[memory, new_name]` | `rounds == (round 1,)`, every trial is round 1, `lizyml:round:1` exists. |
| 24 | `test_resume_call_matrix.py::test_after_load_resume_without_storage_tuning_failed` | The existing export/load golden tests must pass unchanged. |
| 25 | `::test_reattached_same_target_continues_and_name_change_refused` | |
| 26 | `test_objective_identity.py` (unit) and `test_resume_identity.py::test_data_identity_refusals[index_labels, unused_category, column_order, int64_float64, int64_Int64, ordered, object_1_vs_str, excluded_values]` | |
| 27 | `test_resume_identity.py::test_mismatch_context_names_element[data, config, metric]` | `context["mismatched"]` contains `data.values`, `config.model` or `metric` respectively. |
| 28 | `::test_provider_fixed_change_refused[versions_same, versions_changed]` | `monkeypatch.setattr(LGBMProvider, "default_fixed_params", ...)`. |
| 30.2 | `test_resume_lineage.py::test_loaded_model_expands_from_study_best` | Export and load a Model tuned on another study, then reattach to study A with `expand_boundary=True`. The `boundary_report` best values equal A's best trial. |
| 30.3 | `::test_fixed_from_study_not_model_field[cleared, overwritten, widget_transfer]` | Set `_tuning_fixed_params` to None or `{"first_metric_only": False}`, or pickle `_study` onto a fresh Model. The spied trial `model_params["first_metric_only"]` is `True`, and the committed field equals the study's `fixed`. |
| 31 | `::test_roundsummary_counts_actual_trials_on_timeout` | Timeout 0.5 s with `n_trials` 1000. Open-round report equality is checked together with 8; `total`, `cumulative` and `round` together with 18. |
| 32 | `::test_end_record_contents` | `completed` has a float `best_after` and an ISO `ended_at`; `failed` has `["none", null]`. |
| 33 | `test_resume_call_matrix.py::test_unreadable_and_empty_is_unreadable` | |

Criterion 14 (part), in `test_resume_call_matrix.py`:
- `::test_same_storage_continues[str, same_object, omitted]`.
- `::test_different_target_refused[study_name, url_string, other_storage_object, memory_plus_storage]`. Two `RDBStorage(url)` instances give two different objects; `"sqlite:///./" + relpath` is the same file spelled differently.
- `::test_reattach_existing`.
- `::test_resume_creates_missing_study_with_info_log`: caplog shows `tune.study_created`, `lizyml:study` exists, round 1 runs.
- `::test_resume_without_study_or_storage_tuning_failed`.

Criterion 19 (part), in `test_resume_lineage.py`:
- `::test_stop_before_study_record_is_uninitialised`: `die_at_attr("lizyml:study")` leaves state E; `resume=True` initialises it and runs round 1.
- `::test_legacy_refused`.
- `::test_no_round_starts_round1`: `die_at_attr("lizyml:round:1")`.
- `::test_inconsistent_refused[no_rounds_with_trials, gap, missing_mid_end, start_gt_0]`, using `sqlite_exec` on a 2-round study.
- `::test_uninitialised_admission_failure_writes_nothing` and `::test_no_round_admission_failure_writes_nothing`: an empty study plus a `lizyml:study` built with the `_tuning_resume` helpers for a config whose user `num_leaves_ratio` dimension has its owner off.
- `::test_objective_crash_is_open_and_counts_fail`.

### 6.4 Criterion 22 table (PR 1 rows)

Each row builds its state, takes `before = (snapshot_study, snapshot_model)`, calls `tune`, checks the error code, and checks that both snapshots are unchanged.

| Row | State / setup | Expected |
|---|---|---|
| `storage_no_name[resume F/T]` | in-memory tuned Model | `CONFIG_INVALID` |
| `name_no_storage[resume F/T]` | in-memory tuned Model | `CONFIG_INVALID` |
| `resume_false_existing[C, E]` | stored study | `CONFIG_INVALID` |
| `other_target[name, url_string, storage_object, memory_plus_storage]` | | `CONFIG_INVALID` |
| `identity[data, config, metric]` | C | `CONFIG_INVALID` |
| `legacy` | L | `CONFIG_INVALID` |
| `unreadable` | S | `CONFIG_INVALID` |
| `unreadable_empty` | S∧N | `CONFIG_INVALID` |
| `inconsistent[4]` | X | `CONFIG_INVALID` |
| `config_space`, `config_space_mode` | C | `CONFIG_INVALID` |
| `open_n_trials`, `open_expand` | O with RUNNING trial | `CONFIG_INVALID` |
| `open_identity` | O with RUNNING trial + data change | `CONFIG_INVALID` |
| `no_round_identity` | N + data change | `CONFIG_INVALID` |
| `uninitialised_admission` | E | `CONFIG_INVALID` |
| `no_round_admission` | N | `CONFIG_INVALID` |
| `direction_mismatch` | E study with the opposite direction | `CONFIG_INVALID` |
| `no_study_no_storage` | fresh Model | `TUNING_FAILED` |

## 7. Existing tests that pin old behaviour

Criterion 21 cites line numbers at `6273d39`. Current locations were found by content; H-0121 shifted them by about 6 lines.

| Test (current location) | What it pins now | Rewrite (keep the test) |
|---|---|---|
| `test_space_merge.py:190` `test_real_tune_retains_defaults_and_resume_policy` (spec :183) | A config `space_mode`/`space` edit is ignored without notice on resume (P3) | Keep the first half. Then assert `resume=True` after the edit is `CONFIG_INVALID` and leaves `_space`, `_tuning_fixed_params` and the trial count unchanged. Restore the config; resume passes; `fit` still has `first_metric_only is True`. |
| `:269` `test_removed_dimensions_match_trial_and_fit[resume]` (spec :262) | `resume=True` after a replace-mode edit keeps the old dim and applies the previous best | `resume=False` branch unchanged. `resume=True` → `CONFIG_INVALID` (§5.1), state unchanged. Removed-dim routing is covered by the unit tests (6/6b); the full flow comes in PR 2 with `space=`. |
| `:311` `test_export_load_refit_retains_fixed_policy[mode]` (spec :305) | Mutating `space_mode` and then resuming uses the stored fixed | Resume without editing; mutate `space_mode` after tune. The point stays that fit and export follow the tuning round's fixed, not the current config. Keep the load assertions. Add: a resume after the mutation → `CONFIG_INVALID`. |
| `:381` `test_removed_training_dimension_admission[resume]` (spec :375 / :395) | `resume=True` after editing `model.params` and the space → "already controlled" | `resume=True` → `CONFIG_INVALID`. The §5.1 check runs first because the space was edited, so assert the code and the space surface. `fit()` still raises "already controlled". `resume=False` branch unchanged. |
| `test_tuner_persistence.py:252` `test_model_tune_load_if_exists_resumes_across_instances` | `resume=False` reattaches | `m2.tune(storage, name)` → `CONFIG_INVALID`, still 2 trials. Then `m2.tune(storage, name, resume=True)` → 4 trials, rounds 1 and 2, `lizyml:round:2` present. Rename to `test_model_tune_reattaches_only_with_resume`. |
| `test_retune.py:807` `test_resume_without_prior_tune_raises` | `TUNING_FAILED` (still correct, row 8) | Keep it. Add `test_resume_with_storage_but_no_name_is_config_invalid_first` for the new order. The message still contains "resume". |
| `test_retune.py:834` comment | "enqueue counts as 1 of the 2 trials" | Replace: round 2 runs exactly `n_trials=2` TPE trials (no enqueue, §6.2); 3 + 2 = 5 is unchanged. |
| `test_error_code_raising.py:57` `_resume_without_a_study` (`:133-136` expects `{"resume", "round_number"}`) | Old raise site | Rename to `_resume_without_study_or_storage`. Expected keys `{"resume", "round_number", "storage_given"}`. The subset check passes, and the context is never all-None (`test_no_empty_context.py`). |
| `test_smart_dimension_admission.py:178` `test_reattaching_to_a_stored_study_is_out_of_scope` (+ docstring :14-15) | H-0121 temporary pin: `resume=False` onto a stored empty (E) study reattaches and keeps the ratio | Rewrite as `test_resume_false_onto_a_stored_study_is_refused`: criterion 9, `CONFIG_INVALID`, no trials, no `lizyml:` attrs. Add `test_resume_true_onto_an_uninitialised_study_is_a_new_study_and_drops`: E counts as new, the ratio is dropped, `lizyml:study` is written. Docstring: "a tune that creates a new study (an absent name, or a stored study in state E/N)". |

**Other tests affected (not pins)**
- `test_space_merge.py:447` patches `Tuner.tune`, which Model no longer calls. Retarget it to `lizyml.tuning.tuner.create_study` so the guard isn't vacuous.
- `test_space_merge.py:36, 134, 170` drop the `resume=False` keyword.
- `test_space_merge.py:369` and `test_fit_params_override.py:3090` patch `_run_tune_round`. They stay valid: the name is kept, and it runs before commit.
- `test_knob_reachability.py:539-556, 459-462`: one `Tuner` with `storage`/`study_name`, and config `n_trials`/`timeout` reach the fresh tune's Tuner. E0 satisfies both.
- `patch("optuna.create_study")` guards in `test_dimension_consumption.py:47, 59, 87`, `test_smart_dimension_admission.py:62` and `test_direction_reconciliation.py:61` stay valid: creation goes through `_optuna.create_study`, and every refusal comes before E1.
- **Do not change** the Tuner-direct tests `test_tuner_persistence.py:76-221, 277-298` and `test_direction_reconciliation.py:94-119` (criterion 21).
- These pass as they are; counts are now exact:
  - `tests/test_persistence/_load_census.py:221-224` (in-memory resume, unchanged config);
  - `test_retune.py` `TestModelRetuneResume` and `TestBoundaryEdgeCases`;
  - `test_retune.py:648-676` (5 resume rounds).

**Enqueue and reattach mentions outside code** — PR 2 docs scope, no test asserts the text:
- `docs/api.md:88`, `CHANGELOG.md:383`, `BLUEPRINT.md:1122` ("before enqueue").
- `notebooks/tutorial_regression_tuning_lgbm.ipynb` cell near :426; its code path still works.
- grep over `tests/test_docs` and `tests/test_notebooks` found nothing.

## 8. Risks and open questions

Each item has a recommended resolution.

- **R1. "Remove the enqueue at `tuner.py` ~201" conflicts with criterion 21.** `test_direction_reconciliation.py:111-115` calls `Tuner.tune(enqueue_params=...)`. **Recommendation:** the Model stops passing it and Tuner keeps the parameter. §6.2's intent is met because no Model path enqueues.
- **R2. Malformed lineage beyond the §7.4 X list:**
  - orphan `:end`;
  - undecodable round or `:end` payloads;
  - non-monotone `start_trial`, or one greater than the trial count;
  - malformed `lizyml:round:*` keys;
  - unrouted trial param names;
  - route category conflicts.

  LizyML's own writes can't produce any of these.

  **Recommendation:** classify all as X. That fits conservative identity, and the X message ("record and trials disagree") applies. It extends a table the spec calls closed, so record it in the PR 1 decision record. Other unknown `lizyml:*` keys are ignored.
- **R3. State N and §5.1.** §7.4 row N says "round 1 like a new study (config space)" but doesn't say whether to compare against the already-written `origin_space` and `space_mode`. **Recommendation:** compare (C13a applies to N) so `lizyml:study` and round 1 can never disagree.
- **R4. `data.path` and identity.** §4.1 lists "data settings" but also requires every route to summarise the read content. **Recommendation:** exclude the `data.path` string. Constructor data and a `data.path` read of the same frame then match, and changed file content is still refused (criterion 13).
- **R5. Config `n_trials` warning when an `n_trials` argument is passed.** **Recommendation:** warn only when `n_trials is None`, because with an argument the config value isn't the one being used.
- **R6. `boundary_threshold` on an open round** is outside U2, so a different value is ignored without error (DC1-like). **Recommendation:** follow the spec (no refusal) and add `INFO event='tune.resume_open' ignored=boundary_threshold`. Ask the maintainer if a refusal is wanted.
- **R7. Storage schema initialisation.** `optuna.get_all_study_names(url)` constructs `RDBStorage`, which creates the SQLite file and schema on first contact (C9), before some refusals. No study or attribute rows are written, so INV-R4 (about the study) holds. PR 2's criterion 14 wording ("nothing written to storage" for `space=` refusals) should accept this, or PR 2 must move the existence check after all refusals on the creation path.
- **R8. Per-call transients.** `_block_values` and `_input_categories` are written by `_prepare_training_data` (`model.py:763, 783, 805`) before refusals. **Recommendation:** leave them, and exclude them from the criterion 22 snapshot with a comment, because `fit` recomputes them before reading. `_ensure_run_dir` and `self._provider` move to E0.
- **R9. Unhashable object cells.** `hash_pandas_object` raises `TypeError` on list or dict cells, which can occur in excluded columns. **Recommendation:** fall back to hashing `col.map(repr)` for that column.
- **R10. Pre-change pickled Models** (LizyStudio cloudpickle, spec §10) lack the new fields. Read them with `getattr(..., None)`. Their study has no attrs, so it reaches L and the documented `CONFIG_INVALID`.
- **R11. INV-R5 and sampler state.** A reattach builds a fresh `TPESampler(seed)`, so sampled values after a crash differ from an uninterrupted run. INV-R5 and criterion 8 promise space, lineage, round numbers and counts only. Tests must not compare trial values across crash and no-crash runs. For the criterion-31 report, compare against the decoded round record.
- **R12. §5.3 bullets 1–2** (changed/new dims and an out-of-range best are not expanded) are PR 2. In PR 1 the effective range only grows and every trial was sampled inside some earlier effective range, so the best is always in range and bullet 2 can't trigger. Leave a hook: `_expand_for_next_round(..., frozen_names: frozenset[str] = frozenset())`.
- **R13. Criteria 6/6b and 12 (category change)** can't be exercised end to end without `space=`. PR 1 ships `split_by_routes` and `merge_routes` with unit tests (pr-split.md:21). The refusal wiring and end-to-end tests come in PR 2.
- **R14. Reporting surfaces between the PRs.** `plots/tuning.py:90-118` still sums `RoundSummary.n_trials`. With actual counts and omitted failed rounds, round separators can shift until PR 2 (16a), as accepted in pr-split.md:25.
- **R15. Training/model name collision.** `early_stopping_rounds` as a training dim versus a model dim spelled the same way, which canonicalises to `early_stopping_round`. Routes keep training names raw, so there is no false conflict. A model dim with that spelling is already refused by `check_training_managed_space` (`_model_tuning.py:229`).
- **R16. WAITING trials** (left by old enqueues) exist only in pre-change studies, which are L. No special handling.
- **R17. Check precedence.** Moving the config-space admission checks (`:217-237`) after C13a means an edited config space gets the §5.1 message instead of an unknown-name message. Either way it is `CONFIG_INVALID` before any write. PR 2 retargets these checks to the round's declared space (§5.2-4).
- **R18. Firing-rate note** (the spec's last line). After implementing, run the existing tune/storage tests and the `design-vs-implementation.md` probes P1–P7 through the new checks, and record the refusal count in the PR 1 decision record. `probes/probe_resume.py` and `probes/probe_p4b.py` are cited (`design-vs-implementation.md:68`), but only `optuna_facts.py` is in the worktree; find or recreate them.

**Suggested TDD order**
1. codec
2. classifier / lineage
3. identity
4. tuner seams (`run`, study I/O, direction)
5. `_tuning_plan` / `_tuning_resume`
6. `_model_tuning` wiring and commit
7. pinned-test rewrites (§7)
8. integration files in the order of §6.3
9. the criterion 22 table

### Critical Files for Implementation
- /home/rem/repos/LizyML/lizyml/core/_model_tuning.py
- /home/rem/repos/LizyML/lizyml/tuning/tuner.py
- /home/rem/repos/LizyML/lizyml/tuning/rounds.py
- /home/rem/repos/LizyML/lizyml/tuning/search_space.py
- /home/rem/repos/LizyML/lizyml/core/model.py
- New: /home/rem/repos/LizyML/lizyml/tuning/lineage.py, /home/rem/repos/LizyML/lizyml/tuning/lineage_codec.py, /home/rem/repos/LizyML/lizyml/core/_tuning_identity.py, /home/rem/repos/LizyML/lizyml/core/_tuning_resume.py, /home/rem/repos/LizyML/lizyml/core/_tuning_plan.py
- Tests: /home/rem/repos/LizyML/tests/test_tuning/test_space_merge.py, /home/rem/repos/LizyML/tests/test_tuning/test_tuner_persistence.py, /home/rem/repos/LizyML/tests/test_tuning/test_retune.py, /home/rem/repos/LizyML/tests/test_core/test_error_code_raising.py, /home/rem/repos/LizyML/tests/test_tuning/test_smart_dimension_admission.py