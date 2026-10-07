# Deprecation Registry

Central tracking of deprecated public surfaces and their removal targets.
Every `DeprecationWarning` / `UserWarning` raised by LizyML lists "Will be
removed in v1.0." in its message; this file is the single source of truth.

The CI test `tests/test_core/test_deprecation_registry.py` asserts that
every deprecation warning emitted during a representative test run carries
a removal-version suffix matching the pattern `Will be removed in v\d+\.\d+`.

## Schedule

| Target | Replacement | Removal | Deprecated since |
|---|---|---|---|
| `EarlyStoppingConfig.validation_ratio` (input) | `inner_valid.ratio` | **v1.0** | H-0069 (2026-04, made `computed_field` in #111) |
| `CalibrationConfig.n_splits` | (removed; outer split is reused) | **v1.0** | H-0058 (2026-04) |
| `purged_time_series.purge_window` | `purge_gap` | **v1.0** | H-0038 |
| `purged_time_series.embargo` | `purge_gap` (the value is added to it) | **v1.0** | H-0115 (2026-10) |
| `purged_time_series.embargo_pct` | `purge_gap` (int, observation count; added) | **v1.0** | H-0040 (target changed from `embargo` to `purge_gap` by H-0115) |
| `purged_time_series.gap` | `purge_gap` (added) | **v1.0** | H-0038 (target changed from `embargo` to `purge_gap` by H-0115) |
| `PurgedTimeSeriesSplitter(embargo=...)` | `purge_gap` (the value is added to it) | **v1.0** | H-0115 (2026-10) |
| `lizyml.core._model_factories.build_calibration_splitter` | (removed; outer split is reused) | **v1.0** | H-0058 |
| `LGBMConfig.params["objective"]` silently stripped (cross-task) | Raise `LizyMLError(CONFIG_INVALID)` at fit time | **already enforced** | H-0079 (2026-05) |
| `ErrorCode.DATA_FINGERPRINT_MISMATCH` | (none -- nothing ever raised it; missing columns raise `DATA_SCHEMA_INVALID`, numeric columns arriving non-numeric raise `INCOMPATIBLE_COLUMNS`) | **removed** (breaking: code that references the member gets `AttributeError`) | H-0106 (2026-10) |

## Migration notes

### `validation_ratio` → `inner_valid.ratio`

```yaml
# Before
training:
  early_stopping:
    enabled: true
    rounds: 50
    validation_ratio: 0.15

# After
training:
  early_stopping:
    enabled: true
    rounds: 50
    inner_valid:
      method: holdout
      ratio: 0.15
```

`validation_ratio` is now a `computed_field` mirroring `inner_valid.ratio`,
so `model_dump()` round-trips remain stable. Reading the field via
`config.training.early_stopping.validation_ratio` keeps working until v1.0.

### `calibration.n_splits` (removed)

The field is silently ignored; calibration cross-fit reuses the outer CV
splits (H-0058). Drop the key from your config:

```yaml
# Before
calibration:
  method: platt
  n_splits: 5

# After
calibration:
  method: platt
```

### `purged_time_series` keys

`purge_window` and `gap` were obs-count integers but had distinct semantics
that have since been unified.

```yaml
# Before
split:
  method: purged_time_series
  purge_window: 5
  embargo_pct: 0.05

# After
split:
  method: purged_time_series
  purge_gap: 15      # 5 + 10: explicit observation counts, not a fraction
```

### `purged_time_series.embargo` merged into `purge_gap` (H-0115)

`embargo` subtracted at the same position as `purge_gap` (the end of each
training block), so the two were one knob, and the splitter never places
training rows after the validation block, where an embargo would act. Write
the total in `purge_gap`: `{purge_gap: 5, embargo: 2}` becomes
`{purge_gap: 7}`. The folds, the inner-validation gap and the calibration
folds are unchanged. Until v1.0, `embargo` (and `embargo_pct` / `gap`) is
accepted with a `DeprecationWarning`, even when `0`, and added to
`purge_gap`; at most one of the three may be given.

**Removal note for v1.0.** `metadata.json` of every `purged_time_series`
artifact exported before H-0115 stores `embargo` (usually `0`) in its
config, and `Model.load()` re-validates that config. When the key is
removed, the load path must keep normalizing it into `purge_gap`, or those
artifacts will be refused by `extra="forbid"`.

### `LGBMConfig.params["objective"]` cross-task injection (H-0079)

Pre-0.15 the `LGBMAdapter._build_params()` body popped `objective` from
the user-supplied params unconditionally and re-set
`_TASK_OBJECTIVE[task]`. Cross-task values like
`task="binary", params={"objective": "regression"}` were silently
dropped — same defensive intent, but no signal to the user.

From v0.15 the same defensive contract is enforced explicitly:

```python
# Before (v0.14 and earlier): silently dropped
LGBMConfig(task="binary", params={"objective": "regression"})  # trained with "binary"

# After (v0.15+): raises CONFIG_INVALID at fit / _build_params
LGBMConfig(task="binary", params={"objective": "regression"})
# LizyMLError: objective 'regression' is not compatible with task 'binary'.
# Valid: ['binary', 'cross_entropy', 'cross_entropy_lambda'].
```

For same-task values (e.g. `task="regression"`, `objective="fair"`) the
silent strip was a **bug** rather than a deprecated contract:
`default_space("regression")` already exposed `objective` as a tunable,
yet the sampled value was discarded. v0.15 honours those values, so
`tune` may now report a different `best_params`/`best_score` for
identical config + data.

### `build_calibration_splitter()` (removed)

Internal API. Users hand-rolling calibration cross-fit should switch to
`fit_result.splits.outer` (already populated from the CV trainer).

## Adding a new deprecation

1. Append "Will be removed in vX.Y." to the warning message.
2. Add a row to the table above.
3. Document the migration path in this file.
4. Ensure tests using the legacy form wrap calls in
   `pytest.warns(DeprecationWarning)` so the deprecation contract is
   exercised in CI.

## Related

- HISTORY.md: H-0076 (this registry), H-0058 (calibration), H-0069
  (`validation_ratio`), H-0021 (purged_time_series).
- Code: `lizyml/config/schema.py`, `lizyml/core/_model_factories.py`.
