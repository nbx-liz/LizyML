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
| `purged_time_series.purge_window` | `purge_gap` | **v1.0** | H-0021 |
| `purged_time_series.embargo_pct` | `embargo` (int, observation count) | **v1.0** | H-0021 |
| `purged_time_series.gap` | `embargo` | **v1.0** | H-0021 |
| `lizyml.core._model_factories.build_calibration_splitter` | (removed; outer split is reused) | **v1.0** | H-0058 |
| `LGBMConfig.params["objective"]` silently stripped (cross-task) | Raise `LizyMLError(CONFIG_INVALID)` at fit time | **already enforced** | H-0079 (2026-05) |

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
  purge_gap: 5
  embargo: 10        # explicit observation count, not a fraction
```

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
