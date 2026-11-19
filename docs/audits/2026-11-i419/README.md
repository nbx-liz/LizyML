# H-0149 measurements (#419): seeded TPESampler sequences across Optuna versions

Measured on 2026-11-20 on develop `6b15200`, machine V, CPython 3.12, NumPy 2.5.5, one
virtualenv per Optuna version (`probes/run_all.sh` states how each was made). The probes
import only Optuna, not LizyML.

| Output | Probe | What it shows |
|---|---|---|
| `tpe-sequences.out` | `probes/tpe_sequences.py` | For 16 Optuna versions, the sha256 prefix of a 40-trial seeded sequence under five sampler variants, and trials 9 and 10 |
| `explicit-options-band.out` | `probes/explicit_options_band.py` | The sequence with the H-0149 options (`multivariate=True`, `constant_liar=True`), and the warnings that constructing that sampler emits |
| `load-writes-nothing.out` | `probes/load_writes_nothing.py` | Every SQLite table of a stored study fingerprinted before and after `optuna.load_study` |
| `widened-dimension.out` | `probes/widened_dimension.py` | A study whose second round widens one dimension: the relative (multivariate) search space at the end of each round |

The objective is a deterministic function of five parameters (log float, float, log int,
int, categorical); it stands in for a training run, so a sequence depends on the sampler only.

## Results

Sequence digests (`tpe-sequences.out`), columns `default`, `v4_options`, `v5_options`,
`multivariate_only`, `constant_liar_only`:

| Optuna | default | 4.x options | 5.x options |
|---|---|---|---|
| 3.1.0, 3.2.0 | `99ffa19f…` | `99ffa19f…` | `55b1dd02…` |
| 3.3.0, 3.4.0 | `99ffa19f…` | `99ffa19f…` | `94e404ff…` |
| 3.5.0 to 4.7.0 (3.5.0, 3.6.0, 3.6.1, 4.0.0 to 4.7.0) | `38544e35…` | `38544e35…` | `7a6bd437…` |
| 5.0.0 | `4a691447…` | `d323306b…` | `4a691447…` |

- Under 5.0.0, passing the 4.x options does not give the 4.x sequence, and under 4.7.0
  passing the 5.x options does not give the 5.0.0 sequence. 5.0 changed the Parzen
  estimator's bandwidth computation (`optuna/samplers/_tpe/parzen_estimator.py`,
  `compute_sigmas`); no constructor argument restores the old one.
- Sequences of different bands first differ at trial 10, the first trial after the ten
  random startup trials.
- `multivariate_only` equals `v5_options` and `constant_liar_only` equals `v4_options` in
  every version: in a sequential study (`n_jobs=1`, no RUNNING trial of another worker)
  `constant_liar` changes nothing.
- Constructing the sampler with `multivariate=True` and `constant_liar=True` emits two
  `optuna.exceptions.ExperimentalWarning` under every version below 5.0.0 and none under
  5.0.0 (`explicit-options-band.out`). Optimizing emits no Python warning in any version.
- `optuna.load_study` leaves every table unchanged under 3.1.0, 4.7.0 and 5.0.0.
- When a later round widens a dimension, that dimension leaves the relative search space
  (it is then sampled independently) under 3.1.0, 4.7.0 and 5.0.0 alike.

Limits: one seed, one search space, 40 trials, one platform. Patch releases other than
3.6.1 were not measured.
