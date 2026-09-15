# PR 3c code review, round 1 (Codex gpt-6-astra, effort medium, read-only, head f67f8c7, 2026-09-15)

Prompt: `prompt-templates/pr3c-code-review-round1.md`. The first run was killed by the host for low memory before a verdict; this is the complete second run.

VERDICT: REQUEST_CHANGES

## Findings

### 1. Blocking · B1 — malformed optimizer methods escape as `TypeError`

**Location:** [lizyml/calibration/_optimizer.py:125](/home/rem/repos/LizyML/lizyml/calibration/_optimizer.py:125)  
**Contract:** criteria 6a/6c.

The membership check `method not in METHODS` runs before checking that `method` is a string. Lists and dictionaries survive value normalization, then fail because they are unhashable.

**Reproduced:** using `make_config("binary", n_estimators=5, n_splits=3, calibration=method, calibration_params={"method": []}, tuning_n_trials=1)` and `make_binary_df(n=240, seed=4)`:

- Both Platt and Beta, through both `Model.fit()` and `Model.tune()`, raise `TypeError: unhashable type: 'list'`.
- `{"method": {}}` also reproduces through Platt fit.
- **Expected:** `CONFIG_INVALID` identifying `calibration.params` and the invalid method.

Training stops, but the promised error contract is broken.

### 2. Blocking · B3, silent-discard exception — generated fitters omit the runtime acceptance contract

**Location:** [lizyml/codegen/templates.py:563](/home/rem/repos/LizyML/lizyml/codegen/templates.py:563)  
**Defect classes:** DC1, DC3.

`_minimize_kwargs` reads recognized keys and discards everything else. Neither generated Platt nor generated Beta performs the runtime parameter validation.

**Reproduced:** executed the actual fitter functions extracted from `render_train_py()` in memory, without changing their bodies. For 600 scores generated with RNG seed 0 (`normal(0,2)`, labels sampled from `expit(0.5*s+1)`):

- Runtime Platt/Beta with `{"C": 0.001}` raise `CONFIG_INVALID`.
- Generated Platt/Beta with exactly those parameters return fitted coefficients, silently ignoring `C`.

**Expected:** the same refusal contract. H-0059 describes `config.json` as editable; H-0100 decision 4 requires generated fitters to reproduce the optimizer contract. Editing the newly exported `calibration_params` therefore opens the accept-and-discard path again. Originally exported, validated settings do not trigger this example.

This is B3’s explicit blocking exception for parameters discarded without refusal.

### 3. Should-change · B3 — failed Platt optimization is silently accepted as fitted

**Location:** [lizyml/calibration/platt.py:134](/home/rem/repos/LizyML/lizyml/calibration/platt.py:134), also [generated Platt:617](/home/rem/repos/LizyML/lizyml/codegen/templates.py:617)  
**Defect class:** DC1.

The implementation copies `result.x` without inspecting or exposing `result.success`.

**Reproduced through `Model.fit()`:** same 240-row fixture and configuration as finding 1, with Platt parameters `{"options": {"maxls": 1}}`. A recording wrapper called the real SciPy optimizer:

- The quadratic validation probes succeeded.
- All three calibration folds and C_final returned `success=False`, message `ABNORMAL:`.
- `Model.fit()` nevertheless succeeded. C_final exported `a=0.0`, `b=0.282862786015832`, its initialization.
- No optimization warning appeared; only the existing binary/kfold advisory appeared.

**Expected:** failed calibration optimization must be surfaced instead of presenting the initial coefficients as a successful fit. The generated fitter has the same unchecked-result behavior. I classify this separately from B1 because §2 does not explicitly specify convergence-failure handling.

## Criteria rows whose evidence is weaker than the claim

These are **B4 evidence gaps**, beyond the production findings above:

| Rows | Gap |
|---|---|
| **3c** | The spy establishes four constructor calls with parameters for Platt/Beta through `fit`. It does not observe parameter use inside each fold’s actual fit, or cover isotonic. |
| **4a–4c** | Tests inspect `resolve_minimize_kwargs` results only. They could pass if a production caller stopped forwarding those results to SciPy. |
| **5b** | Both `s` and `s/2` exceed the rescaling threshold. There is no unscaled reference solve. Converged coefficients also do not establish that the nonzero `x0` was transformed correctly. |
| **6a** | “Wrong types” is represented by `target_smoothing="yes"`; malformed method containers are absent, allowing finding 1. |
| **6c** | Four refusal cases do not cover the whole 6a surface. The tune test observes Booster calls, not study creation, and does not assert the error’s surface text. |
| **8a** | A plain Platt boolean requires no preprocessing. This test cannot establish normalization, isotonic alias canonicalization, or inherited seed in exported configuration. |
| **8b** | Four valid fitter comparisons omit refusal equivalence, tolerance precedence and large-score coordinate transformations. Finding 2 passes these tests. |
| **9a–9b** | Synthetic standalone pickle states test `__setstate__`, not complete historical artifacts through `Model.load()`. My additional nested-joblib probe succeeded, but does not close that artifact-level gap. |

The other named rows provide relevant evidence for their stated cases. I did not independently re-establish row 12’s minimum-dependency result.

## Bounds

- Confirmed HEAD `f67f8c7`; reviewed the production diff against `5ac725e`, H-0100 decisions/rule positions, acceptance criteria and design revision 3, the specified earlier contract sections, BLUEPRINT changes, persistence loading, and mapped tests.
- Individual read-only executions used installed **SciPy 1.17.1**. Confirmed ordinary and large-score Platt agreement with sklearn, all eight Platt methods returning closely matching coefficients, runtime/template method and tolerance-table equality, and actual unknown-option rejection.
- Platt’s objective, analytic gradient, smoothed targets, initialization, slope/bounds coordinate conversion and `a=-A, b=-B` mapping are consistent on inspection.
- Nested legacy joblib state preserved predictions exactly in the probe; a new pickle round trip did too. No missed supported container was identified. Complete historical artifacts and failed-fit legacy states were not verified.
- OOF-only inputs, outer-split reuse and covered-row selection remain unchanged. No calibration-to-estimators imports were found. `tune()` validates calibration settings but does not itself fit a calibrator.
- The requested pytest invocation could not start: uv attempted a cache-lock temporary file on the read-only filesystem. No suite-pass claim, minimum-dependency rerun, or full generated retraining verification is made.
- No files changed and no network calls made.
