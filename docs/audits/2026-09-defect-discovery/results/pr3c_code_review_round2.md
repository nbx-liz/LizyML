# PR 3c code review, round 2 — scoped to the round-1 fixes (2026-09-15, head `761c05e`)

Prompt: `prompt-templates/pr3c-code-review-round2-scoped.md` (`{{HEAD}}` = `761c05e`).

**Carrier:** a fresh-context, read-only Claude subagent (general-purpose). The Codex run was killed three times in a row by the host's memory monitor. Each kill came during light commands (`sed`, `rg`), with about 61 GB free afterwards, so the carrier was changed instead of retrying a fourth time. The subagent received the prompt file only, not the maker's rationale.

## VERDICT: REQUEST_CHANGES

| # | Question | Severity | Finding | Reproduced by main context |
|---|---|---|---|---|
| A | Q2 | blocking | The round-1 fix `_run_minimize` (`lizyml/codegen/templates.py`) turns **every** `OptimizeWarning` into a refusal. Bounded Powell and Nelder-Mead also warn "Initial guess is not within the specified bounds". The runtime accepts and fits these settings, so an unedited export stops the generated `train.py`, and the message names the wrong setting (`'options'`). No written `x0` is needed: the default start is enough. Example: beta `{"method": "Powell", "bounds": [[0, 0.2], [null, null], [null, null]]}`. | **Yes**: runtime `a=0.19999`, generated `ValueError ... 'options' Initial guess is not within the specified bounds` |
| B | Q2 | note | Not in the fix diff. `CFG.get("calibration_params") or {}` in the generated `fit_calibrator` turns an edited `[]`, `0`, `""`, `false` or `null` into `{}`, which runs the defaults. The runtime refuses these values at config validation. The generated side was inferred from the code, not run. | No (inferred) |
| C | Q1 | should-change | Fix 1 closes list/dict `method`. An integer too large for a float (`10**400`) in `tol`, `x0` or `bounds` still escapes as `OverflowError` from `math.isfinite` / `math.isnan`, not `CONFIG_INVALID`. This holds through both `Model.fit()` and `Model.tune()` (12 combinations). | **Yes**: `validate_optimizer_params({"tol": 10**400}, ...)` raises `OverflowError` |

**Out of scope, as the checker reported it:** runtime `PlattCalibrator.fit` also silently accepts the "Initial guess is not within the specified bounds" warning and fits from the clipped start. Related to #297.

**Q3 unaffected:**
- Valid settings (L-BFGS-B, TNC, trust-constr, BFGS, Nelder-Mead) agree between runtime and generated, with no warnings.
- Other warnings are re-emitted.
- The 98 tests in the three touched files pass.

**Not verified by the checker:**
- scipy 1.10 behaviour of `_run_minimize`. Row 12 predates the fix commits.
- A full generated `train.py` run.
- The isotonic path.
- The full suite.
