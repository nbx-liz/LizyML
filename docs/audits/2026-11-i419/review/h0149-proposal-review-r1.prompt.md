Review-kind: review
Review-round: 1

# Review: Proposal H-0149 (HISTORY.md `## H-0149`) — pass TPESampler options explicitly and state tune reproducibility as "under the same Optuna version" (#419)

Review the Proposal text at this head, not an implementation (there is none yet). Answer as findings with `file:line` locations and executed or quoted evidence. `APPROVE` when the Proposal is correct, complete and implementable as written; `CHANGES_REQUESTED` otherwise. Also check the facts stated in issue #419, in the owner-ruling comment on #419, and in the Proposal's 実測 against the code at this head and `docs/audits/2026-11-i419/`.

## Decided (do not reopen)

The owner's rulings in 管理者の判断 (2026-11-20): keep `optuna>=3.1`; pass `multivariate=True` and `constant_liar=True` explicitly; the contract is "same Config and seed under the same Optuna version"; resuming under another Optuna major continues with a specific warning (not a refusal), keyed on the major; a golden trial sequence per Optuna version band, run in every CI lane without skip, failing outside the known bands; the implementation is made by a Codex maker if the Proposal is `Gate-category: none`.

## Checks (run what your environment allows)

1. **Measurements.** Do the probes in `docs/audits/2026-11-i419/probes/` measure what the README and 実測 say, and do the `.out` files support each stated claim (the four bands with the explicit options, the three bands with the defaults, `constant_liar` inert sequentially, the two `ExperimentalWarning`s below 5.0.0, `load_study` writing nothing, the widened dimension leaving the relative space)? Is the claim that no constructor argument of Optuna 5.0.0 restores the 4.x sequence supported by the Optuna 5.0.0 source (`optuna/samplers/_tpe/parzen_estimator.py`, `sampler.py`), or is there an argument the probes did not try?
2. **The three construction sites.** Are `load_stored_study`, `create_study` and `Tuner.run` (`lizyml/tuning/tuner.py`) the only places in `lizyml/` that construct a sampler or a study, including resume (`lizyml/core/_tuning_resume.py`, `lizyml/core/_model_tuning.py`) and `restore_tuning` / `read_study_lineage`? If another path creates or loads a study with a sampler that later samples, the Proposal must name it.
3. **Warning suppression** (対応方針 2): is ignoring `optuna.exceptions.ExperimentalWarning` only during construction sufficient and narrow (no other warning category swallowed; nothing emitted later at sampling time in any measured version)? Is `optuna.exceptions.ExperimentalWarning` importable under every admitted version?
4. **Resume sentence** (対応方針 3): is the place in `_tuning_resume.py` the one every resume path passes through? Is "major = leading integer of the version string; unreadable → no sentence" well defined for the version strings `_tuning_identity.library_versions` records? Do the 互換性 statements about resuming a 4.x study under 4.x after the change hold?
5. **Golden test** (対応方針 4, 受け入れ 3): is the sequence construction (`make_tpe_sampler`, `suggest_params`, `default_space("binary")`, a non-training objective, 12 significant digits for floats) implementable from the cited functions at this head, deterministic across the CI lanes (`.github/workflows/ci.yml`: Python 3.10–3.12, lowest-direct, latest-deps, and whether the smoke lanes on macOS/Windows run `tests/test_tuning`), and does it observe the sampler the production code builds rather than a copy? Does the lowest-direct lane resolve an Optuna in a measured band? Is "fail on an unlisted minor" consistent with how the latest-deps lane will behave on the next Optuna minor, and is that behaviour stated?
6. **Change Gate fields**: is `Gate-category: none` admissible (no state machine, concurrency/ownership or shared-state invariant change)? Does the `Firing rate:` line follow the required grammar and match the `.out` files?
7. **Acceptance criteria**: is each testable at this head, does the 落ちるべき例 actually fail, and does criterion 4 execute the real Optuna versions rather than a fixture (DC7)? Are any DC1 (silent pass) or DC4 (unreachable wiring) gaps left in the test design?

## Environment

The workspace is a digest-fixed, read-only copy of this head. Run single test files or the probes with the workspace's Python if it has the dependencies (the locked Optuna is 5.0.0); never the full suite. Say what could not run.

## Stop criterion

The Proposal's own レビューの停止基準 applies: findings must be contract defects (a construction site not going through `make_tpe_sampler`, an `ExperimentalWarning` escaping, another warning swallowed, the sequence test skipping or passing silently in some lane or outside the bands, the resume sentence disagreeing with the major difference) or false factual claims. The Optuna range, refusing resume, parallel trials and LizyStudio are not findings.
