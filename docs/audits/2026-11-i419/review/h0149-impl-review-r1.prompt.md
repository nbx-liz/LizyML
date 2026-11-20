Review-kind: review
Review-round: 1

# Review: implementation of H-0149 (#419, PR #424) — explicit TPESampler options and per-band seeded sequences

Review the implementation at this head against the approved Proposal H-0149 (`HISTORY.md`, section `## H-0149`, 対応方針 1–6 and 受け入れ基準 1–7). Answer as findings with `file:line` locations and executed or quoted evidence. `APPROVE` when the implementation meets the Proposal and has no correctness defect; `CHANGES_REQUESTED` otherwise.

## Disclosure

The implementation commit (`00440c6`) was made by a Codex maker (gpt-6.1-sol, high). This review is same-provider. Judge the code and the contract only.

## Changed paths (base `23a1f14`, the approved Proposal head)

`lizyml/tuning/tuner.py`, `lizyml/core/_tuning_resume.py`, `tests/test_tuning/test_h0149_sampler_sequence.py`, `tests/test_tuning/data/h0149_tpe_sequences.json`, `docs/audits/2026-11-i419/capture_sequences.py`, `.github/workflows/ci.yml`, `BLUEPRINT.md`, `CHANGELOG.md`, `docs/proposal_dispositions.toml`, and `HISTORY.md` (accepted status and the implementation record, written by the main context). Use `git diff 23a1f14 HEAD`.

## Neutral evidence (run by the main context on machine V; you may not be able to rerun it)

- Full suite with slow tests on `00440c6` (`-m ""`, Python 3.12.3, Optuna 5.0.0): 24084 passed, 264 skipped, 1 failed. The failure is `tests/test_e2e/test_version.py::TestVersion::test_version_matches_package_metadata`: `_version.py` is `0.17.2.dev96+g23a1f14a2.d20261120` and the metadata is `0.17.2.dev96+g23a1f14a2`, because the per-version editable installs regenerated `_version.py` on a dirty tree. It is unrelated to the diff, and CI is authoritative.
- `tests/test_tuning tests/test_docs` went from 2419 passed to 2445 passed, the difference being the 26 new tests. `tests/test_docs` passes after the HISTORY record (809 passed).
- Capture across Optuna 3.1.0–5.0.0 (16 environments): bands 3.1–3.2, 3.3–3.4, 3.5–4.7, 5.0. Two runs gave identical bytes (sha256 `ad8fd158…`).
- **Known open item:** PyPI has Optuna 5.1.0, so the latest-deps lane will resolve it. Measured by the main context with LizyML's `default_space("binary")`, the 5.1.0 sequence equals 5.0.0's. The golden has no `5.1` minor, so `select_band` refuses 5.1.0 and the latest-deps lane fails. A repair is planned that adds 5.1.0 to the capture and moves the unknown-minor test case from `5.1.0` to an unreleased minor. Report anything else this implies.

## Checks

1. **Wiring (受け入れ 1, DC4).** Do all three sites go through `make_tpe_sampler`? Does any other path in `lizyml/` construct a sampler or a study that samples? Does the recording-subclass test actually observe the production constructor, given that `make_tpe_sampler` looks `TPESampler` up at call time?
2. **Warning filter (受け入れ 2).** Is the `ExperimentalWarning` filter scoped to construction, and does it leave other categories visible, including a warning of another category raised during construction? Is `optuna.exceptions.ExperimentalWarning` available in every admitted version?
3. **Resume sentence (受け入れ 5).** Does the rule (the leading integer of each version string; no sentence when either side is unreadable or absent) match 対応方針 3? Is the old message byte-identical otherwise? Do the tests derive the recorded version from the installed one, so that they hold in every lane, including lowest-direct (Optuna 3.1.0)?
4. **Golden (受け入れ 3, 4; DC1, DC3, DC7).** Does the test observe the production sampler (`make_tpe_sampler`) and production `suggest_params` / `default_space`, and not a copy? Is the comparison exact after 12-significant-digit rounding, and are types compared? Can the test pass silently in some lane (a skip, an empty band, a vacuous comparison)? Does `select_band` fail closed for every version outside the listed minors, and is its matching anchored (DC2)? Is the JSON writer deterministic, and does the data's provenance match what the capture recorded? Is importing the capture script by path from a test robust in CI (working directory, `docs/` present in the checkout, no side effects at import)?
5. **CI lanes.** Are both smoke steps changed and nothing else in the workflow? Is any pytest lane in `.github/workflows/ci.yml` left without the test (per 対応方針 4, Extras registry and Notebook index are out of scope)?
6. **Docs.** Are BLUEPRINT §11.5.3, §18.1.1 and §18.1.6, CHANGELOG and the disposition anchors consistent with the code and with the Proposal? Is the HISTORY implementation record accurate?
7. **Defect classes.** Check the diff against DC1–DC8 (`~/.claude/rules/defect-classes.md` is not in your workspace; the classes are: silent pass, loose boundary matching, SSOT/derived drift, inert wiring, stale acceptance, dead-in-practice, unsatisfiable declaration, stale writer).

## Environment

The workspace is a digest-fixed, read-only copy of this head. Run single test files with the workspace's Python if it has the dependencies; never the full suite. State what could not run.

## Stop criterion

The Proposal's own レビューの停止基準 applies: findings are contract defects or false claims. The Optuna range, refusing resume, parallel trials and LizyStudio are out of scope.
