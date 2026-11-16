Review-kind: review
Review-round: 1

# Review: the implementation of H-0146 (#407) — release-path hardening, workflow lint and the CI gate

Review the implementation commit at this head against Proposal H-0146 (HISTORY.md `## H-0146`, Accepted after `h0146-proposal-review` and `h0146-proposal-confirm`). The base is develop `9402dc3`; the Proposal commits are `28fe70c`, `76805df`, `e7c191a`; the implementation is the last commit (`git diff e7c191a HEAD`). Answer as findings with `file:line` locations and executed or quoted evidence. `APPROVE` when the implementation does what the Proposal says and nothing it forbids; `CHANGES_REQUESTED` otherwise.

## Decided (do not reopen)

Everything in the Proposal's 管理者の判断 and 対応方針, and the choices recorded in its 実装の記録: the pinned versions (setup-uv stays on the v7 line), the `zizmor` marker for Python 3.11+, the `uv.lock` revision change from uv 0.13.1, the separate `Route git credentials through gh` step, and that `test_no_checkout_keeps_credentials` alone guards `persist-credentials: false` where zizmor does not.

## Checks (run what your environment allows)

1. **Release path** (`.github/workflows/release.yml`, `.github/scripts/check_release_ref.sh`, `.github/workflows/auto-release.yml`): the build checks out the run's ref with no persisted credential, refuses any ref other than `refs/tags/<tag>` before installing or building, restores no cache, and the publish jobs hold only `id-token: write`. auto-release dispatches with `--ref "$TAG"`. Is there any path by which a publish runs on a non-tag ref, a mutable action reference, or a cache? Does the `release: published` trigger still pass the check (`GITHUB_REF` of a release event)?
2. **Tag push**: with `persist-credentials: false`, does `create_release_tag.sh`'s `git push origin refs/tags/$TAG` authenticate through the helper `gh auth setup-git` writes, given `GH_TOKEN` in the `Create tag` step's env? Can any other step read the token?
3. **CI gate** (`scripts/ci_gate.py`, the `ci-gate` job, `tests/test_ci_gate.py`): passes only when all seven needed jobs are `success`; fails on any other result, a missing or malformed entry, an unexpected key, or bad JSON. Is the needs list plus the exclusion list exactly the jobs of `ci.yml`? Does `if: always()` make the gate report failure (not skip) when a needed job fails or is cancelled?
4. **ci.yml edits**: `concurrency` cancels only pull-request runs; every job has `timeout-minutes`; every checkout has `persist-credentials: false`; the SC2086 rewrite (`read -ra flags <<< "$out"`, `"${flags[@]}"`) passes the same words as before, still fails the step when `scripts/examples_index.py` fails, and works when it prints nothing. Nothing else in the existing jobs changed (`git diff e7c191a HEAD -- .github/workflows/ci.yml`).
5. **Workflow lint job**: `uv sync --frozen --only-group workflow-lint` installs only the three linters; `zizmor --format github .` and `actionlint` fail the job on any finding; the `GH_TOKEN` it gets is the read-only `github.token`.
6. **artifact-language.yml**: only the pin, `persist-credentials: false`, `timeout-minutes` and the commented inline ignore changed; the workflow still validates the same events.
7. **Dependabot** and **pyproject/uv.lock**: the config matches 対応方針 5; the lock adds the three linters and changes no other package's version (`git diff e7c191a HEAD -- uv.lock`).
8. **Tests and docs**: `tests/test_workflow_policy.py`, `tests/test_release/test_check_release_ref.py`, the updated `tests/test_release/test_auto_release_guard.py`; BLUEPRINT §18.2, CONTRIBUTING.md, `docs/proposal_dispositions.toml`, `docs/audits/2026-11-i407/owner-steps.md` against 対応方針 7–9. Do the owner steps follow the Proposal's order and would any of them stop the next release?

## Environment

The workspace is a digest-fixed, read-only copy of this head. You may run `bash .github/scripts/check_release_ref.sh` and `python3 scripts/ci_gate.py` with chosen environments, and single test files with the workspace's Python; never the full suite. There may be no network and no zizmor/actionlint; the local results are in `docs/audits/2026-11-i407/negative-controls.out` and HISTORY's 実装の記録. Say what could not run.

## Stop criterion

The Proposal's own レビューの停止基準 applies: findings must be defects inside the contract (the publish path using a cache, a mutable reference or a non-tag ref; the gate passing when it should fail or failing when it should pass; a ruleset or environment step that would stop the next release; `persist-credentials: false` breaking the tag push) or false statements in the diff. CI duration, the lane layout, vulnerability alerts, a tag ruleset and the zizmor version are not findings.
