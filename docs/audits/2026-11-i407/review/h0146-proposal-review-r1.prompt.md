Review-kind: review
Review-round: 1

# Review: Proposal H-0146 (HISTORY.md `## H-0146`) — harden the release path and give the branch rulesets a required check (#407)

Review the Proposal text at this head, not an implementation (there is none yet). Answer as findings with `file:line` locations and executed or quoted evidence. `APPROVE` when the Proposal is correct, complete and implementable as written; `CHANGES_REQUESTED` otherwise. Also check the facts stated in issue #407 (its body is restated in the Proposal's 実測 and 管理者の判断) against the evidence under `docs/audits/2026-11-i407/` and the workflow files at this head.

## Decided (do not reopen)

The owner's rulings in the Proposal's 管理者の判断 (2026-11-16): CI duration stays as H-0143 set it; both rulesets are enabled with the CI aggregate job and `artifact-language` (`validate`) as required checks; zizmor and actionlint are blocking inside the aggregate; `main` merge commit only, `develop` squash only; rulesets and environments are set by the owner in the UI, never written through the API; environments allow `v*` tags only; SHA pins in all four workflows; Dependabot `uv`, 7-day cooldown, `target-branch: develop`.

## Checks (run what your environment allows)

1. **Release on the tag ref.** With 対応方針 1, 2 and 8 applied as written, does the next release reach PyPI? Trace: `auto-release.yml` (runs from the `main` merge commit) dispatches `release.yml --ref "$TAG"`; `release.yml` at the tag checks the ref and builds; both publish jobs run under an environment allowing only `v*` tags. Name any step that would still run on `main`, refuse, or need the old `ref:` input. Is the `release: published` trigger still consistent with the ref check?
2. **Tag push without persisted credentials.** 対応方針 2: `persist-credentials: false` plus `GH_TOKEN` and `gh auth setup-git` only in the tag step, keeping `.github/scripts/create_release_tag.sh` unchanged. Does `git push origin refs/tags/$TAG` authenticate there? Does any later step in that job read the credential? (Locally measured in a throwaway HOME: `gh auth setup-git` with only `GH_TOKEN` set exits 0 and writes `helper = !/usr/bin/gh auth git-credential`; `git credential fill` for `host=github.com` then returned `username=x-access-token` and the `GH_TOKEN` value.) Also check the ruleset values in 対応方針 7, including `require_extra_approval_for_unattributed_changes`, against GitHub's REST documentation for repository rules.
3. **The aggregate job.** 対応方針 3: is the `needs` list plus the closed exclusion list exactly the jobs of `.github/workflows/ci.yml` at this head? Does `if: always()` with `toJSON(needs)` judge `cancelled`/`skipped` correctly? Compare `scripts/notebook_index_gate.py`. Is excluding the `continue-on-error` latest-deps lane right?
4. **Ruleset and environment ordering** (対応方針 7, 8): could any step of the order (develop ruleset after this merges to develop; main ruleset after the release PR merges; environments right before that release PR merges) block a PR or a release? Consider Dependabot PRs still targeting `main` until the release, and the release PR itself being a PR into `main`.
5. **Dependabot** (対応方針 5): are `package-ecosystem: uv`, `versioning-strategy: lockfile-only`, `cooldown.default-days` and `target-branch` valid together, and is "the config is read from the default branch" correct?
6. **zizmor and actionlint** (対応方針 3, 4): is the evidence (`zizmor-develop-9402dc3.txt`, `actionlint-develop-9402dc3.txt`) consistent with the 実測 counts? Does the plan clear every finding class, and is the single inline ignore the only exemption? Is online zizmor in a required check a sound choice?
7. **Change Gate fields**: the two `Firing rate:` lines follow the form and match `concurrency-overlap.out` and the zizmor output; `Gate-category: none` is admissible.
8. **Acceptance criteria**: is each criterion testable, and does criterion 6 (runtime evidence, `Refs #407`) cover what the repository cannot test before a release?

## Environment

The workspace is a digest-fixed, read-only copy of this head. There may be no network and no zizmor/actionlint binaries; read the stored outputs instead and say what could not run. Never run the test suite.

## Stop criterion

The Proposal's own レビューの停止基準 applies: findings must be defects inside the contract (the publish path still using a cache, a mutable reference or a non-tag ref; the aggregate passing when it should fail or failing when it should pass; a ruleset or environment step that would stop the next release; `persist-credentials: false` breaking the tag push) or false factual claims. CI duration, the lane layout, vulnerability alerts, a tag ruleset and the choice of zizmor version are not findings.
