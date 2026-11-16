# H-0146 — settings the maintainer applies in the GitHub UI

Nothing here is written through the API (owner ruling, #407). Apply each block at
the moment its heading names; each block ends with how to undo it.

The required check names are copied from the implementation PR's real check list
(`gh pr checks <PR>`), not from the YAML: `CI (gate)` and `validate`.

## 1. "Protect develop" — after the implementation PR is merged into develop

Settings → Rules → Rulesets → Protect develop:

- Enforcement status: **Active**
- Bypass list: empty
- Target: `refs/heads/develop` (already set)
- Rules:
  - Restrict deletions: on (already)
  - Block force pushes: on (already)
  - Require a pull request before merging: on
    - Required approvals: 0; dismiss stale approvals, code owner review, approval
      of the most recent push, conversation resolution: off
    - Allowed merge methods: **Squash** only
  - Require status checks to pass: on
    - Require branches to be up to date before merging: off
    - Checks: `CI (gate)` and `validate`, source GitHub Actions

Pull requests opened before the merge show no `CI (gate)` until they take in
develop (merge develop into the branch, or push again after rebasing).

Undo: set Enforcement status back to Disabled.

## 2. `testpypi` and `pypi` environments — right before merging the release PR that carries H-0146

Settings → Environments → `testpypi`, then the same for `pypi`:

- Deployment branches and tags: **Selected branches and tags**
- Add rule: type **Tag**, pattern `v*`
- No branch rule.

Apply it at the release confirmation, before merging `release: vX.Y.Z`: that
release's auto-release run (from the merge commit, which carries H-0146)
dispatches `release.yml --ref vX.Y.Z`, so its publish jobs run on the tag.

If a publish job is refused by the environment: nothing has been uploaded (the
refusal comes before TestPyPI). Set the environment back to "No restriction",
re-run with `gh workflow run release.yml --ref vX.Y.Z -f tag=vX.Y.Z`, and report it.

## 3. "Protect main" — after that release PR is merged into main

Settings → Rules → Rulesets → Protect main:

- Enforcement status: **Active**
- Bypass list: empty
- Rules: keep deletion, force-push and pull-request rules; in the pull-request
  rule set Allowed merge methods to **Merge** only (approvals 0;
  `require_extra_approval_for_unattributed_changes` may stay on — it only adds an
  approval for Copilot-opened PRs when a non-zero count is required)
- Require status checks to pass: on, up to date: off, checks `CI (gate)` and
  `validate`, source GitHub Actions

Before this point, pull requests into main whose merge ref lacks H-0146 (for
example Dependabot PRs opened before the release) show no `CI (gate)`.

Undo: set Enforcement status back to Disabled.

## Evidence to record on #407 afterwards

- the first PR after step 1: `mergeStateStatus` is `BLOCKED` until `CI (gate)` passes;
- the release: tag push succeeded, `Build distribution` log has no `Cache restored`,
  the `release.yml` run's ref is the tag, both publish jobs passed the environment;
- after the release: Dependabot reports no config error and its first uv PR targets
  develop and changes only `uv.lock`.
