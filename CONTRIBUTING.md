# Contributing to LizyML

Thank you for your interest in contributing to LizyML!

## Development Setup

```bash
git clone https://github.com/nbx-liz/LizyML.git
cd LizyML
uv sync --frozen --dev
git config core.hooksPath .githooks
```

## Workflow

1. **Branch from develop**: `feat/`, `fix/`, `docs/`
2. **Write tests first** (TDD): RED → GREEN → REFACTOR
3. **Run quality gates** before pushing: `make ci`
4. **Create PR** to `develop` (squash merge on GitHub)
5. **Conventional Commits**: `<type>(<scope>): <description>`

### Release (develop → main)

1. Add a CHANGELOG entry on develop (via a feature PR)
2. `gh pr create --base main --head develop --title "release: vX.Y.Z"`, or
   `uv run python scripts/release.py vX.Y.Z`, which opens the same PR and refuses
   (without committing or pushing) when `CHANGELOG.md` is dirty or local develop
   differs from `origin/develop`
3. Verify CI passes: `gh pr checks <PR#>`
4. Merge with **Create a merge commit** (NOT squash — squash breaks history sync)
5. `auto-release.yml` auto-creates tag + GitHub Release. It first checks that the
   PR's head is `develop` in this repository and that its merge commit has two
   parents, and refuses to tag anything else (H-0117)
6. No post-release sync PR needed

### Commit Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `refactor` | Code restructuring (no behavior change) |
| `docs` | Documentation only |
| `test` | Adding or updating tests |
| `chore` | Maintenance tasks |
| `perf` | Performance improvement |
| `ci` | CI/CD changes |

## Quality Gates

All of these must pass before a PR can be merged:

```bash
make ci
```

This runs:

- `uv run ruff check .` — linting
- `uv run ruff format --check .` — formatting
- `uv run mypy lizyml/` — type checking
- `uv run pytest` — tests (80%+ coverage required)

## Dependencies

### Version-bound policy

Runtime dependencies declare a **lower bound only** (e.g. `pandas>=2.0`) — **no
upper caps**. LizyML already runs the current major releases (numpy 2.x,
pandas 3.x), so a conservative cap such as `pandas<3` would be regressive.
Forward compatibility is protected by CI rather than by pinning:

- **`Quality (latest deps, non-blocking)`** — a CI lane that resolves
  dependencies *upward* (`uv sync --upgrade`) and runs the suite. It is
  **non-blocking**: it surfaces an upstream breaking change (pandas / numpy /
  lightgbm) early without gating merges on a third-party regression.
- **`Quality (lowest-direct deps)`** — pins the declared lower bounds
  (`--resolution lowest-direct`, BLUEPRINT §18.2) so the floor stays honest.
- **`Smoke (<os>)`** — an OS matrix (ubuntu / windows / macos) running the
  file-I/O-sensitive subset to catch path / newline portability issues.

When the latest-deps lane goes red, fix forward (adapt the code) and bump
`uv.lock` — do **not** add an upper cap to silence it. Adding or removing a
runtime dependency still requires a Change-Gate proposal (see below).

## Spec-First Development

LizyML follows a **specification-first** workflow. Before implementing changes to:

- Public API (`Model` methods, Config, FitResult, PredictionResult, Artifacts)
- Split/leakage boundaries
- Export/simulate formats
- Persistence format

You **must** add a Proposal to `HISTORY.md` first.

### Proposal Template

```markdown
## H-XXXX: <Title>

- **ステータス**: Proposed
- **起票日**: YYYY-MM-DD
- **関連**: H-YYYY (if applicable)

### 目的
Why is this change needed?

### 変更内容
What will change? List affected files and behaviors.

### 影響範囲
Which modules, configs, or result shapes are affected?

### 互換性
Is this backward compatible? Does format_version need a bump?

### 代替案
What alternatives were considered and why rejected?

### 受け入れ基準（テスト観点）
What tests prove the change is correct?
```

The proposal must be **accepted** (reviewed) before implementation begins. Commit order: `docs(history): add proposal` → `feat/fix: implement` → `test: add tests`.

### Documentation Priority

When specifications conflict, priority is:

1. `BLUEPRINT.md` (structure, contracts, invariants)
2. `HISTORY.md` (proposals and decisions)
3. `CONTRIBUTING.md` (this file: workflow, release and review rules)
4. Implementation code

`PLAN.md` is a roadmap and status document, not a specification. Agent
instruction files (`CLAUDE.md`, `.claude/AGENTS.md`, `.claude/skills/`) are local
and not tracked in this public repository: they summarise the rules above for an
agent session and never override them (H-0117).

## Testing Requirements

- **Minimum 80% coverage** for all new code
- **Contract tests** for public API / Config / Result shape changes
- **Leak detection tests** for split / calibration changes (must include "should-fail" cases)
- **Reproducibility tests** with seed pinning for new features
- **A test named for an effect asserts it where it happens** (#270). If the name or
  docstring claims that training changes, a parameter reaches the Booster, a public
  `Model` method behaves a certain way, or an export loads back, observe that: the
  params `lgb.train` received (`tests/_train_spy.py`), the trained Booster, or what
  `Model.fit` / `predict` / `load` returned. A test of a helper is named for the helper
  and points at the boundary test in its docstring. A refusal test checks which gate
  refused when its input would trip another one too, and no assertion sits inside an
  `if` that skips it when the thing claimed is missing. To check a test, break the code
  it claims to cover and see it fail;
  `docs/audits/2026-09-defect-discovery/instruments/kill_producers.py` does that for
  `lgb.train`, `Model`'s public methods, metrics and splitters (`LIZYML_KILL=...`).

## Language Convention

- `BLUEPRINT.md`, `HISTORY.md`, `PLAN.md`, and the local agent instruction files: Japanese
- Code, docstrings, commit messages, PR descriptions: English

## Running Tests

```bash
uv run pytest                          # full suite
uv run pytest tests/test_metrics/      # single directory
uv run pytest -k "test_ece"            # by keyword
uv run pytest --cov=lizyml -q          # with coverage
```

## Adding a New Metric

1. Create a class inheriting `BaseMetric` in `lizyml/metrics/regression.py` or `classification.py`
2. Decorate with `@MetricRegistry.register("metric_name")`
3. Implement `__call__(self, y_true, y_pred) -> float`, `needs_proba`, `greater_is_better`
4. Add to the task whitelist in `lizyml/estimators/lgbm/metric_bridge.py` (if applicable as feval)
5. Add codegen implementation in `lizyml/codegen/templates.py` (if feval)
6. Add tests: correctness, boundary values, and `get_metric("name")` registry lookup
7. Update `docs/config-reference.md` metric table

## Adding a New Estimator

See [docs/add-estimator-guide.md](docs/add-estimator-guide.md) for the full checklist.

## Getting Help

- Open an [issue](https://github.com/nbx-liz/LizyML/issues) for bug reports or feature requests
- Check [docs/api.md](docs/api.md) for public API reference
- Check [docs/faq.md](docs/faq.md) for common questions
- Check `BLUEPRINT.md` for architectural context
- Check `HISTORY.md` for design decision history
