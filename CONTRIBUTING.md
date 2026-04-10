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
2. `gh pr create --base main --head develop --title "release: vX.Y.Z"`
3. Verify CI passes: `gh pr checks <PR#>`
4. Merge with **Create a merge commit** (NOT squash — squash breaks history sync)
5. `auto-release.yml` auto-creates tag + GitHub Release
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
3. `AGENTS.md` (operational principles)
4. `skills/*` (implementation procedures)
5. Implementation code

## Testing Requirements

- **Minimum 80% coverage** for all new code
- **Contract tests** for public API / Config / Result shape changes
- **Leak detection tests** for split / calibration changes (must include "should-fail" cases)
- **Reproducibility tests** with seed pinning for new features

## Language Convention

- `BLUEPRINT.md`, `HISTORY.md`, `PLAN.md`, `CLAUDE.md`: Japanese
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
