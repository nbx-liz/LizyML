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

You **must** add a Proposal to `HISTORY.md` first. See `skills/history-proposals/SKILL.md` for the format.

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

## Getting Help

- Open an [issue](https://github.com/nbx-liz/LizyML/issues) for bug reports or feature requests
- Check `BLUEPRINT.md` for architectural context
- Check `HISTORY.md` for design decision history
