# Dev Tooling Notes (LISAI)

This is a short roadmap for dev tooling adoption while the refactor is still in progress.

## Current minimal setup (recommended now)

`requirements-dev.txt`

```txt
pytest==8.3.3
ruff==0.6.9
```

Why now:

- `pytest`: catch regressions while fixing refactor bugs.
- `ruff`: fast, high-signal linting (imports/errors) with low setup friction.

## Next step (after the current bug-fix phase stabilizes)

Consider adding:

- `pre-commit==3.8.0`
- optional `black==24.8.0`

Why:

- `pre-commit` runs checks automatically before commits.
- `black` enforces consistent formatting if/when you want style standardization.

## Later step (once architecture/contracts are more stable)

Consider adding:

- `pytest-cov==5.0.0`
- `mypy==1.11.2`

Why:

- `pytest-cov`: track test coverage in CI.
- `mypy`: static typing checks (best added incrementally after refactor churn is lower).

## Suggested progression

1. Start: `pytest` + `ruff` only.
2. Then: add `pre-commit` (and optionally `black`).
3. Finally: add `pytest-cov` and `mypy` in a targeted way.

## Reminder

Runtime dependencies should stay managed by Conda env files:

- `environment.cpu.yml`
- `environment.cuda.yml`

Use dev requirements for tooling only.
