# Contributing

## Branch and merge workflow

- Keep `main` releasable. Do not commit feature work directly to `main`.
- Create a short-lived branch for each focused change, using `feature/`, `fix/`, `docs/`, or `ci/` followed by a short kebab-case name.
- Open a pull request into `main`, describe the change and verification, and wait for CI to pass before merging.
- Use squash merges for a readable history. Delete the feature branch after merge.
- Keep pull requests small enough to review and include documentation when behavior or setup changes.

## Local checks

Install the project dependencies with Poetry, then run the same static checks used in GitHub Actions:

```bash
poetry run ruff check --extend-select W291,W293 src scripts tests
poetry run python -m compileall -q src scripts tests
poetry run python -m unittest discover -s tests -v
poetry run python scripts/test_telegram_logic.py
```

The CI workflow runs these checks on pull requests to `main`, pushes to `main`, and manual dispatches. Changes that need external services, model downloads, or Telegram credentials should also include a separate manual end-to-end verification record; secrets must be stored in GitHub Actions secrets or a local `.env` file and never committed.

## Pull request checklist

- Explain the purpose and user-visible effect.
- List checks run and their outcomes.
- Call out required secrets or operational changes without including secret values.
- Confirm that generated data, model weights, and credentials are not part of the change.
