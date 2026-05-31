# Contributing to Research Orchestrator

Thanks for contributing. Keep changes small, testable, and easy to review.

## Development Setup

1. Use Python `3.10+`, with `3.11` recommended.
2. Install dependencies:

```bash
pip install -r requirements.txt
pip install pytest pytest-asyncio
```

3. Copy environment templates when needed:

```bash
cp .env.template .env
cp .env.tools.template .env.tools
```

## Recommended Workflow

1. Create a feature branch from `main`.
2. Make focused changes with clear commit boundaries.
3. Run local checks before opening a pull request:

```bash
pytest -q
python scripts/run_single.py --help
python scripts/run_eval.py --help
python scripts/build_search_cache.py --help
```

4. Update docs when behavior, configuration, or public interfaces change.
5. Open a pull request with a concise summary, validation notes, and any known risks.

## Coding Expectations

- Prefer small, composable functions over large monolithic changes.
- Preserve backward compatibility for configs and scripts where practical.
- Avoid committing secrets, API keys, large datasets, local outputs, or model checkpoints.
- Keep mock mode and smoke-test paths usable for contributors without paid APIs.

## Tests

- Add or update tests for user-visible behavior changes.
- Favor deterministic tests that do not require live external services.
- If a change depends on live APIs or private infrastructure, document the verification gap in the pull request.

## Issues and Pull Requests

- Use GitHub issues for bugs, regressions, and enhancement proposals.
- Include reproduction steps, relevant config, and exact error output for bugs.
- For research-facing changes, note expected effects on evidence quality, cost, latency, or benchmark behavior.
