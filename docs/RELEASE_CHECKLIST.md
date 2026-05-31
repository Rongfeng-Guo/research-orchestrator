# Release Checklist

Use this checklist before tagging or announcing a new repository release.

## Versioning

- Update version metadata in `pyproject.toml`
- Update version metadata in top-level `__init__.py` if needed
- Update version metadata in `src/__init__.py`
- Add a new dated entry to `CHANGELOG.md`

## Validation

- Run `pytest -q`
- Run `python scripts/run_single.py --help`
- Run `python scripts/run_eval.py --help`
- Run `python scripts/build_search_cache.py --help`
- Optionally run `python scripts/run_repl.py --help`
- Optionally run `python scripts/run_benchmark.py --help`

## Packaging

- Confirm `python -m build` succeeds in a clean release environment
- Confirm public repository URLs and author metadata are correct in `pyproject.toml`
- Confirm `.gitignore` excludes local outputs, secrets, datasets, and checkpoints

## Documentation

- Review `README.md` quick start and badges
- Review environment templates and config docs for any changed keys
- Document known limitations or release caveats when behavior changed

## Release Hygiene

- Ensure GitHub Actions are green on the release commit
- Verify the target branch and remote belong to `Rongfeng-Guo/research-orchestrator`
- Create a tag only after validation artifacts and notes are ready
