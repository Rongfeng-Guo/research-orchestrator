# Release Checklist

Use this checklist before tagging or announcing a new repository release.

## Versioning

- Update version metadata in `pyproject.toml`
- Update version metadata in top-level `__init__.py` if needed
- Update version metadata in `src/__init__.py`
- Add a new dated entry to `CHANGELOG.md`

## Validation

- Run `pytest -q`
- Run `python -m compileall src scripts evaluation`
- Run `pytest tests/test_cli_help_clean.py -q`
- Confirm the CLI smoke list in `.github/workflows/python-ci.yml` covers all public `argparse` entry points in `scripts/`

## Packaging

- Confirm `python -m build` succeeds in a clean release environment
- Confirm `python -m pip wheel --no-deps . --wheel-dir <tmp_wheel_dir>` succeeds
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
