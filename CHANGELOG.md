# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and versioning in this repository currently follows a lightweight semantic versioning style.

## [Unreleased]

### Fixed

- Synced package runtime dependencies with the active requirements file so wheel/source installs include tracing and HTML extraction dependencies.
- Made research-reporting tests independent of ignored local `outputs/` artifacts by generating minimal fixtures during test setup.
- Allowed research brief rendering to fall back to indexed summary metadata when referenced artifact JSON files are unavailable.
- Indexed `policy_head2head_cv*/cv_summary.json` cross-fold reports in the research output index.
- Surfaced indexed cross-fold head-to-head reports in the research dashboard run registry.
- Added cross-fold head-to-head summary fields and Markdown evidence to the research brief.
- Exposed cross-fold head-to-head summary fields in the one-shot research refresh manifest headline.
- Added cross-fold headline metrics to the research dashboard JSON, Markdown, and HTML outputs.
- Added CLI smoke coverage for research reporting and cross-fold policy entry points.
- Documented the current benchmark, policy head-to-head, cross-fold, and research reporting workflows.
- Expanded CLI smoke coverage to every public `argparse` script and guarded CI coverage against drift.
- Verified wheel console script metadata in packaging CI and pinned the expected pyproject entry point registry in tests.
- Made configuration subpackages explicit so packaged YAML directories are discovered consistently.
- Updated package license metadata to the SPDX expression format expected by current setuptools builds.
- Replaced remaining legacy source-level project identifiers with the public `Research Orchestrator` identity.
- Extended packaging CI to verify installed wheel license metadata and bundled configuration resources.
- Corrected optional dependency installation examples to use project extras instead of invalid requirements-file extras.

### Added

- Added the MIT license text referenced by project metadata and documentation.

## [0.1.1] - 2026-05-31

### Added

- GitHub Actions for automated test and packaging checks
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, and a pull request template
- Packaging metadata regression tests
- A release checklist at `docs/RELEASE_CHECKLIST.md`

### Changed

- Renamed public project identity from legacy `deep-research-agent` packaging metadata to `research-orchestrator`
- Aligned README, docs entry points, and CLI help text with the `Research Orchestrator` repository identity
- Added CLI smoke checks to CI for core public entry points

### Fixed

- Added missing package markers for `scripts`, `configs`, and `evaluation` to stabilize console entry imports
- Corrected repository links and author metadata in `pyproject.toml`
- Synced `src/__init__.py` version metadata to `0.1.1`
