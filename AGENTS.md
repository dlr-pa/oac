# AGENTS.md

Orientation for coding agents working in this repo. For contribution process,
governance, and commit conventions, see `CONTRIBUTING.md`.

## What this is

OpenAirClim (`openairclim/`) is a climate response model for air traffic
emissions. `core/` is the simulation engine; `gui/` is an optional Panel-based
GUI on top of it; `addon/` integrates optional premium functionality
(feature-detected, may not be installed).

## Environment / running tests

```bash
pixi install -e dev     # or --all for both the default and dev environments
pixi run -e dev test    # generates test fixture data first, then runs pytest
```

Pixi (`[tool.pixi.*]` in `pyproject.toml`) is the primary dev tool, with two
environments: `default` (runtime, GUI included) and `dev`.
`environment.yaml`/`environment_dev.yaml` are generated from them via
`pixi run export-envs` (`scripts/export-envs.sh`) — don't hand-edit them;
`check-env-exports.yml` fails CI if they drift out of sync.
`pip install -e . --group dev` (pip>=25.1) also works, without pixi's task
runner.

The GUI dependencies (panel, param) are core dependencies, not an extra, so
every PyPI/conda-forge install includes the GUI. There are no
`[project.optional-dependencies]`: development-only dependencies are PEP 735
`[dependency-groups]` (`test`, `docs`, and `dev`, which includes the other
two), which aren't published to PyPI. pixi maps the `dev` group to its `dev`
feature; the conda-forge equivalents in `[tool.pixi.feature.dev.dependencies]`
take precedence, so a new dependency usually needs adding in both places.

There is no Python pin for development: pixi's Python comes from
`requires-python`, and `pixi.lock` fixes the exact version (bumped by
`pixi-update.yml`). Note that `pixi update` can keep the currently locked
Python on the platform you run it on even when a newer one would solve.

`pip-install-test.yml`/`conda-install-test.yml`/`pixi-install-test.yml` are
the full OS × Python (or, for pixi, OS-only — the lock pins one Python)
compatibility matrices - they test packaging, not code correctness, so they
only run on push to `main`/`dev`, weekly, and on `workflow_dispatch`, not on
every PR push. `quick-test.yml` covers PR pushes instead, with a single
pixi/ubuntu job (`pixi run -e dev test`) for fast feedback. `lint.yml` also
runs in the pixi `dev` environment, calling the `corr-ruff`/`corr-mypy` tasks
with the PR's changed files (both take an optional `files` argument).
`build-docs.yml` deploys on a published GitHub Release, or on
`workflow_dispatch` from `main`/a tag (other branches only build).

Test files are named `*_test.py` (not `test_*.py`) and use class-based
`TestXxx` / `test_yyy` grouping, one class per function under test.

## Layout

- `openairclim/core/` — simulation engine. `config_model.py` is the single
  source of truth for the TOML config schema (pydantic `Config` model) and
  does *structural* validation only (no filesystem access). `read_config.py`
  layers filesystem/cross-reference checks on top (file existence, aircraft
  csv merging) — see its module docstring for the full validation pipeline.
- `openairclim/gui/` — Panel app. `config_io.py` holds GUI-facing config
  logic with no Panel dependency (parse/validate/save, plus the per-card
  required-field checks used by both the Config tab and the sidebar's
  Validate button). `tabs/*.py` build the actual widgets. `gui/` may import
  from `core/`, never the other way round.
- `openairclim/addon/_premium.py` — feature-detects `openairclim_premium`;
  check `OAC_PREMIUM_AVAILABLE` before assuming premium symbols exist.
- `tests/core/`, `tests/gui/` — mirror the source tree 1:1
  (`tests/gui/tabs/scenario_test.py` tests `openairclim/gui/tabs/scenario.py`).
  `tests/conftest.py` holds a shared `valid_config`/`working_dir` fixture
  pair, backed by real fixture files in `tests/core/repository/` — reuse it
  rather than hand-rolling another valid config dict.
- `openairclim/repository.py` — resolves/downloads the response-surface and
  background-concentration data ("repository data"), published separately in
  `dlr-pa/oac-repository`. Explicit-trigger only (`oac-download-data`); never
  called implicitly from library code.
- `openairclim/utils/` — packaged CLI utilities (`oac-download-zenodo`,
  `oac-create-*`). `create_test_data.py`/`create_test_files.py` are dev-only
  fixture generators, not console-scripts.
- `docs/source/api_ref/{core,gui,utils}/*.rst` — one stub file per module,
  `automodule::`-based, plus a top-level `oac.repository.rst`. `api_ref.rst`
  globs each folder automatically, but a new module still needs its own stub
  file or it won't appear.

## Linting & docstrings

Ruff + mypy are the linters. `openairclim/gui/` and `tests/gui/` are excluded
via `pyproject.toml` (`extend-exclude` / `exclude`) — not yet clean — but that
only applies to whole-repo/directory runs (`ruff check .`, `mypy .`). Both
tools bypass `exclude`/`extend-exclude` by default when given explicit file
arguments (including `lint.yml`'s CI, which only lints PR-changed files), so
any hand-built file list must filter `gui/` paths out itself — `ruff` accepts
`--force-exclude` to opt back in to config-based excludes even with explicit
paths; `mypy` has no equivalent, filter before invoking it.

Docstrings are Google-style (napoleon). Things that are easy to get wrong,
none of which `ruff`/`mypy`/`pytest` will catch — verify with an actual docs
build (or by rendering through `sphinx.ext.napoleon.GoogleDocstring`
directly):

- Inline code/literals use double backticks (`` ``like_this`` ``). Single
  backticks are reserved for Sphinx roles.
- Cross-references use `:func:`, `:class:`, `:mod:`, `:data:` roles (e.g.
  `` :func:`~pkg.mod.func` ``) in prose, including for external types
  mentioned in prose. In the type-slot of an `Args`/`Returns` entry itself
  (the part before the colon), use the plain type name instead.
- `Returns:` continuation-line indentation is *not* the same as `Args:`.
  `Args:` and `Raises:` are definition lists (`name (type):` / `ExceptionType:`
  start each entry), so their continuation lines must hang-indent one level
  deeper than that starting line, or Napoleon parses the next line as a new
  sibling entry and mangles it. `Returns:` is the odd one out — a single
  unnamed field — so its continuation lines must instead align flush with
  its own first line. Failure modes here are visually subtle and only show up
  in rendered output.
- `main()` functions (CLI entry points) don't get a `-> None` return
  annotation; every other zero-return function does.

## Conventions worth knowing

- `openairclim/core/*` changes get reviewed closely by the maintainer. Always
  confirm there before making them.
- Optional TOML `dir` fields (`Path = Path("")` in `config_model.py`)
  normalise to `Path(".")`, not `""`, once validated — check `== Path(".")`,
  not truthiness, when detecting "unset".
- In `gui/tabs/*.py`, when a refresh function sets a widget's `.value` to a
  resolved/suggested value, also write that value into the config dict
  directly — don't rely on the widget's change-event to persist it. It won't
  fire if the value happens to already match, silently desyncing the dict
  from what's displayed.
- `docs/source/demos/*` pages are reStructuredText files with
  `jupyter-execute` blocks (`jupyter_sphinx`), executed during
  `sphinx-build` and not covered by `pytest`. Run the build from the repo
  root (`pixi run docs-build`), not `docs/`: each demo's first cell
  `os.chdir`s into `docs/source/demos/<demo>/` relative to the root, and
  paths in later cells and the demo's `.toml` config are relative to that
  directory. Executing the demos requires the real repository data
  (`oac-download-data`).
