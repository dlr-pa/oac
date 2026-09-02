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
conda env create -f environment_dev.yaml   # or environment_minimal.yaml
conda env update -f environment_gui.yaml -n <env>   # to add GUI deps
pytest tests/
```

`environment_dev.yaml` pins `python<3.14`: Prospector's mypy integration
crashes outright under Python 3.14 (an upstream Prospector/mypy/argparse
incompatibility, unrelated to this codebase). If setting up a dev environment
via `pip install ".[dev]"` instead of conda, use a 3.11-3.13 interpreter for
the same reason — `pip` won't manage/select this for you. This pin is
dev-tooling-only, not a statement about which Python versions OpenAirClim
itself supports (see `requires-python` in `pyproject.toml`). CI's conda
install test therefore builds its environment from `environment_minimal.yaml`
+ `environment_gui.yaml` rather than `environment_dev.yaml`, so it can still
cover the full supported Python range.

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

## Conventions worth knowing

- Docstrings are Google-style (napoleon); cross-reference other modules with
  Sphinx roles (`` :func:`~pkg.mod.func` ``) since docstrings render in the
  Sphinx docs.
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
- `docs/source/demos/*` pages are MyST Markdown notebooks (`file_format:
  mystnb` front matter), executed by `myst_nb` during `sphinx-build` (run
  from `docs/`, not repo root) and not covered by `pytest`. Execution cwd is
  each notebook's own directory (`docs/source/demos/<demo>/`), not `docs/` —
  paths inside a demo's code cells and its `.toml` config are relative to
  that directory. `nb_execution_mode = "cache"` (see `conf.py`) only
  re-executes a notebook when its content changes (keyed by content hash in
  `docs/build/.jupyter_cache`) — verify changes to a demo with an actual docs
  build, not just the test suite, and expect that build to take longer/hit
  the network the first time or after editing that demo's `.md`/`.toml`.
