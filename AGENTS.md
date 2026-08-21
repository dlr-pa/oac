# AGENTS.md

Orientation for coding agents working in this repo. For contribution process,
governance, and commit conventions, see `CONTRIBUTING.md`.

## What this is

OpenAirClim (`openairclim/`) is a climate response model or air traffic
emissions. `core/` is the simulation engine; `gui/` is an optional Panel-based
GUI on top of it; `addon/` integrates optional premium functionality
(feature-detected, may not be installed).

## Environment / running tests

```bash
conda env create -f environment_dev.yaml   # or environment_minimal.yaml
conda env update -f environment_gui.yaml -n <env>   # to add GUI deps
pytest tests/
```

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
- `docs/source/api_ref/{core,gui}/*.rst` — one stub file per module,
  `automodule::`-based. `api_ref.rst` globs the folder automatically, but a
  new module still needs its own stub file or it won't appear.

## Conventions worth knowing

- Docstrings are Google-style (napoleon); cross-reference other modules with
  Sphinx roles (`` :func:`~pkg.mod.func` ``) since docstrings render in the
  Sphinx docs.
- `openairclim/core/*` changes get reviewed closely by the maintainer. Always
  confirm there before making them.
