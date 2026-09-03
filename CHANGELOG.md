# Changelog

## [0.18.1] - 2026-09-03

## What's Changed
### Documentation
* Introduce native `uv` installation option by @liammegill in https://github.com/dlr-pa/oac/pull/142
### Other Changes
* Remove separate plot windows by @liammegill in https://github.com/dlr-pa/oac/pull/143
* Bump uv.lock to include sphinx-design by @liammegill in https://github.com/dlr-pa/oac/pull/144

**Full Changelog**: https://github.com/dlr-pa/oac/compare/v0.18.0...v0.18.1

## [0.18.0] - 2026-09-02

## What's Changed
### Added
* Package OpenAirClim for PyPI: pyproject.toml, replace cf-units with pint, pip/conda CI matrix, remove zenodo_get by @liammegill in https://github.com/dlr-pa/oac/pull/126
### Breaking Changes
* Prepare OpenAirClim for PyPI release by @liammegill in https://github.com/dlr-pa/oac/pull/128
### Fixed
* Refactor coverage badge generation in build-docs.yml by @liammegill in https://github.com/dlr-pa/oac/pull/138
### Maintenance
* Bump conda-incubator/setup-miniconda from 3 to 4 by @dependabot[bot] in https://github.com/dlr-pa/oac/pull/131
* Bump actions/cache from 4 to 6 by @dependabot[bot] in https://github.com/dlr-pa/oac/pull/132
* Bump actions/checkout from 4 to 7 by @dependabot[bot] in https://github.com/dlr-pa/oac/pull/133
* Bump actions/setup-python from 5 to 7 by @dependabot[bot] in https://github.com/dlr-pa/oac/pull/136
* Bump dorny/paths-filter from 3 to 4 by @dependabot[bot] in https://github.com/dlr-pa/oac/pull/134
### Other Changes
* Final preparations for initial PyPI release by @liammegill in https://github.com/dlr-pa/oac/pull/139
* Fix lambda CO2 by @liammegill in https://github.com/dlr-pa/oac/pull/130

## New Contributors
* @dependabot[bot] made their first contribution in https://github.com/dlr-pa/oac/pull/131

**Full Changelog**: https://github.com/dlr-pa/oac/compare/v0.17.0...v0.18.0

## [0.17.0] - 2026-08-25

### Added
- Graphical user interface, command-line entry points and pydantic validation by @liammegill in https://github.com/dlr-pa/oac/pull/123
    - Add GUI #117
    - Add command-line capability #118
- Update efficacies in `example.toml` to match MRV specification #115

### Breaking Changes
- Package reorganised into `openairclim/core/`, `openairclim/gui/` and `openairclim/addon/` #123
- Config directory paths no longer require a trailing `/` #123

**Full Changelog**: https://github.com/dlr-pa/oac/compare/v0.16.0...v0.17.0

## [0.16.0] - 2026-03-19

### Added
- Parametric module by @ahsawa in https://github.com/dlr-pa/oac/pull/106
    - Add parametric scenario module #84
    - Refactoring and integration by @stefan-voelk

### New Contributors
- @ahsawa made his first contribution in #106 

**Full Changelog**: https://github.com/dlr-pa/oac/compare/v0.15.0...v0.16.0

## [0.15.0] - 2026-03-18

### Added
- OpenAirClim Contrail Module by @liammegill in https://github.com/dlr-pa/oac/pull/109
    - New Ice Supersaturation Frequency #68 
    - Introduce contrail attribution methods #69
    - Add premium functionality #108 

### Fixed
- Protect CFDD calculation from too high/low plev values #86 
- Wingspan correction does not allow limit values #95

**Full Changelog**: https://github.com/dlr-pa/oac/compare/v0.14.0...v0.15.0

## [0.14.0] - 2026-03-18

### Added
- Introduced Stratospheric Water Vapour (SWV) module. #107 @atzeharmsen

### New Contributors
- @atzeharmsen made his first contribution in #107

**Full Changelog**: https://github.com/dlr-pa/oac/compare/v0.13.0...v0.14.0

## [0.13.0] - 2025-11-19

### Added

- Attribution methodologies for species CO2 and CH4: #96 @liammegill
    - Residual, marginal, proportional (default) and differential attribution
- Aircraft characteristics provided from csv file: #92 @liammegill
- Capability to switch on/off plots: #29 @liammegill
- Capability to switch off climate metrics: #75 @liammegill

### Fixed

- Normalization / scaling uses incorrect reference emission inventory: #97 @liammegill

## [0.12.0] - 2025-09-23

### Added

- Online documentation on [openairclim.org](https://openairclim.org/) bundling different types of documentation: #82 @liammegill @stefan-voelk
    - Introduction
    - Installation guide
    - Getting started
    - User Guide (new)
    - Demonstrations (new)
    - Scientific Background (new) #62 @liammegill
    - Publications (new)
    - API Reference
    - Governance (new)
    - Contact and Support (new)
    - Bibliography

### Updates

- Streamline README considering website as a new focus for documentation

## [0.11.1] - 2025-04-15

### Fixed

- Fixed `PermissionError` when example input directory does not yet exist. #76 @stefan-voelk

## [0.11.0] - 2025-04-02

### Added

- Capability for multiple aircraft to be present within the input emission inventory along data variable "ac"

### Updates

- Added capability for multiple aircraft within same emission inventory. #16 @liammegill
- Fixed logger handlers at end of OpenAirClim run. #66 @liammegill

## [0.10.0] - 2025-03-06

### Added

- Contrails module: Megill_2025 methodology after [Megill & Grewe, in prep.]( https://doi.org/10.5194/egusphere-2024-3398)

### Updates

- Time evolution with function `adjust_inventories(config, inv_dict)` for application on emission inventories **before** simulation, see [workflow documentation](docs/workflows/workflows.md)

## [0.9.0] - 2024-12-04

### Added

- Species: $O_3$, $CH_4$, PMO and Contrails

### Limitations

- Limited resolution of response surfaces and pending validation for species $O_3$, $CH_4$ and PMO
- Stratospheric Water Vapor (SWV) not considered in this version
- Contrails module: AirClim 2.1 methodology including simulations for $H_2$ from AHEAD project
- Climate impact of longer species lifetimes in the stratosphere not considered
- Overhanging effect on next year not considered for species lifetimes in the order of time step (year)

### Updates

- Change of versioning scheme to [semantic versioning](https://semver.org/)
- Move repository directory
- Integrate default configuration settings

## [2.8.3] - 2024-09-04

### Added

- Processing of 4D emission data sets: (lon, lat, plev) for multiple inventory years
- Supported species: $CO_2$ and $H_2O$
- Temperature evolution and climate metrics
- Some response functions available
