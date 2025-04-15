# Changelog

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