# Changelog

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