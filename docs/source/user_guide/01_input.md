# Input data

OpenAirClim requires several input data to be present before executing a simulation run.

## Configuration file

A configuration file serves as the main user interface to the OpenAirClim framework. The [TOML](https://toml.io/en/) format is used which is known for its simple syntax and human readability. Refer to `example/example.toml` for an example configuration.

The configuration file is structured using *tables* which are collections of key/value pairs. Each table is defined by a header, i.e. a `[string]` enclosed by square brackets. Each table represents a section of the configuration file.

The comments in `example.toml` describe specific settings more in detail. Here, an overview over the different tables (sections) of the configuration file is given:

- `[species]` Here, the atmospheric species are defined which are present in the emission inventories, and those species producing the climate impact (response). In some cases, the species defined in the `inv` array and those defined in the `out` array are the same, for example "CO2". In other cases, the response species differ from the species given in the inventory. For example, "NOx" produces a climate response through other species ("O3", "CH4" and "PMO"). By changing the arrays, the simulation of climate impacts can be switched on and off for individual species.
- `[inventories]` This section specifies the input directory and an array of emission inventory files which are considered for the simulation run. Additionaly, base emission inventories can be defined (only relevant for the computation of contrail climate impacts).
- `[output]` Here, settings for the output of simulation results are defined. Using the flags `full_run` and `concentrations`, parts of the simulation workflow can be switched on and off.
- `[time]` Settings regarding the time dimension are specified here. The `range` setting defines the period and step in years considered for the simulation run. If `file` is set in this section, an additional time evolution is read in and processed. Refer to the documentation *Time Evolution* for more details.
- `[background]` Here, the atmospheric backgrounds of atmospheric species and considering several Shared Socioeconomic Pathway (SSP) scenarios are defined. The atmospheric background is relevant for the computation of climate impacts.
- `[responses]` This section comprises settings of the implemented response surfaces and methodologies used.
- `[temperature]` This section defines the climate sensitivity parameters and efficacies of atmospheric species relevant for the computation of temperature changes.
- `[metrics]` The array `types` defines the climate metrics which should be computed and written to the output. The arrays `H` and `t_0` define time horizons and start times for the metrics calculations. The program iterates over these arrays permuting over all combinations.
- `[aircraft]` The strings in array `types` correspond to (optional) aircraft identifiers present in the emission inventories. This functionality is convenient for the classification of different aircraft types with different properties relevant for the climate impacts.

## Emission inventories

The emission inventories comprise spatially resolved aircraft emissions on a yearly basis. Refer to the example emission inventories, either generated via script `create_artificial_inventories.py`, or the inventories available for [download](https://doi.org/10.5281/zenodo.11442323) from the DLR project [Development Pathways for Aviation up to 2050 (DEPA 2050)](https://elib.dlr.de/142185/).

![inventory](../_static/emission-inventory.png)

The emission inventories are stored as netCDF files using a flat data structure, i.e. an unordered list of entries (not-gridded). Only the naming conventions and units defined in the example inventories should be used. The entry `Inventory_Year` in the attribute section of the netCDF file defines the inventory year.

## Time evolution (optional)

If no extra evolution file is specified in the configuration, OpenAirClim performs a temporal interpolation between discrete inventory years. Alternatively, a time evolution of type **normalization** or **scaling** can be specified in another netCDF file. For more details on that topic, refer to the *Time Evolution* documentation and the example evolution files generated via script `create_time_evolution.py`.