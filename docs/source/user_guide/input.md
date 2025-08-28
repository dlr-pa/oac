# Input data

OpenAirClim requires several input data to be present before executing a simulation run.

## Configuration

A configuration file serves as the main user interface to the OpenAirClim framework. The [TOML](https://toml.io/en/) format is used which is known for its simple syntax and human readability. Refer to the `example` folder of the framework for an example configuration `example.toml`.

The configuration file is structured using *tables* which are collections of key/value pairs. Each table is defined by a header, i.e. a `[word]` enclosed by square brackets. Each table represents a section of the configuration file.

The comments in the example configuration describe specific settings more in detail. Here, we give an overview over the different tables (sections) of the configuration file:

- `[species]` Here, the atmospheric species are defined which are present in the emission inventories, and those species producing the climate impact (response). In some cases, the species defined in the `inv` array and those defined in the `out` array are the same, for example "CO2". In other cases, the response species differ from the species given in the inventory. For example, "NOx" produces a climate response through other species ("O3", "CH4" and "PMO"). By changing the arrays, the simulation of climate impacts can be switched on and off for individual species.
- `[inventories]` This section defines the input directory and an array of emission inventory files which are considered for the simulation run. Additionaly, base emission inventories can be defined (only relevant for the computation of contrail climate impacts).
- `[output]` Here, settings for the output of simulation results are defined. The flags `full_run` and `concentrations` give the option to switch on and off parts of the simulation workflow. The setting `full_run = False` requires that there are available previously simulated time series of Radiative Forcings and temperature changes. The naming of the directory and file must match the settings given in this section. The computation of climate metrics will then be performed as a post-processing step.
- `[time]` Settings regarding the time dimension are specified here. The `range` setting defines the period and step in years considered for the simulation run. If `file` is set in this section, an additional time evolution is read in and processed. Refer to the documentation *Time evolution* for more details.
- `[background]` Here, the atmospheric background is defined for certain atmospheric species which are relevant for the computation of climate impacts.


## Emission inventories


## Time evolution (optional)