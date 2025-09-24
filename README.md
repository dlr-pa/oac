# OpenAirClim

[![DOI](https://zenodo.org/badge/851165490.svg)](https://zenodo.org/doi/10.5281/zenodo.13682728)
![Install and Test workflow](https://github.com/dlr-pa/oac/actions/workflows/install_and_test.yml/badge.svg)


## Description

OpenAirClim is a model for simplified evaluation of the approximate chemistry-climate impact of air traffic emissions. The model represents the major responses of the atmosphere to emissions in terms of composition and climate change. Instead of applying time-consuming climate-chemistry models, a response model is developed and applied which reproduces the response of a climate-chemistry model without actually calculating ab initio all the physical and chemical effects. The responses are non-linear relations between localized emissions and Radiative Forcing and further climate indicators. These response surfaces are contained within look-up tables. OpenAirClim builds upon the previous AirClim framework. In comparison with AirClim, following new features are introduced:

- Standardized formats for configuration file (user interface) and emission inventories (input) and program results (output)
- Possibility of full 4D emission inventories (3D for several time steps)
- Non-linear response functions for NOx including contribution approach (tagging) and dependency on background
- Contrail formation also depending on fuels and overall efficiencies
- Inclusion of different fuels
- Choice of different CO2 response models
- Choice of temperature models and sea-level rise
- Uncertainty assessment and Robustness Metric based on Monte Carlo Simulations
- Parametric scenarios as sensitivities, e.g. at post-processing level: climate optimized routings

### Scientific Background

The impact of aviation on climate amounts to approximately 5% of the total anthropogenic climate warming. A large part of the aviation’s impact arises from non-CO2 effects, especially contrails and nitrogen oxide emissions. Impact of non-CO2 effects depend in particular on the location and time of emissions, hence a regional dependence of impacts exists. As impacts of individual non-CO2 effects show a different spatial dependence, the relationship between impacts and associated emissions can be best described in non-linear relationships, i.e. equations or algorithms based on look-up tables. Specifically, the climate impact of an aircraft depends on where (and when) an aircraft is operated. In addition, using different types of fuel generally changes the importance of the non-CO2 effects.

## Layout

![Overview on the layout of the OpenAirClim framework](img/OAC-chart.png)
<figcaption>Overview on the layout of the OpenAirClim framework</figcaption>

- User interface for settings in the run control and outputs (<grey>grey</grey>)
- Definition of background conditions, such as aviation scenarios, uncertainty ranges and aviation inventories (<orange>orange</orange>)
- A link to a pre-processor for aviation inventories (<blue>light blue</blue>).
- Processor for a full 4D-emission inventory at multiple timesteps (<magenta>violet</magenta>)
- A framework for the application of non-linear response functions (<red>red</red>) to these emission inventories.
- Response functions for CO2 and climate / temperature and sea-level changes
- Parametric scenarios as sensitivities (<yellow>yellow</yellow>), e.g. at post-processing level: climate optimized routings
- Output: Warnings, errors (log files), climate indicators and diagnostics (<green>green</green>), values of climate metrics and robustness metrics (<grey>grey</grey>)

## Documentation

Please refer to [openairclim.org](https://openairclim.org/) for the documentation of the OpenAirClim framework.
This documentation includes installation manuals, quick-start and user guides, example demonstrations, an API reference, as well as information on the scientific background and OpenAirClim governance.

## Installation

If you build OpenAirClim from source, you first have to access the [repository](https://github.com/dlr-pa/oac). To obtain the repository, the most convenient way is using following [Git](https://git-scm.com/) command:
```
git clone https://github.com/dlr-pa/oac.git
```

Make sure that either the [conda](https://docs.conda.io/projects/conda/en/latest/index.html) or [mamba](https://mamba.readthedocs.io/en/latest/index.html) package manager is installed on your system.

The source code includes configuration files `environment_xxx.yaml` that enable the installation of a virtual conda environment with all required dependencies. This installation method is suitable for working across platforms. Change directory to the root folder of the downloaded source and create a conda environment:
```
cd oac
conda env create -f environment_xxx.yaml
```

Finally, to install the openairclim package system-wide on your computer, execute one of the following commands:
```
pip install .
```
or
```
pip install -e .
```
The `-e` flag treats the openairclim package as an editable install, allowing you to make changes to the source code and see those changes reflected immediately. The latter command is recommended for developers.

After having installed the conda ennvironment and required dependencies, proceed with the steps described in section [Getting started](##getting-started). 


## Getting started

### Download emission inventories
Air traffic emission inventories are essential input to OpenAirClim. You can [download](https://doi.org/10.5281/zenodo.11442322) example emission inventories based on the DLR project [Development Pathways for Aviation up to 2050 (DEPA 2050)](https://elib.dlr.de/142185/). These inventories comprise realistic emission data sets.

Depending on the settings made in the configuration file, the computational time of the configured simulations could be long. If you are more interested in testing or developing OpenAirClim software, you might want to generate artificial data.

### Create input data
If you do not have custom input files available, input files with artificial data can be autogenerated using command line scripts. For that, change directory to [utils/](utils/) and execute following commands in order to create artificial input files:
```
cd utils/
python create_artificial_inventories.py
python create_time_evolution.py
```
The script `create_artificial_inventories.py` creates a series of inventories comprising random emission data. The script `create_time_evolution.py` creates two time evolution files, controlling the temporal evolution of the emission data: one file is intended for normalizing inventory emission data, and the other file is intended for scaling inventory emission data along the time axis. Emission inventories and time evolution files are both .nc files and are located in directory [example/input](example/input/).

### Create test files
If you contribute to the software development of OpenAirClim, you will probably execute the testing procedures which require additional test files. Following command creates these files:
```
python create_test_files.py
```
### Usage

After installation, the package can be imported and used in Python scripts:
```
import openairclim as oac
```

Refer to the [example/](example/) folder within the repository for a minimal example and the demonstrations given on [openairclim.org](https://openairclim.org/).


## Roadmap

The scheduling of major software releases and milestone planning are partially dependent on the contractractual framework with our stakeholders. For the version history of the completed releases, see the [changelog](CHANGELOG.md). The full development stage as currently planned is described in the [layout](#layout).

## Contributing
Contributions are very welcome. Please read our [contribution guidelines](CONTRIBUTING.md) to get started.

## License
The license of the OpenAirClim sofware can be found [here](LICENSE).
