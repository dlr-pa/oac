# OpenAirClim

![Pip installation](https://github.com/dlr-pa/oac/actions/workflows/pip-install-test.yml/badge.svg)
![Conda installation](https://github.com/dlr-pa/oac/actions/workflows/conda-install-test.yml/badge.svg)
[![Docs](https://github.com/dlr-pa/oac/actions/workflows/build-docs.yml/badge.svg)](https://openairclim.org)
![Coverage](https://openairclim.org/_static/coverage.svg)
[![Python versions](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://github.com/dlr-pa/oac/blob/main/pyproject.toml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Latest tag](https://img.shields.io/github/v/tag/dlr-pa/oac)](https://github.com/dlr-pa/oac/tags)
[![Commits since last release](https://img.shields.io/github/commits-since/dlr-pa/oac/latest.svg)](https://github.com/dlr-pa/oac/commits/main)
[![Contributors](https://img.shields.io/github/contributors/dlr-pa/oac)](https://github.com/dlr-pa/oac/graphs/contributors)
[![License](https://img.shields.io/github/license/dlr-pa/oac)](https://github.com/dlr-pa/oac/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/851165490.svg)](https://zenodo.org/doi/10.5281/zenodo.13682728)


## Description

OpenAirClim is a model for simplified evaluation of the approximate chemistry-climate impact of air traffic emissions. The model represents the major responses of the atmosphere to emissions in terms of composition and climate change. Instead of applying time-consuming climate-chemistry models, a response model reproduces the response of a climate-chemistry model without actually calculating ab initio all the physical and chemical effects. The responses are non-linear relations between localized emissions and Radiative Forcing and further climate indicators. These response surfaces are contained within look-up tables. OpenAirClim builds upon the previous AirClim framework.

In comparison with AirClim, the following new features are introduced:

- Standardised formats for configuration file (user interface) and emission inventories (input) and program results (output)
- Possibility of full 4D emission inventories (3D for several time steps)
- Non-linear response functions for NOx including contribution approach (tagging) and dependency on background
- Contrail formation also depending on fuels and overall efficiencies
- Inclusion of different fuels
- Choice of different CO2 response models
- Choice of temperature models and sea-level rise
- Uncertainty assessment and Robustness Metric based on Monte Carlo Simulations
- Parametric scenarios as sensitivities, e.g. at post-processing level: climate optimized routings
- Graphical user interface (GUI) for interactive configuration and results exploration

### Scientific Background

The impact of aviation on climate amounts to approximately 5% of the total anthropogenic climate warming. A large part of the aviation’s impact arises from non-CO2 effects, especially contrails and nitrogen oxide emissions. Impact of non-CO2 effects depend in particular on the location and time of emissions, hence a regional dependence of impacts exists. As impacts of individual non-CO2 effects show a different spatial dependence, the relationship between impacts and associated emissions can be best described in non-linear relationships, i.e. equations or algorithms based on look-up tables. Specifically, the climate impact of an aircraft depends on where (and when) an aircraft is operated. In addition, using different types of fuel generally changes the importance of the non-CO2 effects.

## Layout

![Overview on the layout of the OpenAirClim framework](https://raw.githubusercontent.com/dlr-pa/oac/main/docs/source/_static/OAC-chart.png)
<figcaption>Overview on the layout of the OpenAirClim framework</figcaption>

- User interface for settings in the run control and outputs (grey)
- Definition of background conditions, such as aviation scenarios, uncertainty ranges and aviation inventories (orange)
- A link to a pre-processor for aviation inventories (light blue).
- Processor for a full 4D-emission inventory at multiple timesteps (violet)
- A framework for the application of non-linear response functions (red) to these emission inventories.
- Response functions for CO2 and climate / temperature and sea-level changes
- Parametric scenarios as sensitivities (yellow), e.g. at post-processing level: climate optimized routings
- Output: Warnings, errors (log files), climate indicators and diagnostics (green), values of climate metrics and robustness metrics (grey)

## Graphical User Interface

OpenAirClim ships with an optional graphical user interface (GUI) for creating, loading and editing configuration files, inspecting input data, running simulations and exploring results, without writing or editing TOML by hand. See [Installation](#installation) for how to add the GUI's dependencies and [Usage](#usage) for how to launch it, or the [GUI documentation](https://openairclim.org/gui.html) for the full picture.

## Documentation

Please refer to [openairclim.org](https://openairclim.org/) for the documentation of the OpenAirClim framework.
This documentation includes installation manuals, quick-start and user guides, example demonstrations, an API reference, as well as information on the scientific background and OpenAirClim governance.

## Installation

If you build OpenAirClim from source, you first have to access the [repository](https://github.com/dlr-pa/oac). To obtain the repository, the most convenient way is using following [Git](https://git-scm.com/) command:
```bash
git clone https://github.com/dlr-pa/oac.git
```

Once the repository has been cloned, there are two options to install the necessary packages.

### Installation using conda

Make sure that either the [conda](https://docs.conda.io/projects/conda/en/latest/index.html) or [mamba](https://mamba.readthedocs.io/en/latest/index.html) package manager is installed on your system.

The source code includes configuration files `environment_xxx.yaml` that enable the installation of a virtual conda environment with all required dependencies. This installation method is suitable for working across platforms. Change directory to the root folder of the downloaded source, create a conda environment and activate it:
```bash
cd oac
conda env create -f environment_xxx.yaml
conda activate <env>
```

Replace `xxx` with the relevant file and `<env>` with the correct name of the installed conda environment, e.g. `oac` or `oac_minimal`.
To add the optional dependencies for the [GUI](#graphical-user-interface), update the conda environment using:
```bash
conda env update -f environment_gui.yaml -n <env>
```

Finally, to install the openairclim package system-wide on your computer, execute one of the following commands within the activated conda environment.
This last installation step isn't necessary if the user has otherwise added the path to the oac source folder to `PYTHONPATH`.
```bash
pip install .
```
or
```bash
pip install -e .
```
The `-e` flag treats the openairclim package as an editable install, allowing you to make changes to the source code and see those changes reflected immediately. The latter command is recommended for developers.

### Installation using pip

Alternatively, change directory to the root folder of the downloaded source and install directly with `pip` (Python >= 3.11.5 required):
```bash
pip install .
```
or, for an editable install:
```bash
pip install -e .
```
If you are planning on making changes to the code or contributing to the development of OpenAirClim, install the `dev` extra instead, which pulls in testing, linting and documentation tooling on top of the base dependencies:
```bash
pip install ".[dev]"
```
To add the optional dependencies for the [GUI](#graphical-user-interface), install the `gui` extra:
```bash
pip install ".[gui]"
```

After installing the required dependencies, proceed with the steps described in section [Getting started](#getting-started).


## Getting started

### Download repository data
OpenAirClim's response surfaces and background concentration scenarios are published separately, in [dlr-pa/oac-repository](https://github.com/dlr-pa/oac-repository) — cloning `oac` no longer includes this data. Download it once with:
```bash
oac-download-data
```
This fetches the data into a shared cache directory that any config file leaving `background.dir`/`responses.dir` unset automatically resolves to. See the [installation guide](https://openairclim.org/installation.html) for override options (custom location, specific version/record).

### Download emission inventories
Air traffic emission inventories are essential input to OpenAirClim. You can [download](https://doi.org/10.5281/zenodo.11442322) example emission inventories based on the DLR project [Development Pathways for Aviation up to 2050 (DEPA 2050)](https://elib.dlr.de/142185/). These inventories comprise realistic emission data sets.

Depending on the settings made in the configuration file, the computational time of the configured simulations could be long. If you are more interested in testing or developing OpenAirClim software, you might want to generate artificial data.

### Create input data
If you do not have custom input files available, input files with artificial data can be autogenerated using command line scripts from [openairclim/utils/](https://github.com/dlr-pa/oac/tree/main/openairclim/utils):
```bash
python -m openairclim.utils.create_artificial_inventories -o "example/input/"
python -m openairclim.utils.create_time_evolution -o "example/input/"
```
The script `create_artificial_inventories.py` creates a series of inventories comprising random emission data. The script `create_time_evolution.py` creates two time evolution files, controlling the temporal evolution of the emission data: one file is intended for normalizing inventory emission data, and the other file is intended for scaling inventory emission data along the time axis. Emission inventories and time evolution files are both .nc files and are located in directory [example/input](https://github.com/dlr-pa/oac/tree/main/example/input).

### Create test files
If you contribute to the software development of OpenAirClim, you will probably execute the testing procedures which require additional test files. Following command, run from the repository root, creates these files:
```bash
python -m openairclim.utils.create_test_files -o tests/core/repository/
```
### Usage

After installation, OpenAirClim can be run from the command line using:
```bash
oac-run path/to/config.toml
```
(equivalently, `python -m openairclim path/to/config.toml`)

Alternatively, the package can be imported and used in Python scripts using:
```python
import openairclim as oac
```

Once the [optional GUI dependencies](#graphical-user-interface) are installed, the graphical user interface can instead be used to create, edit and run configurations interactively, without writing TOML or Python directly:
```bash
oac-gui
```
(equivalently, `python -m openairclim.gui`)
See the [GUI documentation](https://openairclim.org/gui.html) for more details, including how to pass a config or results file on the command line.

Refer to the [example/](https://github.com/dlr-pa/oac/tree/main/example) folder within the repository for a minimal example and the demonstrations given on [openairclim.org](https://openairclim.org/).


## Roadmap

The scheduling of major software releases and milestone planning are partially dependent on the contractual framework with our stakeholders. For the version history of the completed releases, see the [changelog](CHANGELOG.md). The full development stage as currently planned is described in the [layout](#layout).

## Contributing
Contributions are very welcome. Please read our [contribution guidelines](CONTRIBUTING.md) to get started.

## License
The license of the OpenAirClim software can be found [here](LICENSE).
