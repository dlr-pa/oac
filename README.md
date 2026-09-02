# OpenAirClim

![Pip installation](https://github.com/dlr-pa/oac/actions/workflows/pip-install-test.yml/badge.svg)
![Conda installation](https://github.com/dlr-pa/oac/actions/workflows/conda-install-test.yml/badge.svg)
[![Docs](https://github.com/dlr-pa/oac/actions/workflows/build-docs.yml/badge.svg)](https://openairclim.org)
![Coverage](https://openairclim.org/_static/coverage.svg)
<br>
[![pypi](https://img.shields.io/pypi/v/openairclim?color=orange&label=pypi&logo=python&logoColor=white)](https://pypi.org/project/openairclim/)
[![pypi - python version](https://img.shields.io/pypi/pyversions/openairclim.svg?color=orange&logo=python&label=python&logoColor=white)](https://pypi.org/project/openairclim/)
[![downloads](https://img.shields.io/pypi/dm/openairclim)](https://pypi.org/project/openairclim/)
<br>
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Latest tag](https://img.shields.io/github/v/tag/dlr-pa/oac)](https://github.com/dlr-pa/oac/tags)
[![Commits since last release](https://img.shields.io/github/commits-since/dlr-pa/oac/latest.svg)](https://github.com/dlr-pa/oac/commits/main)
[![Contributors](https://img.shields.io/github/contributors/dlr-pa/oac)](https://github.com/dlr-pa/oac/graphs/contributors)
[![License](https://img.shields.io/github/license/dlr-pa/oac)](https://github.com/dlr-pa/oac/blob/main/LICENSE)
<br>
[![DOI](https://zenodo.org/badge/851165490.svg)](https://zenodo.org/doi/10.5281/zenodo.13682728)


## Description

OpenAirClim is an open-source response model for quantifying the climate impact
of air traffic emissions. Rather than explicitly simulating physical processes,
it uses response functions derived from comprehensive climate-chemistry models.
This makes OpenAirClim particularly fast and efficient, with individual runs
taking seconds to minutes on a conventional computer.

## Motivation

Aviation operations account for around **3.5% of Effective Radiative Forcing**
and its share is expected to grow. A large part of aviation's impact arises
from non-CO2 effects, in particular nitrogen oxide emissions and the formation
of contrails. The impact of non-CO2 effects is highly dependent on the location
and time of the emission, as well as on the characteristics of the emitting
aircraft. Emerging aircraft and fuels (e.g. SAF, hydrogen, hybrid-electric)
demand **new, open and efficient tools** to quantify their climate impacts.
However, existing models are either closed, too general or computationally
intense.

OpenAirClim and its add-ons constitute an open-source framework to rapidly
model aviation emissions and their climate response: supporting science,
industry and policy. Development is being led by the German Aerospace Center
(Deutsches Zentrum für Luft- und Raumfahrt, DLR) and includes various research
and industry partners.

### Highlights
OpenAirClim builds upon the previous AirClim framework. Compared to AirClim,
the new OpenAirClim framework:

- Provides standardised, open formats for the simulation configuration file,
    emission inventories and results
- Provides a **Graphical User Interface** (GUI) for interactive configuration
    and results exploration
- Handles **multiple emission inventories** over time (4D dependence)
- Allows **attribution of climate impact** to specific aircraft or fleets
- Implements **tagging** for atmospheric chemistry
- Extends contrail calculations to **novel aviation fuels**
- Enables the calculation of parametric scenarios at post-processing level, 
    e.g. climate optimised routing
- Provides **uncertainty and robustness metrics** (work in progress)
- Provides various outputs, including time series of radiative forcing and
    temperature change, various climate metrics and sea-level rise

### Typical use cases
OpenAirClim is aimed both at research and industry. Typical research questions
that can be answered by using OpenAirClim relate to:

- fleet-wide scenarios, e.g. the introduction of a new aircraft type; climate
    impact of operations from a specific airline or airport
- aviation industry scenarios, e.g. the introduction of a new fuel type;
    climate-optimal distribution of SAF
- operational procedures, e.g. intermediate stop operations; flying 
    lower and slower

### Layout
![Overview on the layout of the OpenAirClim framework](https://raw.githubusercontent.com/dlr-pa/oac/main/docs/source/_static/OAC-chart.png)
<figcaption>Overview of the OpenAirClim framework</figcaption>


## Documentation

Please refer to [openairclim.org](https://openairclim.org/) for the
documentation of the OpenAirClim framework. The documentation includes
installation manuals, quick-start and user guides, example demonstrations, an
API reference, as well as information on the scientific background and
OpenAirClim governance.


## Installation

OpenAirClim is currently available from PyPI at
https://pypi.org/project/openairclim or from source at
https://github.com/dlr-pa/oac. Later OpenAirClim versions will also be
available from conda-forge (work in progress).

### Install with pip
To install OpenAirClim from PyPI with [pip](https://pip.pypa.io/en/stable/)
(Python 3.11 or later required):

```bash
pip install openairclim

# install with optional dependencies (gui, docs, test, dev)
pip install openairclim[dev]
```

To install the latest development version directly from GitHub, you have two
options:

```bash
# with git
git clone https://github.com/dlr-pa/oac.git

# with pip
pip install git+https://github.com/dlr-pa/oac.git
```

### Install with conda
To install OpenAirClim with conda, make sure that either the
[conda](https://docs.conda.io/en/latest/) or
[mamba](https://mamba.readthedocs.io/en/latest/) package manager is installed
on your system. Currently, the only installation possibility with conda is by
first cloning the [GitHub repository](https://github.com/dlr-pa/oac) and then
installing the required dependencies using the provided `environment_xxx.yaml`
files:

```bash
git clone https://github.com/dlr-pa/oac-git
cd oac
conda env create -f environment_xxx.yaml
conda activate <env>

# optional: install GUI dependencies
conda env update -f environment_gui.yaml -n <env>

# install OpenAirClim in the conda environment with pip
pip install .
```

Replace `xxx` with either `minimal` or `dev` (full installation) and `<env>`
with the correct name of the conda environment (e.g. `oac` or `oac_minimal`).
To install an editable version of the `openairclim` package, allowing you to
make changes to the source code and see those changes reflected immediately,
use `pip install -e .` instead.


## Getting started

### Download repository data
OpenAirClim's response surfaces and background concentration scenarios are
published separately, in [dlr-pa/oac-data](https://github.com/dlr-pa/oac-data).
To download the data to a shared cache, use:

```bash
oac-download-data
```

See the [installation guide](https://openairclim.org/installation.html) for
override options.

### Emission inventories
Air traffic emission inventories are an essential input to OpenAirClim. You can
download example emission inventories based on the DLR project
[DEPA 2050](https://elib.dlr.de/142185/)
[here](https://doi.org/10.5281/zenodo.11442322).
These inventories comprise realistic air traffic between 2020 and 2070.

If you are interested in testing or developing OpenAirClim, you might want to
generate artificial data. This can be done using command line scripts from 
[openairclim/utils/](https://github.com/dlr-pa/oac/tree/main/openairclim/utils):

```bash
oac-create-artificial-inventories -o example/input/
oac-create-time-evolution -o example/input/
```

### Graphical User Interface
OpenAirClim ships with an optional GUI for creating, loading and editing
configuration files, inspecting input data, running simulations and exploring
results. Provided the `gui` dependencies have been installed, it can be
launched using:

```bash
oac-gui
```

### Run OpenAirClim
OpenAirClim can be run from the command line using:

```bash
cd path/to/working/directory
oac-run <config-name>.toml
```

Note that if there are any relative links in the config file (e.g.
`dir = input/`), you must be in the right working directory for OpenAirClim to
run successfully.

### Create test files
If you contribute to the development of OpenAirClim, you will require
additional test files. To create them, use:

```bash
python -m openairclim.utils.create_test_files -o tests/core/repository/
```


## Roadmap

The scheduling of major software releases and milestone planning are partially
dependent on the contractual framework with our stakeholders. For the version
history of the completed releases, see the [changelog](CHANGELOG.md). The full
development stage as currently planned is shown in the [layout](#layout).

## Contributing
Contributions are very welcome. Please read our
[contribution guidelines](CONTRIBUTING.md) to get started.

## License
OpenAirClim is licensed under Apache 2.0, a copy of which can be found
[here](LICENSE).
