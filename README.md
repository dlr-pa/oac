# OpenAirClim

![pip installation](https://github.com/dlr-pa/oac/actions/workflows/pip-install-test.yml/badge.svg)
![conda installation](https://github.com/dlr-pa/oac/actions/workflows/conda-install-test.yml/badge.svg)
![pixi installation](https://github.com/dlr-pa/oac/actions/workflows/pixi-install-test.yml/badge.svg)
[![Docs](https://github.com/dlr-pa/oac/actions/workflows/build-docs.yml/badge.svg)](https://openairclim.org)
![Coverage](https://openairclim.org/_static/coverage.svg)
<br>
[![Latest tag](https://img.shields.io/github/v/tag/dlr-pa/oac?logo=github&label=github)](https://github.com/dlr-pa/oac/tags)
[![pypi](https://img.shields.io/pypi/v/openairclim?color=orange&label=pypi&logo=pypi&logoColor=white)](https://pypi.org/project/openairclim/)
[![conda](https://img.shields.io/conda/vn/conda-forge/openairclim?label=conda-forge&logo=conda-forge&logoColor=white)](https://anaconda.org/conda-forge/openairclim)
[![pypi - python version](https://img.shields.io/pypi/pyversions/openairclim.svg?color=orange&logo=python&logoColor=white)](https://pypi.org/project/openairclim/)
<br>
[![Commits since last release](https://img.shields.io/github/commits-since/dlr-pa/oac/latest.svg)](https://github.com/dlr-pa/oac/commits/main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![Contributors](https://img.shields.io/github/contributors/dlr-pa/oac)](https://github.com/dlr-pa/oac/graphs/contributors)
[![License](https://img.shields.io/github/license/dlr-pa/oac)](https://github.com/dlr-pa/oac/blob/main/LICENSE)
<br>
[![code](https://img.shields.io/badge/10.5281%2Fzenodo.13682728-blue?logo=DOI&logoColor=white&label=code)](https://doi.org/10.5281/zenodo.13682728)
[![data](https://img.shields.io/badge/10.5281%2Fzenodo.22146822-blue?logo=DOI&logoColor=white&label=data)](https://doi.org/10.5281/zenodo.22146822)

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
industry and policy. Development is being led by the
[German Aerospace Center](https://dlr.de/pa/) (Deutsches Zentrum für Luft- und
Raumfahrt, DLR) and includes various research and industry partners.

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
installation manuals, quickstart, user and developer guides, example
demonstrations, an API reference, as well as information on the scientific
background and OpenAirClim governance.

## Installation

OpenAirClim is available from [PyPI](https://pypi.org/project/openairclim),
[conda-forge](https://anaconda.org/conda-forge/openairclim) and
[from source](https://github.com/dlr-pa/oac). OpenAirClim supports installation
via [pip](https://pip.pypa.io/en/stable/), [uv](https://docs.astral.sh/uv/),
[conda](https://docs.conda.io/en/latest/) and [pixi](https://pixi.sh).

### For Users

If you are a _user_ of OpenAirClim and are not planning on developing the
model, use one of the following setups. See our
[user installation](https://openairclim.org/user_guide/installation)
documentation for more details. If you are a _developer_, see the next section.

<details>
<summary>Install with pip</summary>

To install OpenAirClim from PyPI with [pip](https://pip.pypa.io/en/stable/)
(Python 3.11 or later required):

```bash
pip install openairclim

# install with optional dependencies (gui, docs, test, dev)
pip install openairclim[dev]
```

</details>

<details>
<summary>Install with uv</summary>

[uv](https://docs.astral.sh/uv/) is a fast Python package and project
manager. As a drop-in replacement for pip, it can install the published
package the same way:

```bash
uv pip install openairclim

# install with optional dependencies (gui, docs, test, dev)
uv pip install openairclim[dev]
```

</details>

<details>
<summary>Install with conda</summary>

To install OpenAirClim with conda, make sure that either the
[conda](https://docs.conda.io/en/latest/) or
[mamba](https://mamba.readthedocs.io/en/latest/) package manager is installed
on your system. Then, install from
[conda-forge](https://anaconda.org/conda-forge/openairclim):

```bash
conda install -c conda-forge openairclim
```

The conda-forge installation comes with the ``core`` and ``gui`` modules and
dependencies.

</details>

<details>
<summary>Install with pixi</summary>

To install using [pixi](https://pixi.sh):

```bash
# to add to a local project
pixi add openairclim

# to install globally
pixi global install openairclim
```

</details>

### For Developers

If you are planning on developing OpenAirClim, you will need extra files and
data not available in the releases on PyPI and conda-forge. Therefore, start by
cloning the repository from GitHub:

```bash
cd path/to/working/dir
git clone https://github.com/dlr-pa/oac.git
```

We recommend using [pixi](https://pixi.sh) for development. However, there
are many other options, depending on your setup. See our
[developer guide](https://openairclim.org/dev_guide/installation) for a
more complete overview. We recommend using Python 3.13 for development.

<details>
<summary>Install with pixi (recommended)</summary>

To install a local environment from ``pixi.lock``:

```bash
cd oac
pixi install --all
```

Run commands inside it with ``pixi run`` (e.g. ``pixi run -e dev test``), or
activate it directly with ``pixi shell -e dev``.

</details>

<details>
<summary>Install with conda</summary>

To create a conda environment and install an editable version of OpenAirClim:

```bash
cd oac
conda env create -f environment_dev.yaml
conda activate oac
pip install -e .
```

</details>

<details>
<summary>Install with venv</summary>

To create a virtual environment and install an editable version of OpenAirClim,
you will need Python 3.11+:

```bash
cd oac
python3.13 -m venv .venv
source .venv/bin/activate  # or for Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[dev]"
```

</details>

### Download repository data

OpenAirClim's response surfaces and background concentration scenarios are
published separately, in [dlr-pa/oac-data](https://github.com/dlr-pa/oac-data).
This data is **required by users and developers alike**. To download the data
to a shared cache, use:

```bash
oac-download-data
```

See the [user guide](https://openairclim.org/user_guide/installation) for
override options.

## Getting started

See the [quickstart guide](https://openairclim.org/quickstart) for a simple
OpenAirClim simulation setup.

### Graphical User Interface

OpenAirClim ships with an optional GUI for creating, loading and editing
configuration files, inspecting input data, running simulations and exploring
results. Provided the ``gui`` dependencies have been installed, it can be
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
``dir = input/``), you must be in the right working directory for OpenAirClim to
run successfully.

## Roadmap

The scheduling of major software releases and milestone planning are partially
dependent on the contractual framework with our stakeholders. For the version
history of the completed releases, see the [changelog](CHANGELOG.md). The full
development stage as currently planned is shown in the [layout](#layout).

## Contributing

Contributions are very welcome. Please read our
[contribution guidelines](CONTRIBUTING.md) to get started. For more detailed
information, see our [developer guide](https://openairclim.org/dev_guide).

## License

OpenAirClim is licensed under Apache 2.0, a copy of which can be found
[here](LICENSE).
