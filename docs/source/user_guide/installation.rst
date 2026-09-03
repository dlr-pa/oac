Installation
============

OpenAirClim is currently available from PyPI at
https://pypi.org/project/openairclim or from source at
https://github.com/dlr-pa/oac. Later OpenAirClim versions will also be
available from conda-forge (work in progress).

For the fastest way to get a first simulation running, see the
:doc:`../quickstart`.


Installation with pip
---------------------

To install OpenAirClim from PyPI with `pip <https://pip.pypa.io/en/stable/>`__
(Python 3.11 or later required):

.. code-block:: bash

    pip install openairclim

By default, this installs the ``minimal`` environment, with which you are able
to run the core OpenAirClim model. To install the optional dependencies for
running the GUI (see `Installing the GUI`_ below), building the OpenAirClim
documentation or running the software tests, use the following command.
``[dev]`` provides a complete environment with all optional dependencies.

.. code-block:: bash

    # install with optional dependencies (gui, docs, test, dev)
    pip install openairclim[dev]

To install the latest development version directly from Github, you have two
options:

.. code-block:: bash

    # with git
    git clone https://github.com/dlr-pa/oac.git

    # with pip
    pip install git+https://github.com/dlr-pa/oac.git

After successfully installing OpenAirClim, proceed by
`Downloading repository data`_.


Installation with uv
---------------------

`uv <https://docs.astral.sh/uv/>`__ is a fast Python package and project
manager. As a drop-in replacement for pip, it can install the published
package the same way:

.. code-block:: bash

    uv pip install openairclim

    # install with optional dependencies (gui, docs, test, dev)
    uv pip install openairclim[dev]

For a full development environment, clone the repository and let uv manage a
project-local virtual environment from the committed ``uv.lock``:

.. code-block:: bash

    git clone https://github.com/dlr-pa/oac.git
    cd oac
    uv sync --extra dev

This creates a ``.venv`` pinned to the interpreter in ``.python-version``.
Run commands inside it with ``uv run`` (e.g. ``uv run oac-run
<config-name>.toml``), or activate it directly with ``source
.venv/bin/activate`` (Linux and MacOS) or ``source .venv/Scripts/activate``
(Windows).

After successfully installing OpenAirClim, proceed by
`Downloading repository data`_.


Installation using conda
------------------------

We are currently working on making OpenAirClim available at
`conda-forge <https://conda-forge.org/>`__. For the time being, the only
installation possibility with conda is through cloning the repository from
GitHub. First make sure that either the 
`conda <https://docs.conda.io/projects/conda/en/latest/index.html>`__ or
`mamba <https://mamba.readthedocs.io/en/latest/index.html>`__ package manager
is installed. We recommend the open-source solution
`Miniforge <https://github.com/conda-forge/miniforge>`__, which only uses
packages from the community `conda-forge <https://conda-forge.org/>`__ channel.
Since it is open-source, this option is generally available even if the use of
Anaconda is prohibited, but we of course cannot guarantee this. Please check
with your IT department (if applicable).

To install OpenAirClim with conda, use:

.. code-block:: bash

    git clone https://github.com/dlr-pa/oac.git
    cd oac
    conda env create -f environment_xxx.yaml
    conda activate <env>

    # optional: install GUI dependencies
    conda env update -f environment_gui.yaml -n <env>

Replace ``xxx`` with either ``minimal`` or ``dev`` (full installation) and
``<env>`` with the correct name of the conda environment (e.g. ``oac`` or
``oac_minimal``). To install OpenAirClim within your newly created conda
environment, use:

.. code-block:: bash

    pip install .

    # editable installation
    pip install -e .

The ``-e`` flag treats the openairclim package as an editable install, allowing
you to make changes to the source code and see those changes reflected
immediately. The latter command is recommended for developers.

After successfully installing OpenAirClim, proceed by
`Downloading repository data`_.


.. _installing-the-gui:

Installing the GUI
-------------------

The GUI requires additional dependencies on top of a standard OpenAirClim
installation:

.. code-block:: bash

    # with pip
    pip install openairclim[gui]

    # with uv
    uv pip install openairclim[gui]

    # with conda
    conda env update -f environment_gui.yaml -n <env>

See :doc:`../gui` for how to launch and use it.


Downloading repository data
----------------------------

.. note::

    OpenAirClim's simulations require response surface and background
    concentration scenarios, which are published independently of
    ``openairclim`` at `dlr-pa/oac-data <https://github.com/dlr-pa/oac-data>`__.
    From v0.18 onwards, this data must be installed separately, irrespective of
    whether you installed OpenAirClim from source, through pip or through
    conda. It is also possible to use your own data, but it must be in the same
    format.

To download the default data, activate the python environment that includes
``openairclim`` and run once:

.. code-block:: bash

    oac-download-data

By default, this fetches the data version pinned by your installed
``openairclim`` release into a shared, per-user cache directory. This means
that multiple ``openairclim`` installations on the same machine reuse a single
copy of the data, useful for developers. It also allows for multiple different
versions of the data to be present at once.

OpenAirClim will look in the per-user cache directory for the response surface
and background concentration scenario data by default. To use the data in the
cache, leave ``background.dir`` and ``responses.dir`` unset in the config. To
use custom data, or data stored elsewhere on your machine, point OpenAirClim
at the relevant folder instead. See also :doc:`input`.

Useful overrides:

.. code-block:: bash

    # fetch a specific data version, or a specific Zenodo record/DOI
    oac-download-data --version 1.2.0
    oac-download-data --record 10.5281/zenodo.1234567

    # download into a custom, one-off location (only affects this download)
    oac-download-data --output-dir /path/to/data

    # override the default cache location itself, so both downloads and
    # config resolution consistently use it
    export OPENAIRCLIM_DATA_DIR=/path/to/data
    oac-download-data

Run ``oac-download-data --help`` for the full list of options.


Running OpenAirClim
--------------------

After installation, OpenAirClim can be run from the command line using:

.. code-block:: bash

    cd path/to/working/directory
    oac-run <config-name>.toml

(equivalently, ``python -m openairclim <config-name>.toml``). Note that if
there are any relative links in the config file (e.g. ``dir = input/``), you
must be in the right working directory for OpenAirClim to run successfully.

OpenAirClim can also be imported and used in Python programs:

.. code-block:: python

    # to run OpenAirClim
    import openairclim as oac
    oac.run("<config-name>.toml")

    # to run the GUI
    from openairclim.gui import launch
    launch(config_path="<config-name>.toml")

    # or, to use specific functions
    from openairclim.core.calc_dt import calc_dtemp_br2008
    calc_dtemp_br2008(config, "CO2", rf_arr)

Alternatively, use the :doc:`../gui` to create, edit and run configurations
without writing TOML or Python directly. The GUI can be launched using the
command ``oac-gui``, optionally pre-loading a config file with the ``--config``
flag:

.. code-block:: bash

    oac-gui --config <config-name>.toml
