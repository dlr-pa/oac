Installation
============

.. image:: https://img.shields.io/github/v/tag/dlr-pa/oac?label=github&logo=github
    :target: https://github.com/dlr-pa/oac
    :alt: Latest GitHub tag
.. image:: https://img.shields.io/pypi/v/openairclim?color=orange&label=pypi&logo=pypi&logoColor=white
    :target: https://pypi.org/project/openairclim/
    :alt: Latest PyPI release
.. image:: https://img.shields.io/conda/vn/conda-forge/openairclim?label=conda-forge&logo=conda-forge&logoColor=white
    :target: https://anaconda.org/conda-forge/openairclim
    :alt: Latest conda-forge release

OpenAirClim is available from `PyPI <https://pypi.org/project/openairclim>`__,
`conda-forge <https://anaconda.org/conda-forge/openairclim>`__ and 
`from source <https://github.com/dlr-pa/oac>`__. OpenAirClim natively supports
installation with `pip <https://pip.pypa.io/en/stable/>`__,
`uv <https://docs.astral.sh/uv/>`__, `conda <https://docs.conda.io/en/latest/>`__
and `pixi <https://pixi.sh>`__. The next sections provide detailed installation
instructions for each installation method. To get a simulation running as fast
as possible, see the :doc:`../quickstart`.

.. note::

    This guide describes how to install OpenAirClim as a *user*. If you are
    planning on contributing to the development of OpenAirClim, please see
    the :doc:`developer guide <../dev_guide/installation>` instead.

.. tip::

    Not sure which installation method to pick? If you do not already have
    Python installed, or do not use it regularly, we recommend ``conda`` or
    ``pixi``: both install a suitable Python for you automatically and
    include the GUI dependencies by default. If you already work with Python
    day-to-day, ``pip`` and ``uv`` are lighter-weight options.


.. dropdown:: Installation with pip

    To install OpenAirClim from PyPI with
    `pip <https://pip.pypa.io/en/stable/>`__, you first need Python 3.11 or
    later installed (check with ``python --version``; download it from
    `python.org <https://www.python.org/downloads/>`__ if needed).

    We recommend installing into a
    `virtual environment <https://docs.python.org/3/library/venv.html>`__
    rather than system-wide, to avoid version conflicts with other Python
    projects on your machine:

    .. code-block:: bash

        python -m venv .venv
        source .venv/bin/activate  # for Windows: .venv\Scripts\activate
        pip install openairclim

    By default, this installs the ``minimal`` environment, with which you are
    able to run the core OpenAirClim model. To install the optional
    dependencies for running the GUI, run:

    .. code-block:: bash

        pip install openairclim[gui]

    Remember to re-activate the ``.venv`` environment (second line above) in
    every new terminal session before using OpenAirClim.


.. dropdown:: Installation with uv

    `uv <https://docs.astral.sh/uv/>`__ is a fast Python package and project
    manager. Unlike ``pip``, it does not require Python to already be
    installed: if you do not have Python 3.11 or later, uv can fetch one for
    you, e.g. ``uv python install 3.13``.

    uv's pip-compatible interface (used below) installs into a virtual
    environment rather than system-wide, so create and activate one first:

    .. code-block:: bash

        uv venv
        source .venv/bin/activate  # for Windows: .venv\Scripts\activate

    Then install OpenAirClim into it, the same way you would with pip:

    .. code-block:: bash

        uv pip install openairclim

    To install the optional dependencies for running the GUI, run:

    .. code-block:: bash

        uv pip install openairclim[gui]

    Commands can be run within or outside of an activate environment. If you
    are following along, the environment should still be active. If it is not,
    you can run ``source .venv/bin/activate`` (or ``.venv\Scripts\activate`` on
    Windows). The commands can be run using:

    .. code-block:: bash
        
        # inactive environment
        uv run oac-run <config-name>.toml

        # active environment
        oac-run <config-name>.toml


.. dropdown:: Installation with conda

    OpenAirClim can be installed using
    `conda <https://docs.conda.io/projects/conda/en/latest/index.html>`__ or 
    `mamba <https://mamba.readthedocs.io/en/latest/index.html>`__. We recommend
    the open-source solution
    `Miniforge <https://github.com/conda-forge/miniforge>`__, which only uses
    packages from the community `conda-forge <https://conda-forge.org/>`__
    channel, where OpenAirClim is also published. Since Miniforge is
    open-source, this option is generally available even if the use of Anaconda
    is prohibited, but we of course cannot guarantee this. Please check with
    your IT department (if applicable).

    To install OpenAirClim with conda (alternatively, replace ``conda`` with
    ``mamba``), use:

    .. code-block:: bash

        conda create --name oac python=3.13
        conda activate oac
        conda install openairclim

    The `conda-forge <https://anaconda.org/conda-forge/openairclim>`__ version
    of OpenAirClim comes with the GUI dependencies installed (unlike the
    default version installed with ``pip`` from PyPI).


.. dropdown:: Installation with pixi

    `pixi <https://pixi.sh>`__ is a fast, cross-platform package manager for
    conda-forge and PyPI packages. Since it installs from conda-forge by
    default, the GUI dependencies are included automatically, as with a plain
    ``conda`` installation above.

    To add OpenAirClim to a pixi project (run ``pixi init`` first if you do
    not yet have one), use:

    .. code-block:: bash

        pixi add openairclim

    Commands can then be run inside the project's environment with
    ``pixi run``, or by activating the environment directly:

    .. code-block:: bash

        # run OpenAirClim
        pixi run oac-run <config-name>.toml

        # or activate the environment
        pixi shell
        oac-run <config-name>.toml

    Alternatively, to install OpenAirClim as a standalone command line tool,
    available anywhere on your system without creating a project, use:

    .. code-block:: bash

        pixi global install openairclim


Verifying your installation
---------------------------

With the environment you installed OpenAirClim into activated, check that
everything worked with:

.. code-block:: bash

    python -c "import openairclim; print(openairclim.__version__)"

This should print the installed version number (e.g. ``0.18.1``) without any
errors. You can also confirm the command line tools are available with
``oac-run --help``. If you get a ``command not found`` or
``ModuleNotFoundError`` error, the environment you installed OpenAirClim into
is most likely not activated in your current terminal session (see the
relevant dropdown above).


.. _installing-the-gui:

Installing the GUI
------------------

If you installed OpenAirClim from PyPI (i.e. through ``pip install`` or
``uv pip install``), additional dependencies are required to run the GUI. Use
the following commands to install them:

.. code-block:: bash

    # for pip
    pip install openairclim[gui]

    # for uv
    uv pip install openairclim[gui]

If you have installed OpenAirClim from conda-forge using ``conda`` or ``pixi``,
**this is not required** because the installation already includes the GUI
dependencies.

See :doc:`../gui` for how to launch and use the GUI.


.. _downloading-repository-data:

Downloading repository data
---------------------------

.. image:: https://img.shields.io/badge/10.5281%2Fzenodo.22146822-blue?logo=DOI&logoColor=white&label=data
    :target: https://doi.org/10.5281/zenodo.22146822
    :alt: OpenAirClim data repository


.. note::

    OpenAirClim's simulations require response surface and background
    concentration scenarios, which are published independently of
    ``openairclim`` at `dlr-pa/oac-data <https://github.com/dlr-pa/oac-data>`__.
    From v0.18 onwards, **this data must be installed separately, irrespective
    of whether you installed OpenAirClim from source, PyPI or conda-forge**.
    It is also possible to use your own data, but it must be in the same
    format. See the :doc:`developer guide <../dev_guide/installation>` for
    more info.

To download the default data, activate the environment you installed
OpenAirClim into (e.g. ``conda activate oac``, ``source .venv/bin/activate``,
or ``pixi shell`` — whichever applies, see above) and run once:

.. code-block:: bash

    oac-download-data

By default, this fetches the data version pinned by your installed
``openairclim`` release into a shared, per-user cache directory. This makes
running OpenAirClim significantly easier for non-developers, since the data is
always stored in a predictable place. It is also beneficial for developers,
since it allows multiple different versions of the data to be present at once,
which can be used by various ``openairclim`` installations on the same machine.

OpenAirClim will look in the per-user cache directory for the response surface
and background concentration scenario data by default. To use the data in the
cache, leave ``background.dir`` and ``responses.dir`` unset in the config. To
use custom data, or data stored elsewhere on your machine, point OpenAirClim
at the relevant folder instead. See also :doc:`input`.

Useful overrides for the download function are:

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

where ``<config-name>.toml`` is replaced with the path to your own
configuration file (see :doc:`../quickstart` for a ready-made example, and
:doc:`input` for how to write your own). This is equivalent to
``python -m openairclim <config-name>.toml``. Note that if there are any
relative links in the config file (e.g. ``dir = input/``), you must be in the
right working directory for OpenAirClim to run successfully.

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
