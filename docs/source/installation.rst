Installation
============

If you build OpenAirClim from source, you first have to clone the `repository <https://github.com/dlr-pa/oac>`_.
The most convenient way of doing this is by using the following `Git <https://git-scm.com/>`_ command:

.. code-block:: bash

    git clone https://github.com/dlr-pa/oac.git

Once the repository has been cloned, there are two options to install the necessary packages.


Installation using conda
------------------------

If you choose to use conda, the `conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_ or `mamba <https://mamba.readthedocs.io/en/latest/index.html>`_ package manager must first be installed.
We recommend the open-source solution `Miniforge <https://github.com/conda-forge/miniforge>`_, which only uses packages from the community `conda-forge <https://conda-forge.org/>`_ channel.
Since it is open-source, this option is generally available even if the use of Anaconda is prohibited, but we of course cannot guarantee this.
Please check with your IT department (if applicable).

The source code includes configuration files ``environment_xxx.yaml`` that enable the installation of a conda environment with all required dependencies.
This installation method is suitable for working across platforms.
Use the ``dev`` file if you are planning on making changes to the code or contributing to the development of OpenAirClim, otherwise use ``minimal``.
Change directory to the root folder of the downloaded source and create a conda environment and activate it:

.. code-block:: bash

    cd oac
    conda env create -f environment_xxx.yaml
    conda activate <env>

Replace ``xxx`` with the relevant file and ``<env>`` with the correct name of the installed conda environment, e.g ``oac`` or ``oac_minimal``.

Finally, to install the openairclim package system-wide on your computer, execute one of the following commands within the activated conda environment.
This last installation step isn't necessary if the user has otherwise added the path to the oac source folder to ``PYTHONPATH``.

.. code-block:: bash

    pip install .

or

.. code-block:: bash

    pip install -e .

The ``-e`` flag treats the openairclim package as an editable install, allowing you to make changes to the source code and see those changes reflected immediately.
The latter command is recommended for developers.

After installing the conda environment and required dependencies, proceed with the steps described in :doc:`quickstart`.


Installation using pip
----------------------

The prerequisite for this installation method is have installed a python version >= 3.4.
Then, the installer ``pip`` is included by default. 
In your console, change directory to the OpenAirClim root folder and execute the following command:

.. code-block:: bash

    pip install .

To install OpenAirClim in *editable mode*, use the ``-e`` flag:

.. code-block:: bash

    pip install -e .

If you are planning on making changes to the code or contributing to the development of OpenAirClim, extra packages are required.
To install these, use (with or without the ``-e`` flag):

.. code-block:: bash

    pip install ".[dev]"

After installing the packages, proceed with the steps described in :doc:`quickstart`.


.. _installing-the-gui:

Installing the GUI
-------------------

OpenAirClim ships with an optional graphical user interface (GUI), which requires
some additional dependencies on top of a standard installation. Layer these onto
an existing conda environment with:

.. code-block:: bash

    conda env update -f environment_gui.yaml -n <env-name>

or, for a pip installation, install the ``gui`` extra instead:

.. code-block:: bash

    pip install ".[gui]"

See :doc:`gui` for how to launch and use the GUI once installed.


Downloading repository data
----------------------------

OpenAirClim's simulations are driven by response surface data and
background concentration scenarios (the "repository data"), published
independently of the ``openairclim`` package itself, at
`dlr-pa/oac-repository <https://github.com/dlr-pa/oac-repository>`_. This
applies whether you installed from source, via pip, or via conda — cloning
the ``oac`` repository no longer includes the repository data.

Download it once with:

.. code-block:: bash

    oac-download-data

By default, this fetches the data version pinned by your installed
``openairclim`` release into a shared, per-user cache directory (so multiple
``openairclim`` installations/environments on the same machine reuse a
single copy rather than duplicating it). A config file that leaves
``background.dir``/``responses.dir`` unset automatically resolves to this
same cache directory at run time - see :doc:`user_guide/01_input`.

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
