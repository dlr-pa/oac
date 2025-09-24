Installation
============

.. note::

    In the future, it will be possible to install OpenAirClim with conda or pip directly.
    We are currently having some difficulties with dependencies that prevent us from setting this up.
    For now, to use OpenAirClim, you will need to clone the code from GitHub.

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

.. note::

    The installation with ``pip`` currently does not work due to a problem with the dependency ``cf-units``.
    We are working on a solution, see issue `#20 <https://github.com/dlr-pa/oac/issues/20>`_.

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
