Installation
============

OpenAirClim is currently available from PyPI at
https://pypi.org/project/openairclim or from source at
https://github.com/dlr-pa/oac. Later OpenAirClim versions will also be
available from conda-forge (work in progress).


Installation with pip
---------------------

To install OpenAirClim from PyPI with `pip <https://pip.pypa.io/en/stable/>`__
(Python 3.11 or later required):

.. code-block:: bash

    pip install openairclim

By default, this installs the ``minimal`` environment, with which you are able
to run the core OpenAirClim model. To install the optional dependencies for
running the `GUI <gui>`_, building the OpenAirClim documentation or running the
software tests, use the following command. ``[dev]`` provides a complete
environment with all optional dependencies.

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
`downloading the repository data <download-data>`_.


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

    git clone https://github.com/dlr-pa/oac-git
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
`downloading the repository data <download-data>`_.
