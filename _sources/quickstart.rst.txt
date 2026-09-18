Quickstart
==========

This page gets you from a fresh Python environment to your first OpenAirClim
simulation in a few minutes, using the example configuration and a set of
randomly generated emission inventories. For installation alternatives,
custom input data and everything else, see the :doc:`user_guide`.


Install OpenAirClim
--------------------

Install OpenAirClim from PyPI with `pip <https://pip.pypa.io/en/stable/>`__
(Python 3.11 or later required). We recommend installing into a
`virtual environment <https://docs.python.org/3/library/venv.html>`__ rather
than system-wide:

.. code-block:: bash

    python -m venv .venv
    source .venv/bin/activate  # for Windows: .venv\Scripts\activate
    pip install openairclim

or, with `conda <https://docs.conda.io/en/latest/>`__:

.. code-block:: bash

  conda create --name oac python=3.13
  conda activate oac
  conda install openairclim

This installs the ``minimal`` environment, sufficient to run the core model
(the conda-forge version above also includes the GUI dependencies). See
:doc:`user_guide/installation` for uv/pixi installs and optional extras.


Download the repository data
-----------------------------

OpenAirClim's simulations require response surface and background
concentration data, published independently of ``openairclim`` at
`dlr-pa/oac-data <https://github.com/dlr-pa/oac-data>`__. With the
environment activated (as above), fetch the data version matching your
installed release into a shared, per-user cache with:

.. code-block:: bash

    oac-download-data

This only needs to be run once per machine.


Get the example
-----------------

Download :download:`example.toml <../../example/example.toml>` into a new
working directory, then generate the emission inventories it references:

.. code-block:: bash

    cd path/to/working/directory
    oac-create-artificial-inventories --output-dir input/


Run OpenAirClim
-----------------

.. code-block:: bash

    oac-run example.toml

(equivalently, ``python -m openairclim example.toml``, or
``oac.run("example.toml")`` from within Python). Note that if there are any
relative links in the config file (e.g. ``dir = "input/"``), you must be in
the right working directory, in this case the directory created above, for
OpenAirClim to run successfully.


Look at the results
---------------------

The ``example.toml`` config file defines the output directory as
``results/``. In this folder, you can find:

- ``example.nc`` - time series of emissions, concentrations, radiative
  forcing and temperature change for each species
- ``example_metrics.nc`` - the requested climate metrics (AGWP, ATR, AGTP)
- one PNG plot per species (e.g. ``example_CO2.png``), since
  ``output.run_plots = true`` in the example configuration


Next steps
-----------

- Explore configurations and results interactively with the :doc:`gui`.
- The :doc:`user_guide` covers the configuration file in full, building your
  own emission inventories and time evolutions, and the contrail module.
- The :doc:`demos` walk through worked examples in more depth.
