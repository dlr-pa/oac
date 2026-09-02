Quickstart
==========

This page gets you from a fresh Python environment to your first OpenAirClim
simulation in a few minutes, using the example configuration and (randomly
generated) emission inventories bundled with the repository. For installation
alternatives, custom input data and everything else, see the
:doc:`user_guide`.


Install OpenAirClim
--------------------

Install OpenAirClim from PyPI with `pip <https://pip.pypa.io/en/stable/>`__
(Python 3.11 or later required):

.. code-block:: bash

    pip install openairclim

This installs the ``minimal`` environment, sufficient to run the core model.
See :doc:`user_guide/installation` for conda/source installs and optional
extras (GUI, docs, tests).


Download the repository data
-----------------------------

OpenAirClim's simulations require response surface and background
concentration data, published independently of ``openairclim`` at
`dlr-pa/oac-data <https://github.com/dlr-pa/oac-data>`__. Fetch the data
version matching your installed release into a shared, per-user cache with:

.. code-block:: bash

    oac-download-data

This only needs to be run once per machine.


Get the example
-----------------

Clone the repository to get the bundled example configuration and emission
inventories:

.. code-block:: bash

    git clone https://github.com/dlr-pa/oac.git
    cd oac/example


Run OpenAirClim
-----------------

.. code-block:: bash

    oac-run example.toml

(equivalently, ``python -m openairclim example.toml``, or
``oac.run("example.toml")`` from within Python). Note that if there are any
relative links in the config file (e.g. ``dir = "input/"``), you must be in
the right working directory, in this case ``oac/example``, for OpenAirClim to
run successfully.


Look at the results
---------------------

The ``example.toml`` config file defines the output directoy as ``results/``.
In this folder, you can find:

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
