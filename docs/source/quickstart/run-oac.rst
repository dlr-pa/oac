Run OpenAirClim
===============

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

Alternatively, use the :doc:`gui` to create, edit and run configurations
without writing TOML or Python directly.

Refer to the `example <https://github.com/dlr-pa/oac/tree/main/example>`_
folder within the repository for a minimal example and the :doc:`demos` given
on this website.
