Multiple emission inventories
=============================

In this example, **no** time evolution file is given, but multiple emission
inventories are given as input. OpenAirClim will interpolate between discrete
inventory years.


Imports
-------

If the openairclim package cannot be imported, make sure that you have
installed openairclim properly (see :doc:`User Guide → Installation </user_guide/installation>`).

.. jupyter-execute::
    :hide-code:

    import os

    os.chdir("docs/source/demos/03_multi_inv")

.. jupyter-execute::

    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    from openairclim.utils.create_artificial_inventories import (
        ArtificialInventoryDict,
        convert_xr_dict_to_nc,
    )
    import openairclim as oac

    xr.set_options(display_expand_attrs=False)


Input files
-----------

In order to be able to execute this example simulation, two types of input are
required.

* Configuration file ``multi_inv.toml``
* Emission inventories ``rnd_inv_20XX.nc``

Emission inventories
^^^^^^^^^^^^^^^^^^^^

* Artificial (random) emission inventories generated with OpenAirClim
* Inventory years: 2000, 2020, 2040, 2060, 2080, 2100
* Emissions increase linearly with each year (relative to 2000)

The inventories are generated in the input directory of this demo. The random
seed is fixed so that the example gives reproducible results.

.. jupyter-execute::

    np.random.seed(42)
    rnd_inv_dict = ArtificialInventoryDict(year_arr=[2000, 2020, 2040, 2060, 2080, 2100]).create()
    convert_xr_dict_to_nc(rnd_inv_dict, prefix="rnd_inv", out_path="../input/")


Simulation run
--------------

.. jupyter-execute::

    oac.run("multi_inv.toml")


Results
-------

Time series
^^^^^^^^^^^

* Emission sums
* Concentrations
* Radiative forcings
* Temperature changes

.. jupyter-execute::

    results_ds = xr.load_dataset("results/multi_inv.nc")
    display(results_ds)

.. jupyter-execute::

    # Plot Radiative Forcing and Temperature Changes

    ac = "TOTAL"
    species = ["CO2", "H2O", "cont"]
    colors = ["tab:gray", "tab:blue", "tab:cyan"]

    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    ax[0].grid(True)
    ax[1].grid(True)
    for spec, color in zip(species, colors):
        rf = results_ds[f"RF_{spec}"].sel(ac=ac) * 1000
        dt = results_ds[f"dT_{spec}"].sel(ac=ac) * 1000
        rf.plot(ax=ax[0], color=color, label=spec)
        dt.plot(ax=ax[1], color=color, label=spec)
    ax[0].set_ylabel("Radiative Forcing [mW/m²]")
    ax[1].set_ylabel("Temperature Change [mK]")
    ax[0].legend()
    ax[1].legend()
