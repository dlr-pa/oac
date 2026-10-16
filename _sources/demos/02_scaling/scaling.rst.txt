Scaling
=======

In this example, the time evolution of type **scaling** is demonstrated. In the
scenario, the emissions increase linearly from the year 2019 to the year 2039.
The emissions in 2039 are set to be twice as much as in 2019.


Imports
-------

If the openairclim package cannot be imported, make sure that you have
installed openairclim properly (see :doc:`User Guide → Installation </user_guide/installation>`).

.. jupyter-execute::
    :hide-code:

    import os

    os.chdir("docs/source/demos/02_scaling")

.. jupyter-execute::

    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    import openairclim as oac
    from openairclim.utils.create_artificial_inventories import (
        ArtificialInventory,
        convert_xr_dict_to_nc,
    )

    xr.set_options(display_expand_attrs=False)


Input files
-----------

In order to be able to execute this example simulation, three types of input
are required.

* Configuration file ``scaling.toml``
* Emission inventories

  * ``art_inv_2019.nc``
  * ``art_inv_2039.nc``

* Time evolution file for scaling: ``time_scaling_linear_2019-2039.nc``

Emission inventories
^^^^^^^^^^^^^^^^^^^^

* Artificial (random) emission inventory generated with OpenAirClim
* Inventory years

  * 2019 (original)
  * 2039 (identical to 2019 inventory, only year changed)

The inventories are generated in the input directory of this demo. The random
seed is fixed so that the example gives reproducible results.

.. jupyter-execute::

    np.random.seed(42)
    art_inv = ArtificialInventory(year=2019).create()
    art_inv_dict = {
        2019: art_inv,
        2039: art_inv.assign_attrs(Inventory_Year=2039),
    }
    convert_xr_dict_to_nc(art_inv_dict, prefix="art_inv", out_path="../input/")

Time evolution
^^^^^^^^^^^^^^

* Time evolution with **scaling** of emissions
* Time period: 2000 - 2050
* Linear ramp-up between years 2019 and 2039

.. jupyter-execute::

    evo = xr.load_dataset("../input/time_scaling_linear_2019-2039.nc")
    display(evo)

    fig, ax = plt.subplots()
    evo.scaling.plot(ax=ax)
    ax.grid(True)


Simulation run
--------------

.. jupyter-execute::

    oac.run("scaling.toml")


Results
-------

Time series
^^^^^^^^^^^

* Emission sums
* Concentrations
* Radiative forcings
* Temperature changes

.. jupyter-execute::

    results_ds = xr.load_dataset("results/scaling.nc")
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
