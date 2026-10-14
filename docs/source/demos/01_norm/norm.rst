Normalization
=============

In this example, the time evolution of type **normalization** is demonstrated.
A historic emission scenario is simulated with OpenAirClim.


Imports
-------

If the openairclim package cannot be imported, make sure that you have
installed openairclim properly (see :doc:`User Guide → Installation </user_guide/installation>`).

.. jupyter-execute::
    :hide-code:

    import os

    os.chdir("docs/source/demos/01_norm")

.. jupyter-execute::

    import xarray as xr
    import matplotlib.pyplot as plt
    import openairclim as oac

    xr.set_options(display_expand_attrs=False)


Input files
-----------

In order to be able to execute this example simulation, three types of input
are required.

* Configuration file ``norm.toml``
* Emission inventory ``ELK_aviation_2019_res5deg_flat.nc``
* Time evolution file for fuel normalization ``time_norm_historic_SSP.nc``

Emission inventory
^^^^^^^^^^^^^^^^^^

* Source: DLR Project EmissionsLandKarte (`ELK <https://elkis.dlr.de/>`_)
* Resolution down-sampled to 5 deg resolution
* Converted into format suitable for OpenAirClim
* Inventory year: 2019

.. jupyter-execute::

    inv = xr.load_dataset("../input/ELK_aviation_2019_res5deg_flat.nc")
    display(inv)

Time evolution
^^^^^^^^^^^^^^

* Time evolution with **normalization** of fuel use
* Time period: 1920 - 2019

.. jupyter-execute::

    evo = xr.load_dataset("../input/time_norm_historic_SSP.nc")
    display(evo)

    fig, ax = plt.subplots()
    evo.fuel.plot(ax=ax)
    ax.grid(True)


Simulation run
--------------

.. jupyter-execute::

    oac.run("norm.toml")


Results
-------

Time series
^^^^^^^^^^^

* Emission sums
* Concentrations
* Radiative forcings
* Temperature changes

.. jupyter-execute::

    results_ds = xr.load_dataset("results/norm.nc")
    display(results_ds)

.. jupyter-execute::

    # Plot Radiative Forcing and Temperature Changes

    ac = "TOTAL"
    # O3, CH4, PMO and SWV are summed into one NOx term
    nox_species = ["O3", "CH4", "PMO", "SWV"]
    rf_nox = sum(results_ds[f"RF_{s}"].sel(ac=ac) for s in nox_species) * 1000
    dt_nox = sum(results_ds[f"dT_{s}"].sel(ac=ac) for s in nox_species) * 1000

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
    rf_nox.plot(ax=ax[0], color="tab:green", linestyle="--", label="NOx")
    dt_nox.plot(ax=ax[1], color="tab:green", linestyle="--", label="NOx")
    ax[0].set_ylabel("Radiative Forcing [mW/m²]")
    ax[1].set_ylabel("Temperature Change [mK]")
    ax[0].legend()
    ax[1].legend()

Climate metrics
^^^^^^^^^^^^^^^

* Absolute Global Temperature Potential (AGTP)
* Absolute Global Warming Potential (AGWP)
* Absolute Efficacy-weighted Global Warming Potential (AEGWP)
* Average Temperature Response (ATR)

.. jupyter-execute::

    metrics_ds = xr.load_dataset("results/norm_metrics.nc")
    display(metrics_ds)
