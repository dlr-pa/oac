Normalization
=============

In this example, the time evolution of type **normalization** is demonstrated.
A historic emission scenario is simulated with OpenAirClim. This demo was tested with oac v0.11.1.
Adapt the paths defined in this demo as well as the paths defined in the configuration file before executing this demo by yourself. 

Imports
-------

In some cases, it might be necessary to define `sys.path`, such that the interpreter finds the openairclim package.
Alternatively, the environment variable, e.g. `PYTHONPATH`, can be configured.

.. jupyter-execute::

    import xarray as xr
    import matplotlib.pyplot as plt
    import openairclim as oac

    xr.set_options(display_expand_attrs=False)

Input files
-----------

In order to be able to execute this example simulation, three input files are required.

* Configuration file `historic.toml`
* Emission inventory `ELK_all-subsectors_2019_flat.nc`
* Time evolution file for fuel normalization `time_norm_historic_SSP.nc`

Emission inventory
^^^^^^^^^^^^^^^^^^

* Source: DLR Project EmissionsLandKarte (`ELK`_)
* Inventory year: 2019
* Converted into format suitable for OpenAirClim


.. _ELK: https://elkis.dlr.de/

.. jupyter-execute::

    inv = xr.load_dataset("source/demos/input/ELK_all-subsectors_2019_flat.nc")
    display(inv)

Time evolution
^^^^^^^^^^^^^^

* Time evolution with **normalization** of fuel use
* Time period: 1920 - 2019

.. jupyter-execute::

    evo = xr.load_dataset("source/demos/input/time_norm_historic_SSP.nc")
    display(evo)

    fig, ax = plt.subplots()
    evo.fuel.plot(ax=ax)
    ax.grid(True)


Simulation run
--------------

.. jupyter-execute::

    oac.run("source/demos/01_norm/historic.toml")


Results
-------

Time series
^^^^^^^^^^^

* Emission sums
* Concentrations
* Radiative forcings
* Temperature changes

.. jupyter-execute::

    results_ds = xr.load_dataset("source/demos/01_norm/results/historic.nc")
    display(results_ds)

.. jupyter-execute::

    # Plot Radiative Forcing and Temperature Changes

    ac = "TOTAL"
    rf_cont = results_ds.RF_cont.sel(ac=ac) * 1000
    rf_co2 = results_ds.RF_CO2.sel(ac=ac) * 1000
    rf_h2o = results_ds.RF_H2O.sel(ac=ac) * 1000
    dt_cont = results_ds.dT_cont.sel(ac=ac) * 1000
    dt_co2 = results_ds.dT_CO2.sel(ac=ac) * 1000
    dt_h2o = results_ds.dT_H2O.sel(ac=ac) * 1000

    fig, ax = plt.subplots(ncols=2, figsize=(10,5))
    ax[0].grid(True)
    ax[1].grid(True)
    rf_cont.plot(ax=ax[0], color="deepskyblue", label="cont")
    rf_co2.plot(ax=ax[0], color="k", label="CO2")
    rf_h2o.plot(ax=ax[0], color="steelblue", label="H2O")
    dt_cont.plot(ax=ax[1], color="deepskyblue", label="cont")
    dt_co2.plot(ax=ax[1], color="k", label="CO2")
    dt_h2o.plot(ax=ax[1], color="steelblue", label="H2O")
    ax[0].set_ylabel("Radiative Forcing [mW/m²]")
    ax[1].set_ylabel("Temperature Change [mK]")
    ax[0].legend()
    ax[1].legend()

Climate metrics
^^^^^^^^^^^^^^^

* Absolute Global Temperature Potential (AGTP)
* Absolute Global Warming Potential (AGWP)
* Average Temperature Response (ATR)

.. jupyter-execute::

    metrics_ds = xr.load_dataset("source/demos/01_norm/results/historic_metrics.nc")
    display(metrics_ds)
