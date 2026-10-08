Editing configs in loops
========================

It can often be useful to update config files "on the fly", for example within
a ``for`` loop. Here, we demonstrate how this can best be done using
OpenAirClim's internal tools.

Let's say that we have three emission inventories ``invA.nc``, ``invB.nc`` and
``invC.nc``, corresponding to three different aircraft designs. We would like
to run each inventory in turn with generally the same underlying settings,
modifying for example only the aircraft design variables, and then compare the
results. In this case, we could prepare a config template that is shared
between all of the inventories and then update the necessary parameters within
a python loop. 

Let's assume that the example config provided in the OpenAirClim repository is
sufficient for our purposes, and that we have saved it as ``example.toml`` in
our working directory. The three emission inventories are assumed to be in a
folder ``data/``.

.. dropdown:: Example config file

    .. code:: toml

        # Species considered
        [species]
        inv = ["CO2", "H2O", "NOx", "distance"]
        nox = "NO2"
        out = ["CO2", "H2O", "O3", "CH4", "PMO", "cont", "SWV"]

        # Emission inventories                                                    
        [inventories]
        dir = "data/"
        files = []  # updated in loop

        rel_to_base = false
        base.dir = "data/"
        base.files = []

        # Output options
        [output]
        # Switch on/off parts of the program
        run_oac = true
        run_metrics = false
        run_plots = false
        dir = ""  # updated in loop
        name = "res"
        overwrite = true
        concentrations = false

        # Time settings
        [time]
        dir = "../input/"
        range = [1940, 2020, 1]

        # Global background concentrations
        [background]
        # use shared repository data cache
        # dir = ""
        CO2.file = "co2_bg.nc"
        CO2.scenario = "SSP2-4.5"
        CH4.file = "ch4_bg.nc"
        CH4.scenario = "SSP2-4.5"
        N2O.file = "n2o_bg.nc"
        N2O.scenario = "SSP2-4.5"

        # Response options
        [responses]
        # use shared repository data cache
        # dir = ""
        CO2.response_grid = "0D"
        CO2.conc.method = "Sausen&Schumann"
        H2O.response_grid = "2D"
        H2O.rf.file = "resp_RF_H2O.nc" # AirClim response surface
        O3.response_grid = "2D"
        O3.rf.approach = "perturbation"
        O3.rf.file = "resp_RF_O3_pert.nc" # perturbation response surface (AirClim)
        CH4.response_grid = "2D"
        CH4.tau.approach = "perturbation"
        CH4.tau.file = "resp_CH4_pert.nc" # perturbation response surface (AirClim)
        cont.response_grid = "cont"
        cont.resp.file = "resp_cont_lf.nc"
        SWV.file = "ch4_for_swv_calc.nc"

        # Temperature options
        [temperature]
        method = "Boucher&Reddy"
        CO2.lambda = 1.06
        H2O.efficacy = 1.0    # expected range: [0.7, 1.3]
        O3.efficacy = 1.05    # expected range: [0.74, 1.36]
        PMO.efficacy = 1.0    # expected range: [0.7, 1.3]
        CH4.efficacy = 1.04   # expected range: [0.84, 1.26]
        cont.efficacy = 0.4  # expected range: [0.21, 0.59]
        SWV.efficacy = 1.0

        # Climate metrics options
        [metrics]
        # iterate over elements in lists types t_0 and H
        types = []   # valid climate metrics: AGTP, AGWP, AEGWP, ATR
        H = []       # Time horizon, t_final = t_0 + H - 1
        t_0 = []     # Start time for metrics calculation

        # aircraft defined in inventory
        [aircraft]
        types = ["DEFAULT"]
        DEFAULT.G_250 = 1.8  # updated in loop
        DEFAULT.PMrel = 1.0  # updated in loop
        DEFAULT.b = 50.0     # updated in loop

        # Configuration for the parametric scenario module.
        [parametric]
        enabled = false


We start by loading the necessary modules and defining the simulations and 
aircraft design parameters. Make sure that you have a working OpenAirClim
installation.

.. code:: python

    import tomllib

    import openairclim as oac
    from openairclim.utils.config_files import write_toml

    # simulation definitions
    sim_def = {
        "sim-A": {"file": "invA.nc", "G_250": 1.6, "PMrel": 1.1, "b": 50},
        "sim-B": {"file": "invB.nc", "G_250": 1.8, "PMrel": 1.0, "b": 50},
        "sim-C": {"file": "invC.nc", "G_250": 2.0, "PMrel": 0.9, "b": 50},
    }


Now, we can set up the ``for`` loop. We use ``tomllib`` (standard package in
Python from 3.11 onwards) to open the config file, then use OpenAirClim's
inbuilt TOML writer to save the config file in a human- and machine-readable
style.

.. warning::

    All file and folder paths **must be relative to the config file's location**.
    To run a config file located in a different folder, make sure to first
    change working directory to that location. This can be done within a
    python script using for example ``os.chdir("/path/to/folder")``.


.. code:: python

    for n, d in sim_def.items():

        with open("example.toml", "rb") as f:
            config = tomllib.load(f)

        # update inventory and aircraft parameters
        config["inventories"]["files"] = [d["file"]]
        config["aircraft"]["DEFAULT"]["G_250"] = d["G_250"]
        config["aircraft"]["DEFAULT"]["PMrel"] = d["PMrel"]
        config["aircraft"]["DEFAULT"]["b"] = d["b"]

        # save config file
        write_toml(config, f"{n}-cfg.toml")

        # run OpenAirClim
        oac.run(f"{n}-cfg.toml")
