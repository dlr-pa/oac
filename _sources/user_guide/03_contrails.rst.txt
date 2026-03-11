Contrail Module
===============

This webpage describes how to run the contrail module.
More information about the scientific background can be found `here <../background/contrails.html>`_.


Configuration File
------------------

.. warning::

    It is currently not possible to calculate the contrail climate impact for multiple different aircraft within the same emission inventory.
    This is the subject of ongoing work.


In the species section, the following need to be selected:

.. code:: toml

    [species]
    inv = ["...", "distance"]
    out = ["...", "cont"]

This tells OpenAirClim that the "distance" variable in the input emission inventories is to be used and that the contrail climate impact should be calculated.

The emission inventories should of course be defined as normal.
In addition, the variable ``rel_to_base`` need to be defined: if ``false``, then only the emission inventories in ``files`` are considered; if ``true``, then the base emission inventories are also used.
The base emission inventories can be used to simulate background air traffic.
For example, if the contrail climate impact of a single aircraft design is to be calculated, then the base emission inventories could be the remaining air traffic.
This is important because the contrail climate impact is highly non-linear.

So, for example, the following would be the setup if the contrail climate impact of an A320 with background air traffic were to be simulated:

.. code:: toml

    [inventories]
    dir = "path/to/inventories"
    files = [
        "a320_inv_2020.nc",
        "...",
        "a320_inv_2045.nc",
        "a320_inv_2050.nc",
    ]
    rel_to_base = true
    base.dir = "path/to/inventories"
    base.files = [
        "bg_inv_2020.nc",
        "bg_inv_2050.nc",
    ]

The requirement is that the emission inventories and base emission inventories must align at the first and last year, in this case in 2020 and 2050.
OpenAirClim features a built-in interpolator for base emission inventories.
The above configuration will thus work without issue - OpenAirClim will interpolate the base emission inventories onto the years defined by the A320 emission inventories.

In the response options, the following values are relevant for the contrail module:

.. code:: toml

    [responses]
    dir = "path/to/responses"
    cont.response_grid = "cont"  # should not be changed
    cont.resp.file = "resp_cont_lf.nc"
    # cont.method = "Megill_2025"  # this method is chosen by default

A conventional OpenAirClim configuration uses the above values.
The file ``resp_cont.nc`` can be used in conjuction with ``cont.method="AirClim"`` to simulate the AirClim contrail module for testing.
However, this is not generally recommended outside of unit testing, because 1) the simulated AirClim module is restrictive in its input and 2) the OpenAirClim contrail module includes many improvements.

The contrail efficacy can be adapted using:

.. code:: toml
    
    [temperature]
    cont.efficacy = 0.59

Finally, the aircraft and three variables ``G_250``, ``eff_fac`` and ``PMrel`` must be defined.
From OpenAirClim v0.11.0 onwards, the aircraft types are active.
In principle, any aircraft identifier (except ``"TOTAL"``) can be selected, except that the first character must be a letter to comply with python.
These identifiers must match with those present in the input emission inventories (see the next section).
If no identifiers are present in the emission inventories, please use ``types = ["DEFAULT"]``.

.. warning::

    It is currently not possible to calculate the contrail climate impact of multiple different aircraft within the same emission inventory.
    If multiple different types are given whilst simultaneously including ``species.out = ["...", "cont"]``, OpenAirClim will produce an error.

.. code:: toml

    [aircraft]
    types = ["A320", "B737"]
    # dir = "input/"
    # file = "ac_def.csv"
    A320.G_250 = 1.90
    A320.eff_fac = 1.1
    A320.PMrel = 1.0
    B737.G_250 = 1.85
    B737.eff_fac = 1.05
    B737.PMrel = 0.8

It is also possible to define these parameters in an external .csv file.
To do so, uncomment and update the ``dir`` and ``file`` values.
The .csv file must have the columns ``"ac"`` and ``"eff_fac"``.
Additionally, the file must either have the columns ``"G_250"`` and ``"PMrel"``, or additional information such that OpenAirClim can calculate these values online.
For ``G_250``, the following columns must be provided:

- ``"SAC_eq"``: which equation to use to calculate the SAC slope, choice of ``"CON"``, ``"HYB"``, ``"H2C"`` and ``"H2FC"``. See :cite:`megillInvestigatingLimitingAircraftdesigndependent2025` for more details;
- ``"Q_h"``: Lower Heating Value of the fuel [J/kg] for ``"CON"`` (~43.6 MJ/kg), ``"HYB"``, ``"H2C"`` (~120 MJ/kg); formation enthalpy of water vapour [J/mol] for ``"H2FC"``;
- ``"eta"``: Overall propulsion efficiency of the liquid fuel system (for all except ``"H2FC"``);
- ``"eta_elec"``: Efficiency of the electric/fuel cell system (for ``"HYB"`` and ``"H2FC"``);
- ``"EIH2O"``: Emission index of water vapour [kg/kg] (for all except ``"H2FC"``);
- ``"R"``: Degree of hybridisation (for ``"HYB"``). R=1 is pure liquid fuel operation; R=0 pure electric operation.

For ``"PMrel"``, a ``"PM"`` column is required, specifying the nvPM (soot) number emission index.
The relative PM emissions are taken with respect to 1.5e15 #/kg.

If the aircraft characteristics are simultaneously defined in the config and in the .csv file, *the config data will not be overwritten*.
OpenAirClim will warn you if this is the case.


Emission Inventories
--------------------

To calculate a contrail climate impact, the input emission inventories must include a ``distance`` (float) variable.
This corresponds with the total yearly flown distance (km).

Optionally, the emission inventories can have a variable ``ac`` (str), corresponding to the aircraft identifiers defined in the configuration file.
If this variable is defined, all identifiers (also in the base emission inventories) **must** be included in the configuration file.
If this variable is not present, OpenAirClim will use the identifier ``DEFAULT``, which must be defined in the configuration file.
