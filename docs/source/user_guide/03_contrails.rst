Contrail Module
===============

This webpage describes how to run the OpenAirClim contrail module.
More detail is available in the upcoming publications Megill (2026) :cite:`megillAssessingContrailClimateImpacts2026` and Megill et al. (in prep.).
Since some results are still awaiting peer-reviewed publication, they are not yet included in the open-source version of OpenAirClim and are thus also not described here.
More information about the scientific background can be found `here <../background/contrails.html>`_.


Emission Inventories
--------------------

To calculate a contrail climate impact, the input emission inventories must include a ``distance`` (float) variable.
This corresponds with the total yearly flown distance (:math:`\mathrm{km}`).

Optionally, the emission inventories can have a variable ``ac`` (str), corresponding to the aircraft identifiers defined in the configuration file.
If this variable is defined, all identifiers (also in the base emission inventories) **must** be included in the configuration file.
If this variable is not present, OpenAirClim will use the identifier ``DEFAULT``, which must be defined in the configuration file.

We differentiate between:

- emission inventories: inventories that include the air traffic of interest; and
- base emission inventories: inventories that include the background air traffic.

For most research questions, the sum of an emission inventory and base emission inventory for any given year should correspond to the total, global air traffic.
This is important because contrails are highly non-linear: if the background air traffic is not considered, then the contrail climate impact of any given fleet can be significantly overestimated.
More detail can be found `here <../background/contrails.html>`_ and in the upcoming publications Megill (2026) :cite:`megillAssessingContrailClimateImpacts2026` and Megill et al. (in prep.).


Configuration File
------------------

To calculate a contrail climate impact, the following must be selected in the species section of the configuration file:

.. code:: toml

    [species]
    inv = ["...", "distance"]
    out = ["...", "cont"]

This tells OpenAirClim that the "distance" variable in the input emission inventories is to be used and that the contrail climate impact should be calculated.

As mentioned in the previous section, we differentiate between emission inventories and base emission inventories.
If base emission inventories are used, the variable ``rel_to_base`` must be set to ``true``, otherwise it should be set to ``false``.
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
    cont.resp.file = "resp_cont_lf.nc"       # this is required for the Megill_2025 formation method

    # cont.response_grid = "cont"            # default; should not be changed
    # cont.method = "Megill_2026"            # this method is chosen by default
    # cont.formation_method = "Megill_2025"  # this method is chosen by default

A conventional OpenAirClim configuration uses the above values.
The contrail method and formation method flags are used as placeholders -- no other methods are currently available.
The file ``resp_cont.nc`` was previously used to simulate the AirClim contrail module and will be removed in coming updates.

The contrail efficacy can be adapted using:

.. code:: toml
    
    [temperature]
    cont.efficacy = 0.59

Finally, the aircraft and three variables ``G_250``, ``b`` and ``PMrel`` must be defined.
From OpenAirClim v0.11.0 onwards, the aircraft types are active.
In principle, any aircraft identifier (except ``"TOTAL"``) can be selected, except that the first character must be a letter to comply with python requirements.
These identifiers must match with those present in the input emission inventories (see the next section).
If no identifiers are present in the emission inventories, please use ``types = ["DEFAULT"]``.

.. code:: toml

    [aircraft]
    types = ["A320", "B737"]
    # dir = "input/"
    # file = "ac_def.csv"
    A320.G_250 = 1.90
    A320.PMrel = 1.0
    A320.b = 35.0
    B737.G_250 = 1.85
    B737.PMrel = 0.8
    B737.b = 35.0

It is also possible to define these parameters in an external .csv file.
To do so, uncomment and update the ``dir`` and ``file`` values.
The .csv file must have the columns ``"ac"`` and ``"b"``.
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


Premium Functionality 
---------------------

Currently, the open-source version of OpenAirClim is only defined within the high-soot regime (:math:`EI_\mathrm{s} \geq 1.5 \times 10^{14}~\mathrm{kg}^{-1}`).
Until approximately Summer 2027, the definition of the low-soot regime is available as **premium functionality** that needs to be licensed from the German Aerospace Center (DLR).
Please contact the core development group at openairclim@dlr.de for more information on licensing.

If you have the license, make sure that the ``openairclim_premium`` module is installed in the same environment as OpenAirClim.
Whenever OpenAirClim is first loaded, you should receive a message noting that premium functionality is available.
If you do not receive this message and simulations within the low-soot regime fail or show zeros, please contact the core development group.
