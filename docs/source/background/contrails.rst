Contrail Module
===============

Here, the OpenAirClim Contrail Module (oac.cont) is described.
The first section describes the methodology, explaining the scientific basis to the submodules.
Section 2 provides an overview of some important, general open questions.
Section 3 describes the Global Circulation Model (GCM) simulations upon which the contrail module is based.
Finally, Section 4 discusses the current limitations, future improvements and planning.

To understand how to run the contrail module within OpenAirClim, please refer to the `user guide <../user_guide/contrails.html>`_.


.. warning::

    It is currently not possible to calculate the contrail climate impact for multiple different aircraft.
    This is the subject of ongoing work, in particular on attribution.
    If reference is made to a specific fleet :math:`n` in this text, please assume that it refers to the single aircraft design that is inputted.


OpenAirClim Contrail Module Methodology
---------------------------------------

The objective of the OpenAirClim Contrail Module (oac-cont) is to calculate the changes in the contrail-induced global yearly average radiative forcing and temperature due to an input emission inventory and scenario.
A simple representation of the methodology is:

.. mermaid::

    flowchart LR
        inv["Emission inventory"]
        inv --> form["Contrail formation"]
        form --> cccov["Contrail-cirrus coverage"]
        cccov --> RF["Contrail RF"]
        RF --> dT["Contrail dT"]

This is similar to the methodology used by AirClim :cite:`greweAirClimEfficientTool2008, dahlmannMethodeZurEffizienten2011, dahlmannCanWeReliably2016`.
The biggest difference between the AirClim and OpenAirClim methods is that OpenAirClim can accept multiple different aircraft types within the same grid box.
This is complex because the climate impact from contrails is non-linear.



Emission Inventory and Input Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To calculate a contrail climate impact, the yearly distance flown by a given aircraft at a given location (latitude, longitude, altitude) is required within the input emission inventory.
Since oac.cont is defined on a pre-calculated grid, the first step is to identify which flown distances fall within each grid box :math:`(i,j,k)` (latitude, longitude, pressure level).
To reduce computational time in further steps, the emission inventories of identical aircraft-engine combinations (e.g. A320neo with LEAP-1A) are combined as much as possible.
The total yearly distance flown is thus available per distinct fleet :math:`n`.

For each distinct fleet, further information is required in the input configuration file, namely:

- ``G_250``: The Schmidt-Appleman mixing line slope `G` [Pa/K] at 250 hPa
- ``eff_fac``: Overall propulsion efficiency :math:`\eta` compared to the reference (0.333) - planned to be replaced by the actual :math:`\eta`
- ``PMrel``: Particulate emissions compared to the reference (1e15)



Contrail Formation
^^^^^^^^^^^^^^^^^^

Contrail formation is classified by the Schmidt-Appleman Criterion (SAC, :cite:`schumannConditionsContrailFormation1996`) and by ice supersaturation.
As in AirClim 2.0, in oac.cont the concept of `Contrail Flight Distance Density (CFDD)` is used, which can be thought of as the flown distance weighted by the probability of persistent contrail formation.
Currently implemented is,

.. math::

    CFDD(i,j) = \sum_k \left(\frac{\text{Flown distance}(i,j,k)}{\text{Grid area}(i,j)} \cdot p_\text{pcf}(G, i,j,k) \right) 

where:

- :math:`\text{Flown distance}(i,j,k)` is the distance flown in each grid box (input from inventory);
- :math:`\text{Grid area}(i,j)` is the area of each grid box as viewed from above (lat-lon);
- :math:`G` is the slope of the Schmidt-Appleman mixing line [Pa/K]; and
- :math:`p_\text{pcf}` is the probability of a persistent contrail forming for a fleet with slope :math:`G`.

In AirClim, the :math:`p_\text{pcf}` (then called :math:`p_\text{SAC}`) was calculated for three different :math:`G` corresponding to aircraft powered by conventional kerosene, LNG and LH2.
In OpenAirClim, a new, continuous :math:`p_\text{pcf}` was developed using ERA5 data as part of a study into the limiting factors of persistent contrail formation :cite:`megillInvestigatingLimitingAircraftdesigndependent2025`.
The continuous function is based on the sum of two modified logistic functions and valid for :math:`0.48~\text{Pa/K} \leq G`. 




Contrail Coverage
^^^^^^^^^^^^^^^^^

.. warning::

    The coverage submodule is in development.

Currently implemented is the AirClim method, which uses the 2D CFDD to calculate a 2D contrail cirrus coverage :math:`cccov`,

.. math::

    cccov(i,j) = 0.128 \cdot ISS(i,j) \cdot \arctan \left( 97.7 \cdot \frac{CFFD(i,j)}{ISS(i,j)} \right) 

where :math:`a = 0.128` and :math:`b = 97.7` are fitted parameters and :math:`ISS` is the proportion of the grid cell :math:`(i,j,k)` that is supersaturated with respect to ice.
The :math:`cccov` is further multiplied by three weighting functions :cite:`huettenhoferParametrisierungKondensstreifenzirrenFuer2013, greweAssessingClimateImpact2017`:

.. math::

    w_1 = 0.863 \cdot \cos\left(lat \cdot \frac{\pi}{50}\right)

.. math::
    
    w_2 = 1.0 + 15.0 \cdot | 0.045 \cdot \cos\left(0.045 \cdot lat \right) + 0.045 | \cdot (\eta_{fac} - 1.0)


.. math::

    w_3 = 1.0 + 0.24 \cdot \cos\left(lat \cdot \frac{\pi}{23}\right)

Finally, a global contrail cirrus coverage :math:`\overline{cccov}` is obtained by area-weighting the :math:`cccov`.



Development of new coverage calculations are ongoing.
The response for a single fleet :math:`n` will most likely take the following form,

.. math::

    cccov(n,i,j,k) = a \cdot ISS(i,j,k) \cdot \arctan \left(b \cdot \frac{CFDD(n,i,j,k)}{ISS(i,j,k)}\right)

The challenge then becomes how to combine the coverages together, since contrail cirrus coverage is highly non-linear due to saturation effects.
Furthermore, contrails formed by different aircraft, fuels and propulsion technologies do not produce the same coverage and have differing radiative forcing impacts.
For example, the values :math:`a` and :math:`b` may differ depending on the aircraft type, fuel or propulsion technology.




Radiative Forcing
^^^^^^^^^^^^^^^^^

.. warning::

    The radiative forcing submodule is in development.

Currently, the AirClim method is implemented,

.. math::

    RF = 14.9 \cdot \overline{cccov} \cdot PM_{fac}

where:

- :math:`\overline{cccov}` is the global contrail cirrus coverage
- :math:`PM_{fac} = 0.92 \cdot \arctan \left(1.902 \cdot PM_{rel} ^ {0.74} \right)` is a relationship derived from :cite:`burkhardtMitigatingContrailCirrus2018`.




Roadmap
-------

See the contrail-related issues on `GitHub <https://github.com/dlr-pa/oac/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22module%3A%20contrails%22>`_.


