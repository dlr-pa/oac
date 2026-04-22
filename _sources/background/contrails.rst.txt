Contrail Module
===============

This page presents the OpenAirClim Contrail Module (oac.cont).
More detail is available in the upcoming publications Megill (2026) :cite:`megillAssessingContrailClimateImpacts2026` and Megill et al. (in prep.).
Since some results are still awaiting peer-reviewed publication, they are not yet included in the open-source version of OpenAirClim and are thus also not described here.
For specific questions relating to the module, please contact the core development group at openairclim@dlr.de.

To understand how to run the contrail module within OpenAirClim, please refer to the `user guide <../user_guide/contrails.html>`_.


OpenAirClim Contrail Module Methodology
---------------------------------------

The objective of the OpenAirClim Contrail Module (oac.cont) is to calculate the changes in the contrail-induced global annual mean stratospheric-adjusted radiative forcing due to an input emission inventory and scenario.
The high-level setup of oac.cont can be represented as:

.. mermaid::

    flowchart LR
        inv["Emission inventory"]
        inv --> form["Contrail formation"]
        form --> cccov["Contrail-cirrus coverage"]
        cccov --> RF["Contrail RF"]



Emission Inventory and Simulation Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The OpenAirClim emission inventory files are in netCDF format and consist of yearly aggregated emission data for each species and corresponding emission locations (latitude, longitude, pressure level).
Emission data may be provided on either regular or irregular grids and for multiple different years.
OpenAirClim either interpolates between the input emission inventories, or scales the results following a time evolution defined by the user.

To calculate a contrail climate impact, the yearly aggregated, 3D-resolved distance flown by a given aircraft at a given location (latitude, longitude, pressure level) is required within the input emission inventories (``distance`` variable in the netCDF file, in :math:`\mathrm{km}`).
The OpenAirClim Contrail Module uses pre-calculated data on the grid (``lat``: 48, ``lon``: 96, ``plev``: 18); emission inventories with finer resolutions will thus not result in more accurate results.
If multiple aircraft are present within the emission inventory, each entry in the inventory must be accompanied by an identifier (variable ``ac`` in the inventory netCDF file).
The user is free to choose appropriate identifiers for the inventory in question.
For example, the identifiers could correspond to the aircraft type (e.g. ``A320`` or ``B737``) or to the propulsion technology (e.g. ``SAF`` or ``H2C``).
Important is only that all identifiers in the emission inventories are defined in the simulation configuration file.

To calculate a contrail climate impact, the following contrail-relevant parameters must be defined for each identifier:

- ``G_250`` -- Schmidt-Appleman mixing line slope `G` [Pa/K] at 250 hPa. Valid range: :math:`\geq 0.48~\mathrm{Pa~K}^{-1}`. Alternatively, the necessary parameters to calculate the slope can be provided (see the `user guide <../user_guide/contrails.html>`_);
- ``PMrel`` -- Soot number emission index compared to the reference (:math:`1.5 \times 10^{15}~\mathrm{kg}^{-1}`). The valid range depends on whether the premium OpenAirClim functionality is available. For the conventional open-source module, the valid range is :math:`\geq 1.5 \times 10^{14}~\mathrm{kg}^{-1}`. With OpenAirClim Premium, this is extended to :math:`\geq 0.0~\mathrm{kg}^{-1}`;
- ``b`` -- Aircraft wingspan [m]. Valid range (high-soot regime only): [20 m, 80 m] (defaults to 35 m).


Contrail Formation
^^^^^^^^^^^^^^^^^^

Persistent contrail formation is represented using the Contrail Flight Distance Density (CFDD).
The CFDD is calculated for each fleet :math:`n` in the emission inventory and in all three spatial dimensions using,

.. math::

    \mathrm{CFDD}(n,i,j,k) = \frac{d(n,i,j,k)}{A(i,j)} \cdot p_{\mathrm{pcf}}(G_{250}(n),i,j,k)

where:

- :math:`d(n,i,j,k)` is the slant (3D) distance flown [:math:`\mathrm{km}`] in each grid box (aircraft, latitude, longitude, pressure level);
- :math:`A(i,j)` is the horizontal area of each grid column [:math:`\mathrm{km}^2`];
- :math:`G_{250}` is the slope of the Schmidt-Appleman mixing line [:math:`\mathrm{Pa~K}^{-1}`] at 250 hPa; and
- :math:`p_\mathrm{pcf}` is the probability of persistent contrail formation in each grid box (:math:`\in [0,1]`).

The pressure level 250 hPa, corresponding to Flight Level 340 or 34000 ft in the International Standard Atmosphere, is chosen as the reference pressure level without any loss of generality.

The :math:`p_\mathrm{pcf}` values are pre-calculated for 13 aircraft designs with :math:`G_{250}` values between :math:`0.48` and :math:`15.82~\mathrm{Pa~K}^{-1}` following the Megill & Grewe (2025) methodology :cite:`megillInvestigatingLimitingAircraftdesigndependent2025`.
For a given :math:`G_{250}`, the value is interpolated along modified logistic functions of the form,

.. math::

    \overline{p_\mathrm{pcf}}(G) = \frac{L_1}{1 + \mathrm{e}^{-k_1 \left( G - G_{0,1} \right)}} + \frac{L_2}{1 + \mathrm{e}^{-k_2 \left( G - G_{0,2} \right)}} + d

where :math:`\overline{p_\mathrm{pcf}}` is the global, area-weighted average :math:`p_\mathrm{pcf}`.
The function parameters are dependent on the pressure level.


Contrail-Cirrus Coverage
^^^^^^^^^^^^^^^^^^^^^^^^

Two different measures of contrail-cirrus coverage are considered by OpenAirClim:
total coverage across all optical depths (:math:`\mathrm{cov_{all\ \tau}}`) and optically thick coverage (:math:`\mathrm{cov_{\tau>0.05}}`).
The contrail-cirrus coverage submodule is split into two steps:
calculating :math:`\mathrm{cov_{all\ \tau}}` from :math:`\mathrm{CFDD}` using a dedicated parameterisation; and calculating :math:`\mathrm{cov_{\tau>0.05}}` from :math:`\mathrm{cov_{all\ \tau}}` using the scaling factor :math:`s_{\tau}`, derived from literature.

The relationship between :math:`\mathrm{CFDD}` and :math:`\mathrm{cov_{all\ \tau}}` is strongly non-linear.
It is represented by a hyperbolic tangent function of the form,

.. math::

    \mathrm{cov_{all\ \tau}}(n,i,j,k) = a \cdot \tanh \left(b \cdot \mathrm{CFDD}(n,i,j,k) \right)

where parameter :math:`a = 0.0772` controls the asymptotic maximum coverage and :math:`b = 132.3~\mathrm{km}` the sensitivity to the :math:`\mathrm{CFDD}`.
The coverage is then combined vertically using a random overlap assumption, i.e. assuming statistical independence between vertical layers :math:`k`,

.. math::

    \mathrm{cov_{all\ \tau}}(n, i, j) = 1 - \prod_{k} \left(1 - \mathrm{cov_{all\ \tau}}(n, i, j, k) \right)

and then area-weighted over the latitude :math:`i`,

.. math::

    \mathrm{cov_{all\ \tau}}(n, j) = \frac{\sum_{i} \mathrm{cov_{all\ \tau}}(n,i,j) \cdot A(i, j)}{\sum_{i} A(i, j)}

A scaling factor :math:`s_\tau` is applied to obtain an estimate of optically thick contrail-cirrus coverage :math:`\mathrm{cov_{\tau>0.05}}`.
This scaling factor is currently only a function of the soot number emission index :math:`EI_\mathrm{s}`, but further dependencies could be added in future updates.

.. important::

    Currently, the open-source version of OpenAirClim is only defined within the high-soot regime (:math:`EI_\mathrm{s} \geq 1.5 \times 10^{14}~\mathrm{kg}^{-1}`).
    Until approximately Summer 2027, the definition of the low-soot regime is available as **premium functionality** that needs to be licensed from the German Aerospace Center (DLR).

    Please contact the core development group at openairclim@dlr.de for more information.

Within the high-soot regime, the scaling factor is defined as,

.. math::

    s_\tau = 0.91 \cdot \arctan \left( 1.96 \cdot x ^ {0.58} \right)

where :math:`x` is the ``PMrel`` input value (:math:`EI_\mathrm{s}` compared to the reference :math:`1.5\times 10^{15}~\mathrm{kg}^{-1}`).
For a given aircraft :math:`n`, the optically thick contrail-cirrus coverage is then calculated using,

.. math::

    \mathrm{cov_{\tau>0.05}}(n,j) = \mathrm{cov_{all\ \tau}}(n,j) \cdot s_\tau(n)


Radiative Forcing
^^^^^^^^^^^^^^^^^

Separate relationships are derived between :math:`\mathrm{cov_{\tau>0.05}}` and RF for four regions defined by longitude: Americas (235, 330]; Europe, Middle-East and Africa (330, 60]; Asia and Australia (60, 160]; and Pacific (160, 235].
Within each region, RF is determined using a power-law relationship of the form,

.. math::

    \mathrm{RF}(j) = c_j \cdot \mathrm{cov_{\tau>0.05}}(j) ^ {\gamma_j}

where :math:`\mathrm{cov_{\tau>0.05}}(j)` is the area-weighted optically thick contrail-cirrus coverage for longitude band :math:`j` and :math:`c_j` and :math:`\gamma_j` are region-specfiic fit parameters.
The range of validity differs depending on the region due to data availability.
OpenAirClim therefore checks whether the contrail-cirrus coverage values fall outside of the fitted range and issues a warning if extrapolation occurs.

Because the model grid uses equal longitudinal spacing, each band contributes equally to the global mean RF.
It is thus simply the mean of :math:`\mathrm{RF}(j)`.


Post-Processing
^^^^^^^^^^^^^^^

The contrail module currently has a single post-processing step for Radiative Forcing, which depends on the size of the aircraft in the emission inventory.
Large aircraft tend to produce contrails with a higher number of ice crystals, resulting in overall deeper contrails.
Therefore, a simple relationship presented by Bruder et al. (2025, see their Supplementary Section S1) :cite:`bruderDLRCO2equivalentEstimator2025` is applied,

.. math::

    \mathrm{RF}_{\mathrm{mod}}(n) = \mathrm{RF}(n) \cdot \frac{v + w \cdot b(n)}{v + w \cdot b_{\mathrm{ref}}}

where :math:`b(n) \in [20~\mathrm{m}, 80~\mathrm{m}]` is the wingspan of aircraft identifer :math:`n`, :math:`w = 0.0287~\mathrm{m}^{-1}` and :math:`v = 0.396`.
If no wingspan is given, the default :math:`b_\mathrm{ref} = 35~\mathrm{m}` is used.
Currently, the correction is not valid for the low-soot regime -- within this regime, :math:`b(n)` is assumed to be equal to :math:`b_\mathrm{ref}`.


Attribution
^^^^^^^^^^^

The contrail module implements proportional attribution, consistent with the tagging-based approach used in other OpenAirClim modules.
In proportional attribution, the contribution of each aircraft identifier to the output variable is determined from its fractional contribution to the input variable.
Specifically, the contribution of identifier :math:`n` to :math:`\mathrm{cov_{all\ \tau}}` is determined from its relative contribution to :math:`\mathrm{CFDD}` and the contribution to :math:`\mathrm{RF}` is determined from its relative contribution to :math:`\mathrm{cov_{\tau>0.05}}`.


Roadmap
-------

See the contrail-related issues on `GitHub <https://github.com/dlr-pa/oac/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22module%3A%20contrails%22>`_.


