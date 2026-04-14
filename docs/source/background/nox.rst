NO\ :sub:`x` climate impacts
============================

In OpenAirClim, the climate impacts of emitted NO\ :sub:`x` can be simulated using one of the two approaches:
tagging or perturbation. The underlying response surfaces are different and have been created by different 
climate-chemistry model setups.


Tagging approach
----------------

.. important::

    Our baseline development is built upon the tagging approach for NO\ :sub:`x` climate impacts.
    Currently, NO\ :sub:`x` response surfaces using the tagging approach are not yet validated
    and have a limited resolution. Therefore, these responses should currently not be used for scientific purposes.
    Meanwhile, the response surfaces using the perturbation approach can be used.
    Be aware that due to the non-linear NO\ :sub:`x` chemistry, different results are expected
    when comparing both approaches. Particularly, the two approaches should not be mixed between
    species, e.g. O\ :sub:`3` and CH\ :sub:`4`.


Perturbation approach
---------------------

The worfklows for calculating the responses differ for the individual species. This is due to different 
species lifetimes and possible interdependencies between species. The workflows and the underlying 
response surfaces for the perturbation approach have been adpoted from the AirClim framework :cite:`greweAirClimEfficientTool2008`.


Ozone
^^^^^

Ozone has relatively short lifetimes and a direct relation between NO\ :sub:`x` emission and resulting
O\ :sub:`3` radiative forcing is applied as shown in the following workflow diagram.

.. mermaid::
    
    flowchart LR
        inv["NO<sub>x</sub> emissions"]
        inv --> RF["O<sub>3</sub> RF"]

In detail, the RF changes for a perturbation scenario are calculated by folding the emissions 
in the input inventory with the pre-calculated radiative forcings of the idealized emission regions.
Here, the values of the pre-calculated RF are normalized using the imported NO data for each 
idealized emission region.

.. math::

    \mathrm{\Delta RF} = \sum_n E_n \sum_k \epsilon_k \frac{\mathrm{\Delta RF}(i_k, j_k)}{\mathrm{N}(i_k, j_k)}

where:

- :math:`E_n` are the NO\ :sub:`x` emission masses for each location in the inventory.
- :math:`\epsilon_k` are the weighting factors of the surrounding neighbours :math:`(k = 1, .., 4)` on the latitude and altitude dimensions for each emission location.
- :math:`\mathrm{\Delta RF}(i_k, j_k)` are the latitude and altitude dependent changes in radiative forcings of the idealized emission regions.
- :math:`\mathrm{N}(i_k, j_k)` are the latitude and altitude dependent masses of imported NO of the idealized emission regions.


Methane
^^^^^^^

The lifetimes of methane are significantly longer than the one-year step size of the OpenAirClim simulations.
Therefore, the workflow is different from the calculation procedure of ozone RF. According to the following workflow diagram,
first the CH\ :sub:`4` concentration changes are calculated from the NO\ :sub:`x` emissions which are used subsequently
to evaluate changes in radiative forcing.

.. mermaid::

    flowchart LR
        inv["NO<sub>x</sub> emissions"]
        inv --> conc["CH<sub>4</sub> concentration changes"]
        conc --> RF["CH<sub>4</sub> RF"]

The change in methane concentration is derived by regarding the difference of two differential equations :cite:`greweAirClimEfficientTool2008`
which is solved numerically in OpenAirClim:

.. math::

    \frac{d}{dt} \Delta C^{\mathrm{CH_4}} = \frac{\delta}{1+\delta} \tau_{\mathrm{CH_4}}^{-1} C_0^{\mathrm{CH_4}} - \frac{1}{1+\delta} \tau_{\mathrm{CH_4}}^{-1} \Delta C^{\mathrm{CH_4}}

where:

- :math:`\Delta C^{\mathrm{CH_4}}` is the change in methane concentration.
- :math:`\delta` is the relative change in lifetime.
- :math:`\tau_{\mathrm{CH_4}}` is the methane perturbation lifetime (here: 12 years).
- :math:`C_0^{\mathrm{CH_4}}` is the time dependent background methane concentration.

The relative lifetimes are evaluated on the basis of inventory years. Similary as for the ozone
radiative forcings, relative lifetime changes have been pre-calculated for 
idealized emission regions, and an average :math:`\delta` is computed by folding the emissions 
in the input inventory with pre-calculated lifetime changes of the idealized emission regions.

Finally, the radiative forcing is calculated from the :math:`\Delta C^{\mathrm{CH_4}}` time series 
by applying the relation defined in Etminan et al. (2016, see their Table 1) :cite:`Etminan_2016`.


PMO and SWV
^^^^^^^^^^^

The climate impacts of Primary Mode Ozone (PMO) and Stratospheric Water Vapour (SWV) 
are computed independently from the chosen approach (tagging vs. perturbation). Both species
depend on the previously evaluated methane.

According to Dahlmann et al. (2016, see their appendix A.3) :cite:`dahlmannCanWeReliably2016`,
the radiative forcing of PMO is calculated using a linear relationship to the forcing of methane:

.. math::

    \mathrm{\Delta RF}(\mathrm{PMO}) = 0.29 \cdot \mathrm{\Delta RF}(\mathrm{CH_4})

The methods for the computation of climate impacts from SWV are based on the work from Harmsen (2026) :cite:`Harmsen_2026`.
For details, refer to the documentation in :ref:`swv`.
