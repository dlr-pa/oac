.. _swv:
Stratospheric Water Vapour Module
=================================

Here, the OpenAirClim Stratospheric Water Vapour Module (oac.calc_swv) is described. The methods used in this module are based on the work from Harmsen (2026) :cite:`Harmsen_2026`.


CH\ :sub:`4` Oxidation Method
-----------------------------

.. _sec:methodch4:

SWV is formed during the oxidation of CH\ :sub:`4`. Aviation emissions influence this process as the NO\ :sub:`x` emissions of aircraft cause the OH concentration to increase, thereby accelerating the dissimilation of CH\ :sub:`4` and consequently lowering the global CH\ :sub:`4` concentration.

To implement this adjustment within the OAC framework, a two-step approach is taken. This approach consists of first calculating the SWV perturbation mass due to CH\ :sub:`4` oxidation, and second, relating this mass to a radiative forcing (RF).

The impact of changes in CH\ :sub:`4` concentrations on SWV concentrations is determined using the parameterisation originally derived by :cite:`Austin2007,Oman2008` and later adapted by :cite:`Hegglin2014`.

:cite:`Austin2007` state that the total concentration of water vapour at a certain location (pressure level :math:`p` and latitude :math:`\theta`) in the stratosphere at a certain time (also referred to as SWV(:math:`\theta, p, t`)) is dependent on the H\ :sub:`2`\ O entering the stratosphere lagged by the age-of-air (:math:`H_2O|_e(t-AoA)`) and the amount of CH\ :sub:`4` that is oxidized (:math:`CH_4|_0(t-AoA) - CH_4(\theta, p, t)`). The total amount of SWV is estimated using the following equation:


.. math::
    \text{H}_2\text{O}(\theta, p, t) = \text{H}_2\text{O}|_{\text{e}}(t - AoA) + 2[\text{CH}_4|_0(t - AoA) - \text{CH}_4(\theta, p, t)],

The entering concentration is determined at the tropical tropopause. In this equation, the assumption is made that each CH\ :sub:`4` molecule produces two water molecules (:cite:`leTexier1988`), as this factor 2 is widely used in literature. It should be noted that in reality this factor would be dependent on altitude (:cite:`Frank2018`).

The equation above can be rewritten using the fractional release factor :math:`\alpha`. :math:`\alpha` is defined as shown in the equation below by :cite:`Hegglin2014`, where :math:`CH_4(\theta, p, t)` corresponds to the methane concentration at a given location and time in the stratosphere and :math:`CH_4|_e(t-AoA)` is the amount of CH\ :sub:`4` entering the stratosphere lagged by the AoA.

.. math::
   \alpha(\theta, p) =
   \frac{CH_{4}|_e(t-AoA) - CH_{4}(\theta, p, t)}
        {CH_{4}|_e(t-AoA)}

To calculate the change in SWV due to methane oxidation (:math:`\Delta SWV`) based on the change in methane entry concentration with corresponding time lag (:math:`\Delta CH_4|_e(t-AoA)`), both equations above can be combined as done by :cite:`Hegglin2014`. In this new equation, the effect of the water vapour transported to the stratosphere is not included, as the focus is solely on the SWV formed by CH\ :sub:`4` oxidation. This results in:

.. math::
   \Delta SWV(\theta, p, t) =
   2 \alpha(\theta, p) \Delta CH_{4}|_e(t-AoA)

The equation above is implemented in OAC. To achieve this, :math:`\alpha` must be determined. This is done based on a reference scenario, from which :math:`\alpha` can be obtained. :math:`\alpha` is assumed to remain constant over time and is therefore not time dependent. This is valid because the fractional release is dependent on circulation and the background OH concentration, which are assumed to remain constant (:cite:`Hegglin2014`).

The reference scenario used to determine :math:`\alpha` is based on the Halogen Occultation Experiment (HALOE) data. Specifically, the HALOE zonal mean vertical CH\ :sub:`4` profile averaged over the period from October 1991 to 1999, as reported by :cite:`Myhre2007`. This distribution is displayed in Figure 1 of :cite:`Myhre2007`.

The period in the 1990s is chosen because CH\ :sub:`4` levels are relatively steady, so the time lag becomes of minor importance. Using this observed profile together with the average methane entry value over this period of 1772 ppbv determined by the National Oceanic and Atmospheric Administration (NOAA) [#noaa]_, a vertical :math:`\alpha` profile can be estimated.

.. [#noaa] https://gml.noaa.gov/webdata/ccgg/trends/ch4/ch4_annmean_gl.txt (accessed 05-12-2025)

The AoA can be determined using an empirical relationship between :math:`\alpha` and AoA established by :cite:`Hegglin2014`. They describe the relation using a third-order polynomial:

.. math::
   AoA = 0.3 + 15.2\alpha - 21.2\alpha^2 + 10.4\alpha^3

Based on this relationship, an AoA profile is generated. Since the smallest temporal resolution in OAC is one year, the calculated AoA values are rounded to integer years and the corresponding locations based on altitude and latitude are lagged accordingly.

Using the formulas explained above, :math:`\Delta SWV` can be determined for each location for a given CH\ :sub:`4` entry concentration. The resulting change in total SWV mass, :math:`\Delta m_{\mathrm{SWV}}`, is computed by summing over all locations.

This is done by determining the SWV mass at each location using the air mass of the corresponding location based on the International Standard Atmosphere (ISA), :math:`m_{air}(\theta, p)`, and the molar masses of air (:math:`M_{air}`) and water (:math:`M_{H_2O}`). Summing over all pressure levels and latitudes gives the total SWV mass (done in the equation below). This total SWV mass is then used to calculate the associated RF.

.. math::
   \Delta m_{SWV}(t) =
   \sum_{\theta} \sum_{p}
   \Delta SWV(\theta, p, t)
   \, m_{air}(\theta, p)
   \frac{M_{H_2O}}{M_{air}}

The RF of SWV is determined using the relation found by :cite:`Pletzer2024` (Figure 7.17) Using the relation from :cite:`Pletzer2024`, a change in SWV mass can be directly related to a change in RF.

Combining the two steps, the total RF from SWV caused by CH\ :sub:`4` oxidation can be determined using OAC.

In OAC, the value of global CH\ :sub:`4` change due to aviation emissions is already calculated. When combining this CH\ :sub:`4` change with the method described above, the total amount of SWV caused by CH\ :sub:`4` oxidation in the stratosphere can be determined.

It is important to note that due to aviation emissions, CH\ :sub:`4` concentration decreases, and therefore the amount of SWV also decreases. This assumption is valid as long as CH\ :sub:`4` concentrations do not drop below pre-industrial values (which is very unlikely in the upcoming centuries). In such scenarios, the data of :cite:`Myhre2007` might no longer be sufficient, as other non-linear effects may become relevant.

Furthermore, :cite:`Pletzer2024` derived the RF relation from simulation data with perturbations up to 160 Tg. Therefore, for perturbations larger than 160 Tg, this relation is not valid. Additionally, due to the polynomial fit, small SWV increases (less than 1.6 Tg) result in a negative RF.

Since this behaviour does not reflect atmospheric chemistry, for values smaller than 1.6 Tg the RF is set to 0 W m\ :sup:`-2`.

















