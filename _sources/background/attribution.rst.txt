Attribution Methods
===================

The process of attribution refers to quantifying how much each of several causal factors contributes to a change.
This is not a straightforward process for non-linear relationships, which are often found in atmospheric physics and chemistry.
The attribution method used is an important consideration, since it determines the share of responsibility.
Previous research has primarily focused on attribution methods for climate negotiations and policy, for example for determining each country's responsibility for current global temperature change :cite:`trudingerComparisonFormalismsAttributing2005`.
Some recent work, notably :cite:`boucherContributionGlobalAviation2021`, have instead considered sector-wise attribution.

In OpenAirClim, attribution methods are used to quantify the relative contribution of each aircraft identifier, whilst considering the global background and all other air traffic.
Attribution is used for the calculation of CO₂ and CH₄ radiative forcing.
An attribution method for contrails is currently in development.
Five attribution methods are available: none, residual, marginal, proportional (default) and differential.
If "none" is selected, calculations are done against pre-industrial conditions, assuming no other anthropogenic sources.

The attribution method can be selected in the config file:

.. code:: toml

    [responses]
    CO2.rf.attr = "proportional"  # "none", "residual", "marginal", "differential"
    CH4.rf.attr = "proportional"
    cont.attr = "proportional"  # work in progress



Methods
-------

We assume that :math:`x_\mathrm{ac}(t)` and :math:`x(t)` are the sources, for example CO₂ concentration, for a given aircraft identifer :math:`\mathrm{ac}` and all anthropogenic activities at time :math:`t`.
By definition, :math:`x_\mathrm{ac}(t)` must be included within :math:`x(t)`.
The resulting effect :math:`y_\mathrm{ac}(t)`, for example RF, is calculated by the function :math:`f` using one of the attribution methods as shown below.
We use the notation devised by :cite:`boucherContributionGlobalAviation2021`.

We differentiate between non-additive and additive methods :cite:`trudingerComparisonFormalismsAttributing2005`.
If a method is non-additive, the contribution of a group of aircraft identifiers differs depending on whether they are quantified together or separately.
In other words, using a non-additive method, :math:`y_\mathrm{A}(t) + y_\mathrm{B}(t) \neq y_\mathrm{A+B}(t)`.
This is not ideal, since we would prefer for the contributions of different aircraft to be combinable in any manner.
Therefore, OpenAirClim uses the additive *proportional attribution method* by default.


Non-Additive Methods
********************

A common method is the **residual attribution method**, otherwise referred to as the "all-but-one" method.
The effect attributed to a given source is the difference between two simulations: one with all anthropogenic activities and the other excluding the source in question.

.. math::

    y_\mathrm{ac}^R(t) = f(x(t)) - f(x(t) - x_\mathrm{ac}(t))

For small perturbations, this method is equivalent to the **marginal attribution method**, which determines the contribution "at the margin".

.. math::

    y_\mathrm{ac}^M(t) = \frac{\mathrm{d} f}{\mathrm{d} x} \bigg\rvert_{x(t)} \cdot x_\mathrm{ac}(t)

This method requires the derivative of the function to be available in OpenAirClim and may thus not be universally applicable.
The point at which the derivative is calculated is a topic of much contention (see e.g. :cite:`boucherContributionGlobalAviation2021`).
We choose to use the current conditions at time :math:`t`.


Additive Methods
****************

By default, OpenAirClim uses the **proportional attribution method** due to its additivity and simplicity.
For a single species, the proportional attribution method is also equivalent to the tagging method, used within the NOx module.

.. math::

    y_\mathrm{ac}^P(t) = \frac{x_\mathrm{ac}(t)}{x(t)} \cdot f(x(t))

Finally, the **differential attribution method** uses the differential of the effect with respect to the source.

.. math::

    y_\mathrm{ac}^D(t) = \int_0^t \frac{\partial f}{\partial x} \bigg\rvert_{x(t')} \frac{\mathrm{d} x_\mathrm{ac}(t')}{\mathrm{d} t'} \mathrm{d}t'

This method requires the derivative of the function to be available in OpenAirClim and may thus not be universally applicable.
