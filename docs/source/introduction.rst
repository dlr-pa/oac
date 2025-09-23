What is OpenAirClim?
====================

OpenAirClim is a model for simplified evaluation of the approximate chemistry-climate impact of air traffic emissions.
The model represents the major responses of the atmosphere to emissions in terms of composition and climate change.
Instead of applying time-consuming climate-chemistry models, a response model is developed and applied which reproduces the response of a climate-chemistry model without actually calculating ab initio all the physical and chemical effects.
The responses are non-linear relations between localized emissions and Radiative Forcing and further climate indicators.
These response surfaces are contained within look-up tables.
OpenAirClim builds upon the previous AirClim framework.
In comparison with AirClim, following new features are introduced:

- Standardized formats for configuration file (user interface) and emission inventories (input) and program results (output)
- Possibility of full 4D emission inventories (3D for several time steps)
- Non-linear response functions for NOx including contribution approach (tagging) and dependency on background
- Contrail formation also depending on fuels and overall efficiencies
- Inclusion of different fuels
- Choice of different CO2 response models
- Choice of temperature models and sea-level rise
- Uncertainty assessment and Robustness Metric based on Monte Carlo Simulations
- Parametric scenarios as sensitivities, e.g. at post-processing level: climate optimized routings


Scientific Background
---------------------

The impact of aviation on climate amounts to approximately 3.5% of the total anthropogenic climate warming :cite:`leeContributionGlobalAviation2021`.
A large part of the aviation's impact arises from non-CO2 effects, especially contrails :cite:`burkhardtMitigatingContrailCirrus2018, bickelContrailCirrusClimate2025` and nitrogen oxide emissions :cite:`stevensonRadiativeForcingAircraft2004, myhreRadiativeForcingDue2011`.
Impact of non-CO2 effects depend in particular on the location and time of emissions :cite:`lundEmissionMetricsQuantifying2017, frommingInfluenceWeatherSituation2021`, hence a regional dependence of impacts exists.
As impacts of individual non-CO2 effects show a different spatial dependence, the relationship between impacts and associated emissions can be best described in non-linear relationships, i.e. equations or algorithms based on look-up tables.
Specifically, the climate impact of an aircraft depends on where (and when) an aircraft is operated.
In addition, using different types of fuel generally changes the importance of the non-CO2 effects.


Layout
------

.. figure:: _static/OAC-chart.png
    :alt: Overview of the OpenAirClim framework
    :align: center

    Overview of the OpenAirClim framework

- User interface for settings in the run control and outputs (grey)
- Definition of background conditions, such as aviation scenarios, uncertainty ranges and aviation inventories (orange)
- A link to a pre-processor for aviation inventories (blue)
- Processor for a full 4D-emission inventory at multiple timesteps (magenta)
- A framework for the application of non-linear response functions (red) to these emission inventories.
- Response functions for CO2 and climate / temperature and sea-level changes
- Parametric scenarios as sensitivities (yellow), e.g. at post-processing level: climate optimized routings
- Output: Warnings, errors (log files), climate indicators and diagnostics (green), values of climate metrics and robustness metrics (grey)
