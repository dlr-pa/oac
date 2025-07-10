OpenAirClim Documentation
=========================

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



.. toctree::
   :maxdepth: 2

   installation
   quickstart
   api_ref


.. toctree::
   :hidden:

   imprint
   accessibility-statement
   erklaerung-zur-barrierefreiheit
   privacy-policy
   terms-of-use
