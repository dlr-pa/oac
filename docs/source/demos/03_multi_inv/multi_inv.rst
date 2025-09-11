Multiple emission inventories
=============================

In this example, **no** time evolution file is given, but multiple emission inventories are given as input.
OpenAirClim will interpolate between discrete inventory years.

Imports
-------
In some cases, it might be necessary to define `sys.path`, such that the interpreter finds the openairclim package.
Alternatively, the environment variable, e.g. `PYTHONPATH`, can be configured.

.. jupyter-execute::
    
    # Import packages
    
    import sys
    sys.path.append("D:/oac")   # ADAPT THIS PATH TO YOUR SETTINGS
    import xarray as xr
    import matplotlib.pyplot as plt
    import openairclim as oac