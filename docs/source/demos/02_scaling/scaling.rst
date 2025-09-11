Scaling
=======

In this example, the time evolution of type **scaling** is demonstrated.

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