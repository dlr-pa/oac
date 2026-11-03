"""Demonstration of OpenAirClim simulation run"""

# if you have not added the oac folder to your PATH, then you also need to
# import sys and append to PATH using sys.path.append(`.../oac`)
import os
import openairclim as oac
import netCDF4 as nc
import numpy as np
import logging

# logging.disable(logging.CRITICAL)  # Suppresses everything below CRITICAL level
# change directory to match current file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# oac.run("Example_Uncertainty.toml")
# oac.repeated_sampling("Example_Uncertainty.toml")
# oac.uncertainty("Example_Uncertainty.toml")
# oac.sensitivity("Example_Uncertainty.toml")
# oac.surrogate_gsa("Example_Uncertainty.toml")

# logging.disable(logging.NOTSET) 



