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
oac.run("Sensitivity_Analysis.toml")

# logging.disable(logging.NOTSET) 

ds = nc.Dataset('results/example_metrics.nc')
species = list(ds.variables['species'][:])
ATR100 = ds.variables['ATR_100_1940'][:]

total_ATR100 = ATR100[species.index('total')]
print(total_ATR100)

