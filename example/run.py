"""Demonstration of OpenAirClim simulation run"""

# if you have not added the oac folder to your PATH, then you also need to
# import sys and append to PATH using sys.path.append(`.../oac`)
import os
import cProfile
import pstats
import openairclim as oac

# change directory to match current file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

with cProfile.Profile() as profile:
    oac.run("example.toml")

stats = pstats.Stats(profile)
stats.sort_stats(pstats.SortKey.TIME)
stats.print_stats(0.1)
stats.dump_stats("stats.prof")
