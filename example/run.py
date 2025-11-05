"""Demonstration of OpenAirClim simulation run"""

# if you have not added the oac folder to your PATH, then you also need to
# import sys and append to PATH using sys.path.append(`.../oac`)
import os
import openairclim as oac

# change directory to match current file
os.chdir(os.path.dirname(os.path.abspath(__file__)))
oac.run(
    r"C:\Users\atzeh\PycharmProjects\OAC_Thesis\ATZE\Scenario_Analysis\first_scenario.toml"
)
