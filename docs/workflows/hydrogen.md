# Hydrogen fugitive emissions

The purpose of the module `calc_h2.py` is the calculation and asssessment of climate impacts from hydrogen fugitive emissions. The routines within this module are executed as a stand-alone program. This documentation explains the installation of the module and instructs the user how to get started.

## Installation

The module `calc_h2.py` depends on the python package *tensorflow*. On Linux and MacOS, *tensorflow* can be installed with the pip package manager as described in the [online documentation](https://www.tensorflow.org/install/pip). For Windows operating systems, some pre-requirements must be met as explained in the following.

### Windows

Here, the installation of *tensorflow* on Windows operating systems is described. The correct installation was tested on a Windows 11 Enterprise computer with X64 architecture. The installation procedure is similiar to the instruction given in this [online tutorial](https://youtu.be/0w-D6YaNxk8).

#### System Requirements
First of all, the [latest Microsoft Visual C++ Redistributable Version](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170) have to be installed on the Windows computer.

#### conda environment
Create a dedicated conda environment from the given environment file for the execution of `calc_h2.py` routines:

```
conda env create -f environment_h2.yaml
```

The dependencies in this environment files are the same as for the *dev* version. However, a python version of 3.12 is enforced.

#### tensorflow package
Unfortunately, the tensorflow package is not installable with the conda command, and must be installed with the pip package manager instead. To avoid possible dependency conflicts, it is important to have installed all other required dependencies with conda first, e.g. via the environment file as stated above, before proceeding with the pip command.

Navigate to the Python Package Index ([PyPI](https://pypi.org/project/tensorflow/#files)), and download the `tensorflow-xxx.whl` file suitable for CPyhton 3.12 and Windows systems. In your command shell, navigate to the downloaded file, activate the installed conda environment `oac_h2` and execute following commmand replacing the file name with the name of the downloaded version:

```
pip install tensorflow-xxx.whl
```
If the pip command run successfully, the tensorflow package and its dependencies are installed in the dedicated conda environment. However, this installation of tensorflow causes some mismatches with the netcdf4 package. In order to resolve the incompatibility, execute following commands:
```
conda remove netcdf4
pip install netcdf4
```

## Getting started

### Download background SSP scenarios
Essential input to the hydrogen module are the background concentrations from Shared Socioeconomic Pathways (SSP) scenarios. You can [download](https://doi.org/10.5281/zenodo.15475946) corresponding files comprising data from the CMIP6 project which have been adapted for the hydrogen module of OpenAirClim. Place the downloaded netCDF files in the following folder:
```
repository/h2/SSP_scenarios/
```

## Usage

By executing the function `run_h2_case()`, a concrete case of aviation-induced fugitive hydrogen emissions is simulated and the results including perturbations and radiative forcing are ouput to an xarry Dataset. The arguments of this functions are:
```
    wd: working directory
    scenario: SSP scenario
    start_year: year to start the simulation
    t_mid: year of midpoint of adoption curve
    m_adp: adoption rate - slope of adoption curve
    f_app: leakage fraction during application
    f_del: leakage fraction during delivery
    f_prod: leakage fraction during production
    kd: rate of deposition
```
The function can be executed in two ways. The first method is to execute the hydrogen module as a Python script. Then, the pre-defined arguments of the module are used and the output is printed to the console.
```
python calc_h2.py
```
The second method is more flexible in terms of adaption of the input parameters and output of the results. Here, the user creates a dedicated Python script importing the openairclim package and executing the `run_h2_case()` function with customized arguments:
```
import openairclim as oac

oac.run_h2_case(..)
```
When using this method, it is important to set the working directory to the parent folder of the openairclim package, e.g. `wd="C:/oac/"`. The scenario argument must match to the downloaded SSP scenarios and can have one of the following values:
```
"ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp460", "ssp585"
```

## References
- Gunter, F. A. (2024). The Climate Impact of Hydrogen Leakage in Aviation - A Machine Learning Approach to Long-Term Scenario Forecasting. (Master Thesis, TU Delft). https://resolver.tudelft.nl/uuid:67633521-cd00-4565-a5d6-5536d497acb9