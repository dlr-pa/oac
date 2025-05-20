# Hydrogen fugitive emissions

This documentation describes workflows associated with module `calc_h2.py`. The purpose of this module is the calculation and asssessment of climate impacts from hydrogen fugitive emissions.

## Installation

The module `calc_h2.py` depends on the python package *tensorflow*. On Linux and MacOS, *tensorflow* can be installed with the pip package manager as described in the [online documentation](https://www.tensorflow.org/install/pip). For Windows operating systems, some pre-requirements must be met as explained in the following.

### Windows

Here, the installation of *tensorflow* on Windows operating systems is described. The correct installation was tested on a Windows 11 Enterprise computer with X64 architecture. The installation procedure follows the instruction given in this [online tutorial](https://youtu.be/0w-D6YaNxk8).

#### System Requirements
First of all, the [latest Microsoft Visual C++ Redistributable Version](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170) have to be installed on the Windows computer.

#### conda environment
Create a dedicated conda environment from the given environment file for the execution of `calc_h2.py` routines:

```
conda env create -f environment_h2.yaml
```

The dependencies in this environment files are the same as for the *dev* version. However, a python version of 3.11 is enforced.

#### tensorflow package
Unfortunately, the tensorflow package is not installable with the conda command, and must be installed with the pip package manager instead. To avoid possible dependency conflicts, it is important to have installed all other required dependencies with conda first, e.g. via the environment file, before proceeding with the pip command.

Navigate to the Python Package Index ([PyPI](https://pypi.org/project/tensorflow/#files)), and download the `tensorflow-xxx.whl` file suitable for CPyhton 3.11 and Windows systems. In your command shell, navigate to the downloaded file, activate the installed conda environment `oac_h2` and execute following commmand replacing the file name with the name of the downloaded version:

```
pip install tensorflow-xxx.whl
```
If the pip command run successfully, the tensorflow package and its dependencies are installed in the dedicated conda environment.