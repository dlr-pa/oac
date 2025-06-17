""" Through __init__.py openairclim is recognized as a Python package.

Objects defined within the submodules are made available
to the user by "import openairclim".
"""

# from os.path import dirname, abspath
from openairclim.__about__ import *  # noqa: F401, F403
from openairclim.main import *  # noqa: F401, F403
from openairclim.read_config import *  # noqa: F401, F403
from openairclim.read_netcdf import *  # noqa: F401, F403
from openairclim.construct_conc import *  # noqa: F401, F403
from openairclim.interpolate_space import *  # noqa: F401, F403
from openairclim.interpolate_time import *  # noqa: F401, F403
from openairclim.calc_response import *  # noqa: F401, F403
from openairclim.calc_co2 import *  # noqa: F401, F403
from openairclim.calc_ch4 import *  # noqa: F401, F403
from openairclim.calc_cont import *  # noqa: F401, F403
from openairclim.calc_dt import *  # noqa: F401, F403
from openairclim.calc_metric import *  # noqa: F401, F403
from openairclim.uncertainties import *  # noqa: F401, F403
from openairclim.utils import *  # noqa: F401, F403
from openairclim.plot import *  # noqa: F401, F403
from openairclim.write_output import *  # noqa: F401, F403

# __all__ = ['read_config', 'read_inventories']
# ROOT_DIR = dirname(abspath(__file__))
# Logging initialisation code would go here #
