Input Data
==========


Download or generate emission inventories
-----------------------------------------

Four-dimensional air traffic emission inventories are essential inputs to 
OpenAirClim. If this is your first time using OpenAirClim, we recommend using
the example or artifically generated emission inventories. Functionality to
convert existing emission inventories can be found at the OpenAirClim-addon
`gedai <https://liammegill.github.io/gedai>`__. If you have any questions or
issues with these conversion tools, please use ``gedai``'s Issues workflow on
`Github <https://github.com/liammegill/gedai>`__.

The emission inventories which were created as part of the DLR internal
`Development Pathways for Aviation up to 2050 (DEPA 2050) <https://elib.dlr.de/142185/>`__
project comprise realistic emission data sets for global air traffic in 5-year
steps between 2020 and 2050. These example inventories can be accessed at
`Zenodo <https://doi.org/10.5281/zenodo.11442322>`__ and downloaded using the
command line:

.. code-block:: bash

    oac-download-zenodo 11442322 -o "example/input/"

Depending on the settings chosen in the configuration file, the computational
time of the configured simulations could be long. If you are more interested in
testing or developing OpenAirClim, you might want to use artifically generated
data instead. To do so using the build-in random generator, execute the
following command:

.. code-block:: bash

    oac-create-artificial-inventories -o "example/input/"

The underlying script creates a series of emission inventories comprising
random emission data.

It is also possible to create emission inventories from other sources, such as
from ADS-B data or using a trajectory generator. Please check out the
OpenAirClim-addon `gedai <https://liammegill.github.io/gedai>`__ if you are
interested in this. However, be aware that generating OpenAirClim-compatible
emission inventories in such a manner can be time-consuming and computationally
expensive.


Create input data
-----------------

Depending on your use case, you may need to scale or normalize your emission
inventories over time. A common example would be that you have a global
emission inventory for a certain year, for example 2020, and want to simulate
the aviation industry's emissions over time, say between 1940 and 2200. In this
case, you can scale your emission inventory along a time-dependent scenario,
starting from 0 in 1940 and, for example, increasing by x% per year. In
OpenAirClim, this can be achieved using either a *scaling* or *normalization*.
To understand the difference between normalization and scaling, see
:doc:`user_guide/02_evolution`.

Example normalization and scaling files can be created using the following
command:

.. code-block:: bash

    oac-create-time-evolution -o "example/input/"

The underlying script creates two time evoluation files that control the
temporal evoluation of the emission data, one for normalization and the other
for scaling. These files are added to the directory ``example/input``. To
include the files in the OpenAirClim run, the file location must be provided in
the configuration ``.toml`` file as the ``time.file`` variable. The
``example.toml`` file has the following lines commented out:

.. code-block:: toml

    [time]
    # ...
    # file = "time_scaling_example.nc"
    # file = "time_norm_example.nc"

To use either scaling or normalization, simply uncomment one of the lines.
