Download Repository Data
========================

.. note::

    OpenAirClim's simulations require response surface data and background
    concentration scenarios, which are published independently of
    ``openairclim`` at `dlr-pa/oac-data <https://github.com/dlr-pa/oac-data>`__.
    From v0.18 onwards, this data must be installed separately, irrespective of
    whether you installed OpenAirClim from source, through pip or through
    conda. It is also possible to use your own data, but it must be in the same
    format.

To download the default data, activate the python environment that includes
``openairclim`` and run once:

.. code-block:: bash

    oac-download-data

By default, this fetches the data version pinned by your installed
``openairclim`` release into a shared, per-user cache directory. This means
that multiple ``openairclim`` installations on the same machine reuse a single
copy of the data, useful for developers. It also allows for multiple different
versions of the data to be present at once.

OpenAirClim will look in the per-user cache directory for the response surface
and background concentration scenario data by default. To use the data in the
cache, leave ``background.dir`` and ``responses.dir`` unset in the config. To
use custom data, or data stored elsewhere on your machine, point OpenAirClim
at the relevant folder instead. See also :doc:`user_guide/01_input`.

Useful overrides:

.. code-block:: bash

    # fetch a specific data version, or a specific Zenodo record/DOI
    oac-download-data --version 1.2.0
    oac-download-data --record 10.5281/zenodo.1234567

    # download into a custom, one-off location (only affects this download)
    oac-download-data --output-dir /path/to/data

    # override the default cache location itself, so both downloads and
    # config resolution consistently use it
    export OPENAIRCLIM_DATA_DIR=/path/to/data
    oac-download-data

Run ``oac-download-data --help`` for the full list of options.
