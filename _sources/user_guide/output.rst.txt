Output data
-----------


What gets written
=================

A simulation run writes its results into the configured ``output.dir``:

- ``<name>.nc`` - a time series of emissions, concentrations, radiative forcing
  and temperature change for each species and aircraft identifier
- ``<name>_metrics.nc`` - the requested climate metrics (AGWP, AEGWP, ATR,
  AGTP) if ``output.run_metrics = true``
- ``<name>_<spec>.png`` - a PNG plot per species for aircraft identifier TOTAL
  if ``output.run_plots = true``


Embedded metadata
=================

Every ``.nc`` file also carries a set of global attributes recording how it was
produced:

- ``config_hash`` - a 10-character deterministic hash of the resolved
  simulation configuration (input config file plus defaults)
- ``config_json`` - the resolved simulation configuration as JSON
- ``oac_version`` - the OpenAirClim version that produced the file
- ``oac_git_commit`` - the latest git commit of the OpenAirClim repository, if
  the run was performed by a git-tracked version of OpenAirClim
- ``created`` - UTC timestamp of the run
- ``user``, ``platform``, ``python_version`` - further information on where the
  simulation was run

The PNG files carry a smaller subset of the same information (``config_hash``,
``oac_version`` and ``created``) as metadata.

This embedded metadata is especially useful once output files get copied,
shared or moved away from the config that produced them. The ``config_hash``
lets you match a result back to a specific configuration, and ``config_json``
lets you recover that configuration even if the original ``.toml`` file is
gone. It also aids the core development team in helping you if you encounter
any bugs.


Working with the metadata
=========================

The :mod:`~openairclim.utils.output_metadata` module defines two command line
entry points to work with the metadata.

To get an idea of what files (netCDF and PNG) within a folder structure contain
OpenAirClim-specific metadata, you can run:

.. code-block:: bash

    # across a folder structure
    oac-get-metadata path/to/data
    oac-get-metadata path/to/data -r  # recurse into subfolders
    oac-get-metadata path/to/data -f  # show all metadata, not just hash

    # this also works for a single file
    oac-get-metadata path/to/file.nc

    # for more information:
    oac-get-metadata --help

If you have a hash and want to identify all files that belong to it, the
simplest way is to use a recursive ``grep``:

.. code-block:: bash

    # recursively list every file containing a given hash
    grep -rla "f9c18ecfe0" path/to/data

    # narrow the search to OpenAirClim's own output types
    grep -rla "f9c18ecfe0" --include="*.nc" --include="*.png" path/to/data

Finally, if you have identified a netCDF file for which you want to recreate
the config TOML file, you can run:

.. code-block:: bash

    oac-config-from-nc -i path/to/file.nc -o path/to/config.toml
