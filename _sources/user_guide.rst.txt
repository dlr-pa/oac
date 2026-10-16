User Guide
==========

This guide covers how to install and use OpenAirClim as a user - installing
input data, configuring emission inventories, and running simulations to
explore aviation's climate impact. If you are planning on developing
OpenAirClim itself, see the :doc:`developer guide <dev_guide>` instead.

**General Workflow**

.. mermaid::

    ---
    config:
        look: handDrawn
        theme: neutral
    ---
    flowchart LR
        classDef input fill:#FFFAA0
        classDef builtin fill:#D3D3D3
        classDef process fill:#0096FF
        classDef output fill:#32CD32
        CONFIG[/Configuration/]:::input
        INV[/Emission<br>inventories/]:::input
        EVO[/Time evolution/]:::input
        RESP[(Response<br>surfaces)]:::builtin
        BG[(Background<br>inventories)]:::builtin
        OAC[oac]:::process
        TS[/"Time series<br>(emis, conc, RF, dT)"/]:::output
        METR[/"Climate metrics<br>(AGTP, AGWP, ATR)"/]:::output
        DIAG[/Diagnostics/]:::output
        PLT[/Plots/]:::output
        CONFIG --> OAC
        INV --> OAC
        EVO -.-> OAC
        RESP --> OAC
        BG --> OAC
        OAC --> TS
        OAC --> METR
        OAC --> DIAG
        OAC --> PLT


.. toctree::
    :maxdepth: 1
    :hidden:

    user_guide/installation
    user_guide/input
    user_guide/evolution
    user_guide/contrails
    user_guide/output

**Contents**

- :doc:`user_guide/installation` - install OpenAirClim and download the
  repository data it needs to run
- :doc:`user_guide/input` - the configuration file, emission inventories and
  other input data a simulation requires
- :doc:`user_guide/evolution` - scaling or normalising emission inventories
  over time
- :doc:`user_guide/contrails` - running the contrail module
- :doc:`user_guide/output` - the output files a simulation produces, and the
  provenance metadata embedded in them
