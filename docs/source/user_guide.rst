User Guide
==========

Here you can find some useful documentation of the OpenAirClim workflows, modules and data processed.
We are actively working on this guide.

.. contents:: Contents
    :local:

.. toctree::
    :maxdepth: 1
    :glob:

    user_guide/*


General workflow
----------------

In the following flowchart, the general OpenAirClim workflow is depicted.
Input files are shown in yellow, built-in data bases in grey, the OpenAirClim process in blue and output files in green.

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
