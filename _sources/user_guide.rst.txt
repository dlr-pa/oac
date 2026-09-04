User Guide
==========

Here you can find some useful documentation of installing OpenAirClim, the
input data it requires, and its workflows and modules. We are actively working
on this guide.

The general OpenAirClim workflow is depicted in the following flowchart. Input
files are shown in yellow, pre-calculated and built-in databases in grey and
the output files in green.

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
    :caption: Contents

    user_guide/installation
    user_guide/input
    user_guide/evolution
    user_guide/contrails
