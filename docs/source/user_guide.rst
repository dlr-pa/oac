User Guide
==========

Here you can find some useful documentation of the OpenAirClim workflows, modules and data processed.
We are actively working on this guide.

General workflow
----------------

.. mermaid::

    ---
    config:
        look: handDrawn
        theme: neutral
    ---
    flowchart LR
        CONFIG[/Configuration/]
        INV[/Emission<br>inventories/]
        EVO[/Time evolution/]
        RESP[(Response<br>surfaces)]
        BG[(Background<br>inventories)]
        OAC[oac]
        style OAC fill:#87CEEB
        TS[/"Time series<br>(emis, conc, RF, dT)"/]
        METR[/"Climate metrics<br>(AGTP, AGWP, ATR)"/]
        DIAG[/Diagnostics/]
        PLT[/Plots/]
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
    :glob:

    user_guide/*
