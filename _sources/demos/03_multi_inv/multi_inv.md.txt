---
file_format: mystnb
kernelspec:
  name: python3
---

# Multiple emission inventories

In this example, **no** time evolution file is given, but multiple emission
inventories are given as input. OpenAirClim will interpolate between discrete
inventory years.


## Imports
If the openairclim package cannot be imported, make sure that you have
installed the package with pip or added the oac source folder to `PYTHONPATH`.

```{code-cell} ipython3
import xarray as xr
import matplotlib.pyplot as plt
from openairclim.utils.download_zenodo import download
import openairclim as oac

xr.set_options(display_expand_attrs=False)
```


## Input files

In order to be able to execute this example simulation, two types of input are
required.

* Configuration file `multi_inv.toml`
* Emission inventories `emi_inv_20XX.nc`

### Emission inventories

* Source: DLR research study [DEPA 2050](https://elib.dlr.de/142185/)
* Inventory years: 2030, 2040, 2050
* Available for download in suitable OpenAirClim format

```{code-cell} ipython3
%%capture
# Download inventories from zenodo
download(
    record_or_doi="https://doi.org/10.5281/zenodo.11442322",
    file_glob="emi_inv_20[3-5]0.nc",
    output_dir="../input/",
)
```


## Simulation run

```{code-cell} ipython3
oac.run("multi_inv.toml")
```


## Results

### Time series

* Emission sums
* Concentrations
* Radiative forcings
* Temperature changes

```{code-cell} ipython3
results_ds = xr.load_dataset("results/multi_inv.nc")
display(results_ds)
```

```{code-cell} ipython3
# Plot Radiative Forcing and Temperature Changes

ac = "TOTAL"
rf_cont = results_ds.RF_cont.sel(ac=ac) * 1000
rf_co2 = results_ds.RF_CO2.sel(ac=ac) * 1000
rf_h2o = results_ds.RF_H2O.sel(ac=ac) * 1000
dt_cont = results_ds.dT_cont.sel(ac=ac) * 1000
dt_co2 = results_ds.dT_CO2.sel(ac=ac) * 1000
dt_h2o = results_ds.dT_H2O.sel(ac=ac) * 1000

fig, ax = plt.subplots(ncols=2, figsize=(10,5))
ax[0].grid(True)
ax[1].grid(True)
rf_cont.plot(ax=ax[0], color="deepskyblue", label="cont")
rf_co2.plot(ax=ax[0], color="k", label="CO2")
rf_h2o.plot(ax=ax[0], color="steelblue", label="H2O")
dt_cont.plot(ax=ax[1], color="deepskyblue", label="cont")
dt_co2.plot(ax=ax[1], color="k", label="CO2")
dt_h2o.plot(ax=ax[1], color="steelblue", label="H2O")
ax[0].set_ylabel("Radiative Forcing [mW/m²]")
ax[1].set_ylabel("Temperature Change [mK]")
ax[0].legend()
ax[1].legend()
```
