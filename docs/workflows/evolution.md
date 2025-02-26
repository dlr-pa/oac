# Time evolution

In OpenAirClim, there exist three types of possible predefined time evolutions:

1. **Normalization** by given time evolution file via fuel use and emission indices.
2. **Scaling** by given time evolution file, scaling/multiplication of all data variables.
3. **No** evolution file given, temporal interpolation between discrete inventory years.

The preferred way of applying any evolution type is on the inventory data, i.e. before calculating the responses. Alternatively, time evolution can be applied after calculating the responses. However, this is only recommended for linear responses. In the following, the application of time evolution on inventory data is described.

Depending on the user settings in the configuration file, the function `adjust_inventories(config, inv_dict)` applies one of the possible evolution types on the **inventories**. The corresponding settings have to be made within the section (respectively *hash table*) `[time]` in the configuration file.

## Normalization

If a valid time evolution file is set in the configuration, OpenAirClim reads in the netCDF file and checks the global attribute section (metadata). If the key-value pair `Type: norm` is found in the global attributes, the function `norm_inventories(config, inv_dict)` is executed.

The general idea behind the normalization routine is that the driving parameters (fuel use, emission indices and flown distance per fuel use) come from the evolution file. The emission inventories are **normalized** accordingly.

![norm_inventories](../img/norm_inventories.png)

The figure above illustrates the workflow of the normalization routine. First, the evolution data gets processed. The data comprises fuel uses and optionally emission indices for the emitted species and flown distance per fuel use. The evolution data variables are interpolated via function `interp_evolution(config)` to the time range defined in the configuration. This is necessary since the time steps for time range and for evolution data can be different.

Then, the function `calc_inv_quantities(config, inv_dict)` is applied on the dictionary of input inventories `inv_dict`. It outputs the inventory years, the inventory species sums and emission indicies. For further processing, only the emission indices are used which are also available in the evolution data. Therefore, the function `filter_dict_to_evo_keys(config, ei_inv_dict)` filters the emission indices dictionary which have been computed from the inventories.

Since we are interested in adjusting inventories, evolution data is only needed for inventory years. This is done by function `filter_to_inv_years(inv_years, time_range, evo_interp_dict)` (left vertical workflow in the figure).

The filtered fuel use and emission indices from the evolution data together with the computed fuel sums and emission indices from the inventories (right vertical workflow in the figure) are processed within function `calc_norm(evo_filtered_dict, ei_inv_dict)` for calculation of normalization multipliers. Finally, function `norm_inv(inv_dict, norm_dict)` applies these multipliers on the inventory data variables (fuel use, species emissions and distance) in order to create normalized emission inventories `out_inv_dict`.

### Time constraints

Following figure illustrates the time constraints for the evolution type *norm*. Three different time axes have to be considered: `time_range` from the settings in the configuration file, the discrete inventory years `inv_years` from the emission inventories, and the time coordinate `evolution_time` from the evolution file.

![constraints_norm](../img/time-constraints_norm.png)

Following time constraints have to be met for a valid configuration:

- `time_range` must be within `evolution_time`
- At least one `inv_year` must be within `time_range`
- At least one `inv_year` must be within `evolution_time`

Following inventories are considered during simulation:

- `inv_years` <mark>overlapping</mark> with `time_range`


## Scaling

If a valid time evolution file is set in the configuration, OpenAirClim reads in the netCDF file and checks the global attribute section (metadata). If the key-value pair `Type: scaling` is found in the global attributes, the function `scale_inventories(config, inv_dict)` is executed.

The general idea behind the scaling routine is that the parameters in the evolution file are multipliers. The emission inventories are **scaled** accordingly. This routine is useful for the consideration of changes relative to the emission inventories.

![scale_inventories](../img/scale_inventories.png)

The figure above illustrates the workflow for the scaling routine. First, the evolution data gets processed. In the case of a evolution file of type *scaling*, evolution data comprises a time series of scaling factors. In the current implementation, the scaling factors apply equally to all inventory data variables (fuel use, species emissions and flown distance).

The evolution data variables are interpolated via function `interp_evolution(config)` to the time range defined in the configuration. This is necessary since the time steps for time range and for evolution data can be different.

Since we are interested in adjusting inventories, evolution data is only needed for inventory years. This is done by function `filter_to_inv_years(inv_years, time_range, evo_interp_dict)`. Finally, function `scale_inv(inv_dict, evo_filtered_dict)` performs the actual scaling by multiplying inventory data variables by the scaling factors.

### Time constraints

Following figure illustrates the time constraints for the evolution type *scaling*. Three different time axes have to be considered: `time_range` from the settings in the configuration file, the discrete inventory years `inv_years` from the emission inventories, and the time coordinate `evolution_time` from the evolution file.

![constraints_norm](../img/time-constraints_scaling.png)

Following time constraints have to be met for a valid configuration:

- `time_range` must be within `evolution_time`
- At least one `inv_year` must be within `time_range`
- At least one `inv_year` must be within `evolution_time`
- `time_range` first and last year <mark>must</mark> be inventory years

Following inventories are considered during simulation:

- `inv_years` <mark>overlapping</mark> with `time_range`


## No evolution

If no evolution file is set in the configuration, **no evolution** is applied, i.e. the inventory output dictionary `out_inv_dict` and the input dictionary `inv_dict` are the same.

### Time constraints

Following figure illustrates the time constraints for the evolution type *no evolution*. Two different time axes have to be considered: `time_range` from the settings in the configuration file, and the discrete inventory years `inv_years` from the emission inventories.

![constraints_norm](../img/time-constraints_no-evolution.png)

Following time constraints have to be met for a valid configuration:

- At least one `inv_year` must be within `time_range`

Following inventories are considered during simulation:

- `inv_years` <mark>overlapping</mark> with `time_range`

<!---
Following sentence is relevant for apply_evoltion() routine applied on already calculated responses:

For `time_range` outside the `inv_years` sequence, OpenAirClim assumes fill values of 0.0, i.e. no emissions are considered for these periods! A warning is output to the user to ensure that this setting is not made unintentionally.
-->