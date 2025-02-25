# Time evolution

In OpenAirClim, there exist three types to consider predefined time evolutions:

1. **Normalization** by given time evolution file via fuel use and emission indices.
2. **Scaling** by given time evolution file, scaling/multiplication of all data variables.
3. **No** evolution file given, temporal interpolation between discrete inventory years.

The preferred way of applying any evolution type is on the inventory data, i.e. before calculating the responses. Alternatively, time evolution can be applied after calculating the responses. However, this is only recommended for linear responses. In the following, the application of time evolution on inventory data is described.

Depending on the user settings in the configuration file, the function `adjust_inventories(config, inv_dict)` applies one of the possible evolution types on the inventories.

## Normalization

![norm_inventories](../img/norm_inventories.png)

## Scaling

![scale_inventories](../img/scale_inventories.png)

## No evolution