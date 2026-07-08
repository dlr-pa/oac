# Contains the uncertainty factors per iteration in case the sensitivity analysis module is run.
Uncertainty_Factor = {}

def get_uncertainty(key):
    """
    Get the uncertainty factor for the current iteration. Return 1.0 if 
    no sensitivity analysis is performed.

    Args:
        key (string): The key of the uncertainty factor in the Uncertainty_Factor dictionary.
        
    Returns:
        Uncertainty factor corresponding to key.
    """
    factor = Uncertainty_Factor.get(key, 1.0)
    if key == "cont Radiative Forcing" and factor < 0.0:
        factor = 0.0
        print("cont Radiative Forcing clipped to 0.0.")
    if factor == 1.0 and key != "CO2 Efficacy":
        print("UNCERTAINTY ANALYSIS WARNING: No key named '" + key + "' was detected. Defaulted to 1.0.")
    return factor