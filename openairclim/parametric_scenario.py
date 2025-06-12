"""Set of functions used to adapt emissions of CO2 and RF of other species."""

import numpy as np
import openairclim as oac
import logging

# Default values for parametric factors, from: https://doi.org/10.5194/gmd-17-4031-2024

RATIO_DIC_D = {"CO2":1.0812, "O3": 0.9404, "CH4": 1.2017, "H2O": 0.9339, "cont": 0.7133, "PMO": 1}


def process_parametric_config(config):

    enabled      = bool(config.get("parametric", {}).get("enabled"))
    mode         = config.get("parametric", {}).get("mode")

    if enabled: 
        if mode == "incremental":
            mode="incremental"

        elif mode == "direct": 
            mode="direct"

        else: 
            mode = "direct"
            msg = "Unkown mode in parametric module. Default mode: 'direct' activated"
            logging.warning(msg)

    time_horizon = config.get("parametric", {}).get("H")

    if time_horizon is None or int(time_horizon) < 0:
        msg = "Invalid or missing time_horizon. Default time horizon (20 years) selected instead."
        logging.warning(msg)
        time_horizon = 20
    else: 
        time_horizon = int(time_horizon)

    return enabled, mode, time_horizon

def get_factor(config, spec):

    factor = config.get("parametric", {}).get(spec)

    if factor is None or float(factor) <0: 
        logging.info(f"Invalid or missing {spec} parametric factor. Using default value.")

        factor = RATIO_DIC_D[spec]
    else: 
        factor = float(factor)

    return factor



def adapt_co2_emission(config, emis_interp_dict)-> tuple[np.ndarray, dict]:


    time_config = config["time"]["range"]
    time_range = np.arange(
        time_config[0], time_config[1], time_config[2], dtype=int
    )

    enabled, mode, time_horizon = process_parametric_config(config)


    if enabled: 

        if mode == "direct":

            emis_interp_dict["CO2"]=emis_interp_dict["CO2"]*get_factor(config,"CO2")

        elif mode == "incremental":

            n_steps = np.size(emis_interp_dict["CO2"])
            emis_interp_dict["CO2"]=emis_interp_dict["CO2"]*ratio_change(get_factor(config,"CO2"),n_steps,time_horizon)

        else: 
            raise ValueError(
                "Unkown mode"
            )



    return time_range, emis_interp_dict


def adapt_rf(config, rf_interp_dict, species_list):

    time_config = config["time"]["range"]
    time_range = np.arange(
        time_config[0], time_config[1], time_config[2], dtype=int
    )

    enabled, mode, time_horizon = process_parametric_config(config)


    for spec in species_list: 
        if enabled: 

            if mode == "direct":

                rf_interp_dict[spec] = rf_interp_dict[spec]*get_factor(config,spec)

            elif mode == "incremental":

                n_steps = np.size(rf_interp_dict[spec])
                rf_interp_dict[spec] = rf_interp_dict[spec]*ratio_change(get_factor(config,spec),n_steps,time_horizon)

            else: 
                raise ValueError(
                    "Unkown mode"
                )

    return time_config, rf_interp_dict


def ratio_change(final_fue, total_steps, transition_years):

    transition = np.linspace(1, final_fue, transition_years)
    remaining = np.full(total_steps - transition_years, final_fue)
    return np.concatenate([transition, remaining])