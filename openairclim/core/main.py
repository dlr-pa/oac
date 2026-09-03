"""
main.py is the main interface to the submodules and the user script.
"""

import os
import sys
import time
import logging
import shutil
from pathlib import Path

# import OpenAirClim functions
from . import (
    read_config, read_netcdf, calc_co2, calc_ch4, plot
)

from .attribution import apply_attribution
from .calc_cont import check_cont_input, calc_contrails
from .calc_dt import calc_dtemp
from .calc_metric import calc_climate_metrics
from .calc_response import calc_resp_all, calc_resp_sub
from .construct_conc import get_emissions, interp_bg_conc
from .interpolate_time import adjust_inventories, apply_evolution
from .parametric import adapt_co2_emission, adapt_rf
from .utils import convert_nested_to_series
from .write_output import update_output_dict, write_output_dict_to_netcdf, write_climate_metrics



def run(file_name):
    """Runs OpenAirClim

    Args:
        file_name (str): Name of config file
    """

    # record start time
    start = time.time()

    # configure the logger
    logging.basicConfig(
        format="%(module)s ln. %(lineno)d in %(funcName)s %(levelname)s: %(message)s",
        level=logging.INFO,
        # TODO level=logging.DEBUG,
        handlers=[
            logging.FileHandler("debug.log", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Read the config file
    config = read_config.get_config(file_name)
    run_oac = config["output"]["run_oac"]
    output_conc = config["output"]["concentrations"]
    output_dir = config["output"]["dir"]
    parametric_enabled = config["parametric"]["enabled"]

    if run_oac:
        inv_species = config["species"]["inv"]
        # out_species = config["species"]["out"]
        species_0d, species_2d, species_cont, species_sub = read_config.classify_species(config)
        # Read emission inventories
        inv_dict = read_netcdf.open_inventories(config)
        # Adjust emission inventories to given time evolution
        inv_dict = adjust_inventories(config, inv_dict)
        # split inv_dict by aircraft identifiers defined in config
        full_inv_dict = read_netcdf.split_inventory_by_aircraft(config, inv_dict)

        # initialise loop over aircraft identifiers within full_inv_dict
        ac_lst = list(full_inv_dict.keys())
        output_dict = {ac: {} for ac in ac_lst}

        # calculate and save total emissions
        _, emis_dict = get_emissions(inv_dict, inv_species)
        _, emis_interp_dict = apply_evolution(
            config, emis_dict, inv_dict, inventories_adjusted=True
        )
        update_output_dict(output_dict, "TOTAL", "emis", emis_interp_dict)

        # calculate and save emissions for each aircraft identifier
        for ac in ac_lst:
            ac_inv_dict = full_inv_dict[ac]
            _, ac_emis_dict = get_emissions(ac_inv_dict, inv_species)
            _, ac_emis_interp_dict = apply_evolution(
                config, ac_emis_dict, inv_dict, inventories_adjusted=True
            )
            # parametric scenario: adapt CO2 emissions
            if parametric_enabled:
                ac_emis_interp_dict = adapt_co2_emission(
                    config, ac_emis_interp_dict
                )
            update_output_dict(output_dict, ac, "emis", ac_emis_interp_dict)

        # 0D species
        if species_0d:

            # CO2
            if "CO2" in species_0d:
                # calculate concentration of all aircraft identifiers
                emis_co2_dict = {"CO2": output_dict["TOTAL"]["emis_CO2"]}
                conc_co2_dict = calc_co2.calc_co2_concentration(config, emis_co2_dict)

                # calculate background concentration (diff to reference C_0)
                conc_co2_bg_dict = interp_bg_conc(config, "CO2")
                conc_co2_bg_dict["CO2"] -= calc_co2.CO2_0

                # calculate total+background concentration (for attribution)
                tot_conc_co2_dict = {
                    "CO2": conc_co2_dict["CO2"] + conc_co2_bg_dict["CO2"]
                }
                co2_att_method = config["responses"]["CO2"]["rf"]["attr"]

                # calculate concentrations and RF for each aircraft identifier
                for ac in ac_lst:
                    # CO2 concentration
                    ac_emis_co2_dict = {"CO2": output_dict[ac]["emis_CO2"]}
                    ac_conc_co2_dict = calc_co2.calc_co2_concentration(
                        config, ac_emis_co2_dict
                    )
                    update_output_dict(output_dict, ac, "conc", ac_conc_co2_dict)

                    # CO2 RF
                    ac_rf_co2_dict = apply_attribution(
                        calc_co2.calc_co2_rf,  # function to be attributed
                        calc_co2.calc_co2_drf_dconc,  # derivative of func
                        co2_att_method,  # attribution method
                        "CO2",  # species
                        ac_conc_co2_dict,  # sub_dict
                        tot_conc_co2_dict,  # total+bg concentration
                        config=config,  # kwargs
                    )
                    update_output_dict(output_dict, ac, "RF", ac_rf_co2_dict)

                    # CO2 dT
                    ac_dt_co2_dict = calc_dtemp(config, "CO2", ac_rf_co2_dict)
                    update_output_dict(output_dict, ac, "dT", ac_dt_co2_dict)

            else:
                logging.warning(
                    "Species CO2 is not set or response_grid option is not "
                    "set to 0D in config."
                )
        else:
            logging.warning("No species defined in config with 0D response grid.")

        # 2D species
        if species_2d:
            # Response: Emission --> Concentration
            if output_conc:
                # resp_conc_dict = oac.open_netcdf_from_config(
                #    config, "responses", species_2d, "conc"
                # )
                # conc_inv_years_dict = oac.calc_resp_all(
                #    config, resp_conc_dict, inv_dict
                # )
                # conc_series_dict = oac.convert_nested_to_series(
                #    conc_inv_years_dict
                # )
                # _time_range, conc_interp_dict = oac.apply_evolution(
                #    config, conc_series_dict, inv_dict, inventories_adjusted= True
                # )
                # conc_dict = oac.write_concentrations(
                #    config, resp_conc_dict, conc_interp_dict
                # )
                logging.warning(
                    "Computation of 2D concentration responses is not supported "
                    "in this version. Change output settings to: concentrations = false"
                )

            # Response: Emission --> Radiative Forcing
            species_rf, species_tau = read_config.classify_response_types(config, species_2d)

            if species_rf:
                resp_rf_dict = read_netcdf.open_netcdf_from_config(
                    config, "responses", species_rf, "rf"
                )
                # loop over aircraft identifiers and total
                for ac in ac_lst:
                    ac_inv_dict = full_inv_dict[ac]
                    rf_inv_years_dict = calc_resp_all(
                        config, resp_rf_dict, ac_inv_dict
                    )
                    rf_series_dict = convert_nested_to_series(rf_inv_years_dict)
                    _time_range, rf_interp_dict = apply_evolution(
                        config, rf_series_dict, inv_dict, inventories_adjusted=True
                    )
                    # parametric scenario: adapt RF
                    if parametric_enabled:
                        rf_interp_dict = adapt_rf(
                            config, rf_interp_dict, species_rf
                        )
                    update_output_dict(output_dict, ac, "RF", rf_interp_dict)
                    # RF --> dT
                    # Calculate temperature change
                    for spec in species_rf:
                        dtemp_dict = calc_dtemp(config, spec, rf_interp_dict)
                        update_output_dict(output_dict, ac, "dT", dtemp_dict)

            if species_tau:
                resp_tau_dict = read_netcdf.open_netcdf_from_config(
                    config, "responses", ["CH4"], "tau"
                )

                # calculate concentration of all aircraft identifiers together
                tau_inverse_dict = calc_resp_all(config, resp_tau_dict, inv_dict)
                tau_inverse_series_dict = convert_nested_to_series(tau_inverse_dict)
                _, tau_inverse_interp_dict = apply_evolution(
                    config,
                    tau_inverse_series_dict,
                    inv_dict,
                    inventories_adjusted=True,
                )
                conc_ch4_dict = calc_ch4.calc_ch4_concentration(
                    config, tau_inverse_interp_dict
                )

                # calculate background concentration (diff to reference M_0)
                conc_ch4_bg_dict = interp_bg_conc(config, "CH4")
                conc_ch4_bg_dict["CH4"] -= calc_ch4.CH4_0

                # calculate total+background concentration (for attribution)
                tot_conc_ch4_dict = {
                    "CH4": conc_ch4_dict["CH4"] + conc_ch4_bg_dict["CH4"]
                }
                ch4_att_method = config["responses"]["CH4"]["rf"]["attr"]

                # calculate concentrations and RF for each aircraft identifier
                for ac in ac_lst:
                    ac_inv_dict = full_inv_dict[ac]
                    ac_tau_inverse_dict = calc_resp_all(
                        config, resp_tau_dict, ac_inv_dict
                    )
                    ac_tau_inverse_series_dict = convert_nested_to_series(
                        ac_tau_inverse_dict
                    )
                    _, ac_tau_inverse_interp_dict = apply_evolution(
                        config,
                        ac_tau_inverse_series_dict,
                        inv_dict,
                        inventories_adjusted=True,
                    )
                    ac_conc_ch4_dict = calc_ch4.calc_ch4_concentration(
                        config, ac_tau_inverse_interp_dict
                    )
                    update_output_dict(output_dict, ac, "conc", ac_conc_ch4_dict)

                    # CH4 RF
                    ac_rf_ch4_dict = apply_attribution(
                        calc_ch4.calc_ch4_rf,
                        calc_ch4.calc_ch4_drf_dconc,
                        ch4_att_method,
                        "CH4",
                        ac_conc_ch4_dict,
                        tot_conc_ch4_dict,
                        config=config,
                    )
                    # parametric scenario: adapt RF
                    if parametric_enabled:
                        ac_rf_ch4_dict = adapt_rf(
                            config, ac_rf_ch4_dict, species_tau
                        )
                    update_output_dict(output_dict, ac, "RF", ac_rf_ch4_dict)

                    # CH4 dT
                    ac_dt_ch4_dict = calc_dtemp(config, "CH4", ac_rf_ch4_dict)
                    update_output_dict(output_dict, ac, "dT", ac_dt_ch4_dict)

                # give warning until validation is complete
                logging.warning("CH4 response surface is not validated!")

        else:
            logging.warning("No species defined in config with 2D response_grid.")

        if species_sub:
            for ac in ac_lst:
                rf_sub_dict, conc_sub_dict = calc_resp_sub(
                    species_sub, output_dict, ac
                )
                # parametric scenario: adapt RF
                if parametric_enabled:
                    rf_sub_dict = adapt_rf(config, rf_sub_dict, species_sub)
                update_output_dict(output_dict, ac, "RF", rf_sub_dict)
                update_output_dict(output_dict, ac, "conc", conc_sub_dict)
                # RF --> dT
                # Calculate temperature change
                for spec in species_sub:
                    dtemp_dict = calc_dtemp(config, spec, rf_sub_dict)
                    update_output_dict(output_dict, ac, "dT", dtemp_dict)
        else:
            logging.info("No subsequent species (PMO) defined in config.")


        if species_cont:

            # load contrail data
            ds_cont = read_netcdf.open_netcdf_from_config(
                config, "responses", ["cont"], "resp"
            )["cont"]

            # check contrail input
            check_cont_input(ds_cont)

            # calculate contrail RF
            rf_cont_dict = calc_contrails(
                ac_lst, config, inv_dict, full_inv_dict, ds_cont
            )

            # calculate contrail dT and save to output_dict
            for ac in ac_lst:
                ac_rf_cont_dict = {"cont": rf_cont_dict[ac]}

                # parametric scenario: adapt RF
                if parametric_enabled:
                    ac_rf_cont_dict = adapt_rf(config, ac_rf_cont_dict, species_cont)

                # update output_dict
                update_output_dict(output_dict, ac, "RF", ac_rf_cont_dict)

                # calculate temperature change
                dtemp_cont_dict = calc_dtemp(config, "cont", ac_rf_cont_dict)
                update_output_dict(output_dict, ac, "dT", dtemp_cont_dict)

        else:
            logging.warning("No contrails defined in config.")


    # save results
    write_output_dict_to_netcdf(config, output_dict, mode="w")

    # Calculate climate metrics
    run_metrics = config["output"]["run_metrics"]
    if run_metrics:
        metrics_dict = calc_climate_metrics(config)
        write_climate_metrics(config, metrics_dict)

    # Record end time
    end = time.time()
    # Execution time is difference between start and end time
    msg = "Execution time: " + str(end - start) + " sec"
    logging.info(msg)

    # WARNING message: demonstrating purposes
    logging.warning(
        "OpenAirClim is currently in development phase.\n"
        "The computed output is not for scientific purposes "
        "until release of our publication.\n"
        "Amongst others, the climate impact of longer species lifetimes "
        "in the stratosphere is not considered."
    )

    # PLOTS
    run_plots = config["output"]["run_plots"]
    if run_plots:
        # Plot vertical profiles of inventories
        plot.plot_inventory_vertical_profiles(inv_dict, output_dir)

        # Plot results
        output_name = config["output"]["name"]
        output_file = Path(output_dir) / f"{output_name}.nc"
        result_dic = read_netcdf.open_netcdf(output_file)
        plot.plot_results(config, result_dic, marker="o")

    # clean up: close all logger handlers
    logger = logging.getLogger()
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

    # move config and log files to results folder
    shutil.copy2(file_name, f"{output_dir}")
    if os.path.exists(f"{output_dir}/debug.log"):
        os.remove(f"{output_dir}/debug.log")
    shutil.move("debug.log", f"{output_dir}")
