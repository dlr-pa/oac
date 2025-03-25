"""
main.py is the main interface to the submodules and the user script.
"""

import sys
import time
import logging
import openairclim as oac


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
    config = oac.get_config(file_name)
    full_run = config["output"]["full_run"]
    output_conc = config["output"]["concentrations"]
    if full_run:
        inv_species = config["species"]["inv"]
        # out_species = config["species"]["out"]
        species_0d, species_2d, species_cont, species_sub = (
            oac.classify_species(config)
        )
        # Read emission inventories
        inv_dict = oac.open_inventories(config)
        # Adjust emission inventories to given time evolution
        inv_dict = oac.adjust_inventories(config, inv_dict)
        # split inv_dict by aircraft identifiers defined in config
        full_inv_dict = oac.split_inventory_by_aircraft(config, inv_dict)

        # initialise loop over aircraft identifiers within config
        ac_lst = config["aircraft"]["types"]
        output_dict = {ac: {} for ac in ac_lst}
        for ac in ac_lst:
            # calculate and save emissions for each aircraft identifier
            ac_inv_dict = full_inv_dict[ac]
            ac_inv_dict = full_inv_dict[ac]
            inv_years, emis_dict = oac.get_emissions(inv_dict, inv_species)
            _time_range, emis_interp_dict = oac.apply_evolution(
                config, emis_dict, ac_inv_dict, inventories_adjusted=True
            )
            oac.update_output_dict(output_dict, ac, "emis", emis_interp_dict)

            if species_0d:
                # Emissions in Tg
                _inv_years, emis_dict = oac.get_emissions(ac_inv_dict, species_0d)
                # Get CO2 emissions from inventories in Tg CO2
                # emis_co2_dict = {"CO2": emis_dict["CO2"]}
                # Apply time evolution
                _time_range, emis_interp_dict = oac.apply_evolution(
                    config, emis_dict, ac_inv_dict, inventories_adjusted=True
                )
                if "CO2" in species_0d:
                    # Calculate concentrations
                    conc_co2_dict = oac.calc_co2_concentration(
                        config, emis_interp_dict
                    )
                    oac.update_output_dict(output_dict, ac, "conc", conc_co2_dict)
                    # Get background concentration
                    conc_co2_bg_dict = oac.interp_bg_conc(config, "CO2")
                    # Calculate Radiative Forcing
                    rf_co2_dict = oac.calc_co2_rf(
                        config, conc_co2_dict, conc_co2_bg_dict
                    )
                    oac.update_output_dict(output_dict, ac, "RF", rf_co2_dict)
                    # Calculate temperature change
                    dtemp_co2_dict = oac.calc_dtemp(config, "CO2", rf_co2_dict)
                    oac.update_output_dict(output_dict, ac, "dT", dtemp_co2_dict)
                else:
                    logging.warning(
                        "Species CO2 is not set or response_grid option is not "
                        "set to 0D in config."
                    )
            else:
                logging.warning(
                    "No species defined in config with 0D response_grid."
                )


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
                species_rf, species_tau = oac.classify_response_types(
                    config, species_2d
                )
                if species_rf:
                    resp_rf_dict = oac.open_netcdf_from_config(
                        config, "responses", species_rf, "rf"
                    )
                    rf_inv_years_dict = oac.calc_resp_all(
                        config, resp_rf_dict, ac_inv_dict
                    )
                    rf_series_dict = oac.convert_nested_to_series(
                        rf_inv_years_dict
                    )
                    _time_range, rf_interp_dict = oac.apply_evolution(
                        config, rf_series_dict, ac_inv_dict, inventories_adjusted=True
                    )
                    oac.update_output_dict(output_dict, ac, "RF", rf_interp_dict)
                    # RF --> dT
                    # Calculate temperature change
                    for spec in species_rf:
                        dtemp_dict = oac.calc_dtemp(config, spec, rf_interp_dict)
                        oac.update_output_dict(output_dict, ac, "dT", dtemp_dict)
                if species_tau:
                    resp_tau_dict = oac.open_netcdf_from_config(
                        config, "responses", ["CH4"], "tau"
                    )
                    tau_inverse_dict = oac.calc_resp_all(
                        config, resp_tau_dict, ac_inv_dict
                    )
                    tau_inverse_series_dict = oac.convert_nested_to_series(
                        tau_inverse_dict
                    )
                    _time_range, tau_inverse_interp_dict = oac.apply_evolution(
                        config,
                        tau_inverse_series_dict,
                        ac_inv_dict,
                        inventories_adjusted=True,
                    )
                    conc_ch4_dict = oac.calc_ch4_concentration(
                        config, tau_inverse_interp_dict
                    )
                    oac.update_output_dict(output_dict, ac, "conc", conc_ch4_dict)
                    # Get background concentrations
                    conc_ch4_bg_dict = oac.interp_bg_conc(config, "CH4")
                    conc_n2o_bg_dict = oac.interp_bg_conc(config, "N2O")
                    # Calculate Radiative Forcing
                    rf_ch4_dict = oac.calc_ch4_rf(
                        config, conc_ch4_dict, conc_ch4_bg_dict, conc_n2o_bg_dict
                    )
                    oac.update_output_dict(output_dict, ac, "RF", rf_ch4_dict)
                    # Calculate temperature change
                    dtemp_ch4_dict = oac.calc_dtemp(config, "CH4", rf_ch4_dict)
                    oac.update_output_dict(output_dict, ac, "dT", dtemp_ch4_dict)
                    logging.warning("CH4 response surface is not validated!")
            else:
                logging.warning(
                    "No species defined in config with 2D response_grid."
                )


            if species_cont:
                logging.warning(
                    "Contrails are currently not implemented for multiple aircraft."
                    )


            if species_sub:
                rf_sub_dict = oac.calc_resp_sub(species_sub, output_dict, ac)
                oac.update_output_dict(output_dict, ac, "RF", rf_sub_dict)
                # RF --> dT
                # Calculate temperature change
                for spec in species_sub:
                    dtemp_dict = oac.calc_dtemp(config, spec, rf_sub_dict)
                    oac.update_output_dict(output_dict, ac, "dT", dtemp_dict)
            else:
                logging.info("No subsequent species (PMO) defined in config.")

    # save results
    oac.write_output_dict_to_netcdf(config, output_dict, mode="w")

    # Calculate climate metrics
    metrics_dict = oac.calc_climate_metrics(config)
    oac.write_climate_metrics(config, metrics_dict)

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
    # Plot vertical profiles of inventories
    oac.plot_inventory_vertical_profiles(inv_dict)

    # Plot results
    output_dir = config["output"]["dir"]
    output_name = config["output"]["name"]
    output_file = output_dir + output_name + ".nc"
    result_dic = oac.open_netcdf(output_file)
    oac.plot_results(config, result_dic, marker="o")
    # Create 2D concentration plots
    # if output_conc and full_run:
    #    for spec in species_2d:
    #        oac.plot_concentrations(config, spec, conc_dict)
