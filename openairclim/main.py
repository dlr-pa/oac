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
        species_0d, species_2d, species_cont = oac.classify_species(config)
        inv_dict = oac.open_inventories(config)

        # Emissions in Tg, each species
        emis_dict = oac.get_emissions(inv_dict, inv_species)
        _time_range, emis_interp_dict = oac.apply_evolution(
            config, emis_dict, inv_dict
        )
        oac.write_to_netcdf(
            config, emis_interp_dict, result_type="emis", mode="w"
        )

        # Initialize dictionaries
        dtemp_dict = {}

        if species_0d:
            # Emissions in Tg
            emis_dict = oac.get_emissions(inv_dict, species_0d)
            # Get CO2 emissions from inventories in Tg CO2
            # emis_co2_dict = {"CO2": emis_dict["CO2"]}
            # Apply time evolution
            _time_range, emis_interp_dict = oac.apply_evolution(
                config, emis_dict, inv_dict
            )
            if "CO2" in species_0d:
                # Calculate concentrations
                conc_co2_dict = oac.calc_co2_concentration(
                    config, emis_interp_dict
                )
                oac.write_to_netcdf(
                    config, conc_co2_dict, result_type="conc", mode="a"
                )
                # Get background concentration
                conc_co2_bg_dict = oac.interp_bg_conc(config, "CO2")
                # Calculate Radiative Forcing
                rf_co2_dict = oac.calc_co2_rf(
                    config, conc_co2_dict, conc_co2_bg_dict
                )
                oac.write_to_netcdf(
                    config, rf_co2_dict, result_type="RF", mode="a"
                )
                # Calculate temperature change
                dtemp_co2_dict = oac.calc_dtemp(config, "CO2", rf_co2_dict)
                oac.write_to_netcdf(
                    config, dtemp_co2_dict, result_type="dT", mode="a"
                )
            else:
                logging.warning(
                    "Species CO2 is not set or response_grid option is not set to 0D in config."
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
                #    config, conc_series_dict, inv_dict
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
                    config, resp_rf_dict, inv_dict
                )
                rf_series_dict = oac.convert_nested_to_series(
                    rf_inv_years_dict
                )
                _time_range, rf_interp_dict = oac.apply_evolution(
                    config, rf_series_dict, inv_dict
                )
                oac.write_to_netcdf(
                    config, rf_interp_dict, result_type="RF", mode="a"
                )
                # RF --> dT
                # Calculate temperature change
                for spec in species_rf:
                    dtemp_dict = oac.calc_dtemp(config, spec, rf_interp_dict)
                    oac.write_to_netcdf(
                        config, dtemp_dict, result_type="dT", mode="a"
                    )
            if species_tau:
                resp_tau_dict = oac.open_netcdf_from_config(
                    config, "responses", ["CH4"], "tau"
                )
                tau_inverse_dict = oac.calc_resp_all(
                    config, resp_tau_dict, inv_dict
                )
                tau_inverse_series_dict = oac.convert_nested_to_series(
                    tau_inverse_dict
                )
                _time_range, tau_inverse_interp_dict = oac.apply_evolution(
                    config, tau_inverse_series_dict, inv_dict
                )
                conc_ch4_dict = oac.calc_ch4_concentration(
                    config, tau_inverse_interp_dict
                )
                oac.write_to_netcdf(
                    config, conc_ch4_dict, result_type="conc", mode="a"
                )
                # Get background concentrations
                conc_ch4_bg_dict = oac.interp_bg_conc(config, "CH4")
                conc_n2o_bg_dict = oac.interp_bg_conc(config, "N2O")
                # Calculate Radiative Forcing
                rf_ch4_dict = oac.calc_ch4_rf(
                    config, conc_ch4_dict, conc_ch4_bg_dict, conc_n2o_bg_dict
                )
                oac.write_to_netcdf(
                    config, rf_ch4_dict, result_type="RF", mode="a"
                )
                # Calculate temperature change
                dtemp_ch4_dict = oac.calc_dtemp(config, "CH4", rf_ch4_dict)
                oac.write_to_netcdf(
                    config, dtemp_ch4_dict, result_type="dT", mode="a"
                )
                logging.warning(
                    "Computed values for CH4 response are not scientifially meaningful. "
                    "CH4 response surface without noise correction!"
                )
        else:
            logging.warning(
                "No species defined in config with 2D response_grid."
            )
    
        if species_cont:
            # Calculate Contrail Flight Distance Density (CFDD)
            cfdd_dict = oac.calc_cfdd(config, inv_dict)
            
            # Calculate contrail cirrus coverage (cccov)
            cccov_dict = oac.calc_cccov(config, cfdd_dict)
            
            # Calculate global, area-weighted cccov
            cccov_tot_dict = oac.calc_cccov_tot(config, cccov_dict)
            
            # Calculate contrail RF
            rf_cont_dict = oac.calc_cont_RF(config, cccov_tot_dict, inv_dict)
            oac.write_to_netcdf(
                config, rf_cont_dict, result_type="RF", mode="a"
            )
            
            # Calculate contrail temperature change
            dtemp_cont_dict = oac.calc_dtemp(config, "cont", rf_cont_dict)
            oac.write_to_netcdf(
                config, dtemp_cont_dict, result_type="dT", mode="a"
            )
            logging.warning(
                "Contrail values use the AirClim 2.1 method."
            )
        else:
            logging.warning(
                "No contrails defined in config."
            )
            
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
