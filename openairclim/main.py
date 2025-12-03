"""
main.py is the main interface to the submodules and the user script.
"""

import os
import sys
import time
import logging
import shutil
import numpy as np
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
    run_oac = config["output"]["run_oac"]
    output_conc = config["output"]["concentrations"]
    output_dir = config["output"]["dir"]
    if run_oac:
        inv_species = config["species"]["inv"]
        # out_species = config["species"]["out"]
        species_0d, species_2d, species_cont, species_sub = oac.classify_species(config)
        # Read emission inventories
        inv_dict = oac.open_inventories(config)
        # Adjust emission inventories to given time evolution
        inv_dict = oac.adjust_inventories(config, inv_dict)
        # split inv_dict by aircraft identifiers defined in config
        full_inv_dict = oac.split_inventory_by_aircraft(config, inv_dict)

        # initialise loop over aircraft identifiers within full_inv_dict
        ac_lst = list(full_inv_dict.keys())
        output_dict = {ac: {} for ac in ac_lst}

        # calculate and save total emissions
        _, emis_dict = oac.get_emissions(inv_dict, inv_species)
        _, emis_interp_dict = oac.apply_evolution(
            config, emis_dict, inv_dict, inventories_adjusted=True
        )
        oac.update_output_dict(output_dict, "TOTAL", "emis", emis_interp_dict)

        # calculate and save emissions for each aircraft identifier
        for ac in ac_lst:
            ac_inv_dict = full_inv_dict[ac]
            _, ac_emis_dict = oac.get_emissions(ac_inv_dict, inv_species)
            _, ac_emis_interp_dict = oac.apply_evolution(
                config, ac_emis_dict, inv_dict, inventories_adjusted=True
            )
            oac.update_output_dict(output_dict, ac, "emis", ac_emis_interp_dict)

        # 0D species
        if species_0d:

            # CO2
            if "CO2" in species_0d:
                # calculate concentration of all aircraft identifiers
                emis_co2_dict = {"CO2": output_dict["TOTAL"]["emis_CO2"]}
                conc_co2_dict = oac.calc_co2_concentration(config, emis_co2_dict)

                # calculate background concentration (diff to reference C_0)
                conc_co2_bg_dict = oac.interp_bg_conc(config, "CO2")
                conc_co2_bg_dict["CO2"] -= oac.CO2_0

                # calculate total+background concentration (for attribution)
                tot_conc_co2_dict = {
                    "CO2": conc_co2_dict["CO2"] + conc_co2_bg_dict["CO2"]
                }
                co2_att_method = config["responses"]["CO2"]["rf"]["attr"]

                # calculate concentrations and RF for each aircraft identifier
                for ac in ac_lst:
                    # CO2 concentration
                    ac_emis_co2_dict = {"CO2": output_dict[ac]["emis_CO2"]}
                    ac_conc_co2_dict = oac.calc_co2_concentration(
                        config, ac_emis_co2_dict
                    )
                    oac.update_output_dict(output_dict, ac, "conc", ac_conc_co2_dict)

                    # CO2 RF
                    ac_rf_co2_dict = oac.apply_attribution(
                        oac.calc_co2_rf,  # function to be attributed
                        oac.calc_co2_drf_dconc,  # derivative of func
                        co2_att_method,  # attribution method
                        "CO2",  # species
                        ac_conc_co2_dict,  # sub_dict
                        tot_conc_co2_dict,  # total+bg concentration
                        config=config,  # kwargs
                    )
                    oac.update_output_dict(output_dict, ac, "RF", ac_rf_co2_dict)

                    # CO2 dT
                    ac_dt_co2_dict = oac.calc_dtemp(config, "CO2", ac_rf_co2_dict)
                    oac.update_output_dict(output_dict, ac, "dT", ac_dt_co2_dict)

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
            species_rf, species_tau = oac.classify_response_types(config, species_2d)

            if species_rf:
                resp_rf_dict = oac.open_netcdf_from_config(
                    config, "responses", species_rf, "rf"
                )
                # loop over aircraft identifiers and total
                for ac in ac_lst:
                    ac_inv_dict = full_inv_dict[ac]
                    rf_inv_years_dict = oac.calc_resp_all(
                        config, resp_rf_dict, ac_inv_dict
                    )
                    rf_series_dict = oac.convert_nested_to_series(rf_inv_years_dict)
                    _time_range, rf_interp_dict = oac.apply_evolution(
                        config, rf_series_dict, inv_dict, inventories_adjusted=True
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

                # calculate concentration of all aircraft identifiers together
                tau_inverse_dict = oac.calc_resp_all(config, resp_tau_dict, inv_dict)
                tau_inverse_series_dict = oac.convert_nested_to_series(tau_inverse_dict)
                _, tau_inverse_interp_dict = oac.apply_evolution(
                    config,
                    tau_inverse_series_dict,
                    inv_dict,
                    inventories_adjusted=True,
                )
                conc_ch4_dict = oac.calc_ch4_concentration(
                    config, tau_inverse_interp_dict
                )

                # calculate background concentration (diff to reference M_0)
                conc_ch4_bg_dict = oac.interp_bg_conc(config, "CH4")
                conc_ch4_bg_dict["CH4"] -= oac.CH4_0

                # calculate total+background concentration (for attribution)
                tot_conc_ch4_dict = {
                    "CH4": conc_ch4_dict["CH4"] + conc_ch4_bg_dict["CH4"]
                }
                ch4_att_method = config["responses"]["CH4"]["rf"]["attr"]

                # calculate concentrations and RF for each aircraft identifier
                for ac in ac_lst:
                    ac_inv_dict = full_inv_dict[ac]
                    ac_tau_inverse_dict = oac.calc_resp_all(
                        config, resp_tau_dict, ac_inv_dict
                    )
                    ac_tau_inverse_series_dict = oac.convert_nested_to_series(
                        ac_tau_inverse_dict
                    )
                    _, ac_tau_inverse_interp_dict = oac.apply_evolution(
                        config,
                        ac_tau_inverse_series_dict,
                        inv_dict,
                        inventories_adjusted=True,
                    )
                    ac_conc_ch4_dict = oac.calc_ch4_concentration(
                        config, ac_tau_inverse_interp_dict
                    )
                    oac.update_output_dict(output_dict, ac, "conc", ac_conc_ch4_dict)

                    # CH4 RF
                    ac_rf_ch4_dict = oac.apply_attribution(
                        oac.calc_ch4_rf,
                        oac.calc_ch4_drf_dconc,
                        ch4_att_method,
                        "CH4",
                        ac_conc_ch4_dict,
                        tot_conc_ch4_dict,
                        config=config,
                    )
                    oac.update_output_dict(output_dict, ac, "RF", ac_rf_ch4_dict)

                    # CH4 dT
                    ac_dt_ch4_dict = oac.calc_dtemp(config, "CH4", ac_rf_ch4_dict)
                    oac.update_output_dict(output_dict, ac, "dT", ac_dt_ch4_dict)

                # give warning until validation is complete
                logging.warning("CH4 response surface is not validated!")

        else:
            logging.warning("No species defined in config with 2D response_grid.")

        if species_sub:
            logging.warning("PMO response not validated!")
            for ac in ac_lst + ["TOTAL"]:
                rf_sub_dict = oac.calc_resp_sub(species_sub, output_dict, ac)
                oac.update_output_dict(output_dict, ac, "RF", rf_sub_dict)
                # RF --> dT
                # Calculate temperature change
                for spec in species_sub:
                    dtemp_dict = oac.calc_dtemp(config, spec, rf_sub_dict)
                    oac.update_output_dict(output_dict, ac, "dT", dtemp_dict)
        else:
            logging.info("No subsequent species (PMO) defined in config.")


        if species_cont:

            # define ac_lst without "TOTAL"
            ac_no_tot = [ac for ac in ac_lst if ac != "TOTAL"]

            # load contrail data
            ds_cont = oac.open_netcdf_from_config(
                config, "responses", ["cont"], "resp"
            )["cont"]

            # get contrail grid
            cont_grid = oac.get_cont_grid(ds_cont)

            # if necessary, add zero arrays if aircraft not defined in certain
            # inventory years
            inv_yrs = inv_dict.keys()
            for ac in ac_lst:
                full_inv_dict[ac] = oac.pad_inv_dict(
                    inv_yrs, full_inv_dict[ac], ["distance"], cont_grid, ac
                )

            # load base inventories if rel_to_base is TRUE
            if config["inventories"]["rel_to_base"]:
                full_base_inv_dict = oac.load_base_inventories(
                    config, inv_yrs, cont_grid
                )
                base_ac_lst = list(full_base_inv_dict.keys())
                base_ac_lst = [bac for bac in base_ac_lst if bac != "BASE_TOTAL"]

                # copy ac config settings to base_ac
                for base_ac in base_ac_lst:
                    ac = base_ac.removeprefix("BASE_")
                    config["aircraft"][base_ac] = config["aircraft"][ac]

            else:
                full_base_inv_dict = {}
                base_ac_lst = []

            # check contrail input
            oac.check_cont_input(config, ds_cont)

            # initialise storage dictionaries
            cfdd_dict = {ac: {} for ac in ac_no_tot + base_ac_lst}
            cccov_taup05 = {ac: {} for ac in ac_no_tot + base_ac_lst}
            rf_cont_dict = {ac: [] for ac in ac_no_tot + base_ac_lst}

            # loop over ac for CFDD calculation
            for ac in ac_no_tot + base_ac_lst:
                if ac in ac_no_tot:
                    ac_inv_dict = full_inv_dict[ac]
                else:
                    ac_inv_dict = full_base_inv_dict[ac]

                # calculate CFDD
                cfdd_dict[ac] = oac.calc_cfdd(
                    config, ac_inv_dict, ds_cont, cont_grid, ac
                )

            # calculate total CFDD
            cfdd_dict = oac.calc_total_over_ac(cfdd_dict, ac_no_tot + base_ac_lst)
            cfdd_dict_1d = oac.cfdd_to_1d(cfdd_dict, cont_grid)

            # calculate contrail cirrus coverage (all optical depths)
            cccov_alltau_tot = oac.calc_cccov_alltau(
                cfdd_dict["TOTAL"], cont_grid
            )

            # loop over ac for cccov (tau > 0.05) calculation
            for ac in ac_no_tot + base_ac_lst:
                # attribute cccov (all tau) to ac
                att_cccov = oac.contrail_attribution(
                    cccov_alltau_tot, cfdd_dict_1d[ac], cfdd_dict_1d["TOTAL"]
                )

                # calculate cccov (tau > 0.05)
                cccov_taup05[ac] = oac.calc_cccov_taup05(config, att_cccov, ac)

            # calculate total cccov (tau > 0.05)
            cccov_taup05 = oac.calc_total_over_ac(
                cccov_taup05, ac_no_tot + base_ac_lst
            )

            # calculate total RF
            rf_cont_tot = oac.calc_cont_rf(cccov_taup05["TOTAL"], cont_grid)

            # loop over ac for RF calculation
            for ac in ac_no_tot + base_ac_lst:
                # attribute RF to ac
                att_rf = oac.contrail_attribution(
                    rf_cont_tot, cccov_taup05[ac], cccov_taup05["TOTAL"]
                )

                # calculate RF at each inventory year
                ac_rf_arr = np.array([np.mean(arr) for arr in att_rf.values()])

                # apply wingspan correction
                ac_rf_arr = oac.apply_wingspan_correction(config, ac_rf_arr, ac)

                # apply time evolution
                _, ac_rf_cont_dict = oac.apply_evolution(
                    config,
                    {"cont": ac_rf_arr},
                    inv_dict,
                    inventories_adjusted=True
                )

                # add to rf_cont_dict
                rf_cont_dict[ac] = ac_rf_cont_dict["cont"]

            # calculate total
            rf_cont_dict["TOTAL"] = sum(d for d in rf_cont_dict.values())

            # calculate dT and save to output_dict
            for ac in ac_lst:
                ac_rf_cont_dict = {"cont": rf_cont_dict[ac]}

                # update output_dict
                oac.update_output_dict(output_dict, ac, "RF", ac_rf_cont_dict)

                # calculate temperature change
                dtemp_cont_dict = oac.calc_dtemp(config, "cont", ac_rf_cont_dict)
                oac.update_output_dict(output_dict, ac, "dT", dtemp_cont_dict)


        else:
            logging.warning("No contrails defined in config.")


    # save results
    oac.write_output_dict_to_netcdf(config, output_dict, mode="w")

    # Calculate climate metrics
    run_metrics = config["output"]["run_metrics"]
    if run_metrics:
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
    run_plots = config["output"]["run_plots"]
    if run_plots:
        # Plot vertical profiles of inventories
        oac.plot_inventory_vertical_profiles(inv_dict)

        # Plot results
        output_name = config["output"]["name"]
        output_file = output_dir + output_name + ".nc"
        result_dic = oac.open_netcdf(output_file)
        oac.plot_results(config, result_dic, marker="o")

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
