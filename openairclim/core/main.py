"""
main.py is the main interface to the submodules and the user script.
"""

import os
import sys
import time
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

# import OpenAirClim functions
from . import read_config, read_netcdf, calc_co2, calc_ch4, plot

from .attribution import apply_attribution
from .calc_cont import check_cont_input, calc_contrails
from .calc_dt import calc_dtemp
from .calc_metric import calc_climate_metrics
from .calc_response import calc_resp_all, calc_resp_sub
from .construct_conc import get_emissions, interp_bg_conc
from .interpolate_time import adjust_inventories, apply_evolution
from .parametric import adapt_co2_emission, adapt_rf
from .utils import convert_nested_to_series
from .write_output import (
    update_output_dict,
    write_output_dict_to_netcdf,
    write_climate_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class _RunState:
    """Mutable state shared across the processing steps of a single run()."""

    config: dict
    inv_dict: dict
    full_inv_dict: dict
    ac_lst: list
    output_dict: dict
    parametric_enabled: bool


def run(file_name: str) -> None:
    """Runs OpenAirClim.

    Args:
        file_name (str): Name of config file
    """
    start = time.time()
    _setup_logging()

    config = read_config.get_config(file_name)
    output_dir = config["output"]["dir"]
    inv_dict = None
    output_dict = None

    if config["output"]["run_oac"]:
        state, inv_species, spc_0d, spc_2d, spc_cont, spc_sub = _prepare_inventories(
            config
        )
        _calc_emissions(state, inv_species)
        _process_0d_species(state, spc_0d)
        _process_2d_species(state, spc_2d, config["output"]["concentrations"])
        _process_sub_species(state, spc_sub)
        _process_contrails(state, spc_cont)
        inv_dict = state.inv_dict
        output_dict = state.output_dict

    _finalize_output(config, output_dict, start)

    if config["output"]["run_plots"]:
        _generate_plots(config, inv_dict, output_dir)

    _cleanup_files(file_name, output_dir)


def _setup_logging() -> None:
    """Configures the logger for a run."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(module)s:%(lineno)d (%(funcName)s): %(message)s",
        level=logging.INFO,
        # TODO level=logging.DEBUG,
        handlers=[
            logging.FileHandler("debug.log", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def _prepare_inventories(config: dict) -> tuple:
    """Reads emission inventories and classifies species for a run.

    Args:
        config (dict): Configuration dictionary

    Returns:
        tuple: (state, inv_species, spc_0d, spc_2d, spc_cont, spc_sub), where
            ``state`` is a :class:`_RunState` holding the inventories and
            output dictionary shared across the run.
    """
    inv_species = config["species"]["inv"]
    spc_0d, spc_2d, spc_cont, spc_sub = read_config.classify_species(config)

    # Read emission inventories
    inv_dict = read_netcdf.open_inventories(config)
    # Adjust emission inventories to given time evolution
    inv_dict = adjust_inventories(config, inv_dict)
    # split inv_dict by aircraft identifiers defined in config
    full_inv_dict = read_netcdf.split_inventory_by_aircraft(config, inv_dict)

    # initialise loop over aircraft identifiers within full_inv_dict
    ac_lst = list(full_inv_dict.keys())
    output_dict: dict = {ac: {} for ac in ac_lst}

    state = _RunState(
        config=config,
        inv_dict=inv_dict,
        full_inv_dict=full_inv_dict,
        ac_lst=ac_lst,
        output_dict=output_dict,
        parametric_enabled=config["parametric"]["enabled"],
    )

    return state, inv_species, spc_0d, spc_2d, spc_cont, spc_sub


def _calc_emissions(state: _RunState, inv_species: list) -> None:
    """Calculates and stores total and per-aircraft emissions.

    Args:
        state (_RunState): Shared run state, updated in place
        inv_species (list): Inventory species
    """
    # calculate and save total emissions
    _, emis_dict = get_emissions(state.inv_dict, inv_species)
    _, emis_interp_dict = apply_evolution(
        state.config, emis_dict, state.inv_dict, inventories_adjusted=True
    )
    update_output_dict(state.output_dict, "TOTAL", "emis", emis_interp_dict)

    # calculate and save emissions for each aircraft identifier
    for ac in state.ac_lst:
        ac_inv_dict = state.full_inv_dict[ac]
        _, ac_emis_dict = get_emissions(ac_inv_dict, inv_species)
        _, ac_emis_interp_dict = apply_evolution(
            state.config, ac_emis_dict, state.inv_dict, inventories_adjusted=True
        )
        # parametric scenario: adapt CO2 emissions
        if state.parametric_enabled:
            ac_emis_interp_dict = adapt_co2_emission(state.config, ac_emis_interp_dict)
        update_output_dict(state.output_dict, ac, "emis", ac_emis_interp_dict)


def _process_0d_species(state: _RunState, spc_0d: list) -> None:
    """Handles 0D-response species (CO2): concentration, RF and dT.

    Args:
        state (_RunState): Shared run state, updated in place
        spc_0d (list): 0D-response species
    """
    if not spc_0d:
        logger.warning("No species defined in config with 0D response grid.")
        return

    if "CO2" not in spc_0d:
        logger.warning(
            "Species CO2 is not set or response_grid option is not "
            "set to 0D in config."
        )
        return

    config = state.config
    output_dict = state.output_dict

    # calculate concentration of all aircraft identifiers
    emis_co2_dict = {"CO2": output_dict["TOTAL"]["emis_CO2"]}
    conc_co2_dict = calc_co2.calc_co2_concentration(config, emis_co2_dict)

    # calculate background concentration (diff to reference C_0)
    conc_co2_bg_dict = interp_bg_conc(config, "CO2")
    conc_co2_bg_dict["CO2"] -= calc_co2.CO2_0

    # calculate total+background concentration (for attribution)
    tot_conc_co2_dict = {"CO2": conc_co2_dict["CO2"] + conc_co2_bg_dict["CO2"]}
    co2_att_method = config["responses"]["CO2"]["rf"]["attr"]

    # calculate concentrations and RF for each aircraft identifier
    for ac in state.ac_lst:
        # CO2 concentration
        ac_emis_co2_dict = {"CO2": output_dict[ac]["emis_CO2"]}
        ac_conc_co2_dict = calc_co2.calc_co2_concentration(config, ac_emis_co2_dict)
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


def _process_2d_species(state: _RunState, spc_2d: list, output_conc: bool) -> None:
    """Handles 2D-response species: concentration/RF responses and CH4 tau.

    Args:
        state (_RunState): Shared run state, updated in place
        spc_2d (list): 2D-response species
        output_conc (bool): Whether 2D concentration output was requested
    """
    if not spc_2d:
        logger.warning("No species defined in config with 2D response_grid.")
        return

    # Response: Emission --> Concentration
    if output_conc:
        # resp_conc_dict = oac.open_netcdf_from_config(
        #    config, "responses", spc_2d, "conc"
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
        logger.warning(
            "Computation of 2D concentration responses is not supported "
            "in this version. Change output settings to: concentrations = false"
        )

    # Response: Emission --> Radiative Forcing
    spc_rf, spc_tau = read_config.classify_response_types(state.config, spc_2d)

    if spc_rf:
        _process_2d_rf_species(state, spc_rf)

    if spc_tau:
        _process_2d_tau_species(state, spc_tau)


def _process_2d_rf_species(state: _RunState, spc_rf: list) -> None:
    """Handles 2D species with response type 'rf'. Currently: H2O, O3.

    Args:
        state (_RunState): Shared run state, updated in place
        spc_rf (list): 2D species with an RF response
    """
    config = state.config
    output_dict = state.output_dict

    resp_rf_dict = read_netcdf.open_netcdf_from_config(
        config, "responses", spc_rf, "rf"
    )
    # loop over aircraft identifiers and total
    for ac in state.ac_lst:
        ac_inv_dict = state.full_inv_dict[ac]
        rf_inv_years_dict = calc_resp_all(config, resp_rf_dict, ac_inv_dict)
        rf_series_dict = convert_nested_to_series(rf_inv_years_dict)
        _time_range, rf_interp_dict = apply_evolution(
            config, rf_series_dict, state.inv_dict, inventories_adjusted=True
        )
        # parametric scenario: adapt RF
        if state.parametric_enabled:
            rf_interp_dict = adapt_rf(config, rf_interp_dict, spc_rf)
        update_output_dict(output_dict, ac, "RF", rf_interp_dict)
        # RF --> dT
        # Calculate temperature change
        for spec in spc_rf:
            dtemp_dict = calc_dtemp(config, spec, rf_interp_dict)
            update_output_dict(output_dict, ac, "dT", dtemp_dict)


def _process_2d_tau_species(state: _RunState, spc_tau: list) -> None:
    """Handles the 2D response species. Currently: CH4.

    Args:
        state (_RunState): Shared run state, updated in place
        spc_tau (list): 2D species with a tau (lifetime) response
    """
    ch4_context = _calc_ch4_totals(state)

    for ac in state.ac_lst:
        _apply_ch4_response(state, ac, ch4_context, spc_tau)

    # give warning until validation is complete
    logger.warning("CH4 response surface is not validated!")


def _calc_ch4_totals(state: _RunState) -> tuple:
    """Calculates the CH4 tau response data, total+background concentration
    and attribution method shared across all aircraft identifiers.

    Args:
        state (_RunState): Shared run state

    Returns:
        tuple: (resp_tau_dict, tot_conc_ch4_dict, ch4_att_method)
    """
    config = state.config
    inv_dict = state.inv_dict

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
    conc_ch4_dict = calc_ch4.calc_ch4_concentration(config, tau_inverse_interp_dict)

    # calculate background concentration (diff to reference M_0)
    conc_ch4_bg_dict = interp_bg_conc(config, "CH4")
    conc_ch4_bg_dict["CH4"] -= calc_ch4.CH4_0

    # calculate total+background concentration (for attribution)
    tot_conc_ch4_dict = {"CH4": conc_ch4_dict["CH4"] + conc_ch4_bg_dict["CH4"]}
    ch4_att_method = config["responses"]["CH4"]["rf"]["attr"]

    return resp_tau_dict, tot_conc_ch4_dict, ch4_att_method


def _apply_ch4_response(
    state: _RunState, ac: str, ch4_context: tuple, spc_tau: list
) -> None:
    """Calculates and stores CH4 concentration, RF and dT for one aircraft
    identifier.

    Args:
        state (_RunState): Shared run state, updated in place
        ac (str): Aircraft identifier
        ch4_context (tuple): (resp_tau_dict, tot_conc_ch4_dict,
            ch4_att_method), shared across all aircraft identifiers - see
            :func:`_calc_ch4_totals`
        spc_tau (list): 2D species with a tau response - CH4
    """
    resp_tau_dict, tot_conc_ch4_dict, ch4_att_method = ch4_context

    ac_tau_inverse_dict = calc_resp_all(
        state.config, resp_tau_dict, state.full_inv_dict[ac]
    )
    ac_tau_inverse_series_dict = convert_nested_to_series(ac_tau_inverse_dict)
    _, ac_tau_inverse_interp_dict = apply_evolution(
        state.config,
        ac_tau_inverse_series_dict,
        state.inv_dict,
        inventories_adjusted=True,
    )
    ac_conc_ch4_dict = calc_ch4.calc_ch4_concentration(
        state.config, ac_tau_inverse_interp_dict
    )
    update_output_dict(state.output_dict, ac, "conc", ac_conc_ch4_dict)

    # CH4 RF
    ac_rf_ch4_dict = apply_attribution(
        calc_ch4.calc_ch4_rf,
        calc_ch4.calc_ch4_drf_dconc,
        ch4_att_method,
        "CH4",
        ac_conc_ch4_dict,
        tot_conc_ch4_dict,
        config=state.config,
    )
    # parametric scenario: adapt RF
    if state.parametric_enabled:
        ac_rf_ch4_dict = adapt_rf(state.config, ac_rf_ch4_dict, spc_tau)
    update_output_dict(state.output_dict, ac, "RF", ac_rf_ch4_dict)

    # CH4 dT
    ac_dt_ch4_dict = calc_dtemp(state.config, "CH4", ac_rf_ch4_dict)
    update_output_dict(state.output_dict, ac, "dT", ac_dt_ch4_dict)


def _process_sub_species(state: _RunState, spc_sub: list) -> None:
    """Handles subsequent species. Currently: PMO.

    Args:
        state (_RunState): Shared run state, updated in place
        spc_sub (list): Subsequent species (PMO)
    """
    if not spc_sub:
        logger.info("No subsequent species (PMO) defined in config.")
        return

    for ac in state.ac_lst:
        rf_sub_dict, conc_sub_dict = calc_resp_sub(spc_sub, state.output_dict, ac)
        # parametric scenario: adapt RF
        if state.parametric_enabled:
            rf_sub_dict = adapt_rf(state.config, rf_sub_dict, spc_sub)
        update_output_dict(state.output_dict, ac, "RF", rf_sub_dict)
        update_output_dict(state.output_dict, ac, "conc", conc_sub_dict)
        # RF --> dT
        # Calculate temperature change
        for spec in spc_sub:
            dtemp_dict = calc_dtemp(state.config, spec, rf_sub_dict)
            update_output_dict(state.output_dict, ac, "dT", dtemp_dict)


def _process_contrails(state: _RunState, spc_cont: list) -> None:
    """Handles contrail RF and dT.

    Args:
        state (_RunState): Shared run state, updated in place
        spc_cont (list): Contrail species
    """
    if not spc_cont:
        logger.warning("No contrails defined in config.")
        return

    # load contrail data
    ds_cont = read_netcdf.open_netcdf_from_config(
        state.config, "responses", ["cont"], "resp"
    )["cont"]

    # check contrail input
    check_cont_input(ds_cont)

    # calculate contrail RF
    rf_cont_dict = calc_contrails(
        state.ac_lst, state.config, state.inv_dict, state.full_inv_dict, ds_cont
    )

    # calculate contrail dT and save to output_dict
    for ac in state.ac_lst:
        ac_rf_cont_dict = {"cont": rf_cont_dict[ac]}

        # parametric scenario: adapt RF
        if state.parametric_enabled:
            ac_rf_cont_dict = adapt_rf(state.config, ac_rf_cont_dict, spc_cont)

        # update output_dict
        update_output_dict(state.output_dict, ac, "RF", ac_rf_cont_dict)

        # calculate temperature change
        dtemp_cont_dict = calc_dtemp(state.config, "cont", ac_rf_cont_dict)
        update_output_dict(state.output_dict, ac, "dT", dtemp_cont_dict)


def _finalize_output(config: dict, output_dict: dict | None, start_time: float) -> None:
    """Writes output and climate metrics, and logs execution time.

    Args:
        config (dict): Configuration dictionary
        output_dict (dict or None): Output dictionary, or None if run_oac
            was disabled and no new output was computed
        start_time (float): Run start time, as returned by :func:`time.time`
    """
    # save results, if run_oac produced any
    if output_dict is not None:
        write_output_dict_to_netcdf(config, output_dict, mode="w")

    # Calculate climate metrics
    run_metrics = config["output"]["run_metrics"]
    if run_metrics:
        metrics_dict = calc_climate_metrics(config)
        write_climate_metrics(config, metrics_dict)

    # Record end time
    end = time.time()
    # Execution time is difference between start and end time
    msg = "Execution time: " + str(end - start_time) + " sec"
    logger.info(msg)

    # WARNING message: demonstrating purposes
    logger.warning(
        "OpenAirClim is currently in development phase.\n"
        "The computed output is not for scientific purposes "
        "until release of our publication.\n"
        "Amongst others, the climate impact of longer species lifetimes "
        "in the stratosphere is not considered."
    )


def _generate_plots(config: dict, inv_dict: dict | None, output_dir: Path) -> None:
    """Plots inventory vertical profiles and run results.

    Args:
        config (dict): Configuration dictionary
        inv_dict (dict or None): Emission inventories, or None if run_oac
            was disabled and no inventories were read
        output_dir (str): Output directory
    """
    # Plot vertical profiles of inventories, if run_oac read any
    if inv_dict is not None:
        plot.plot_inventory_vertical_profiles(inv_dict, output_dir)

    # Plot results
    output_name = config["output"]["name"]
    output_file = Path(output_dir) / f"{output_name}.nc"
    result_dic = read_netcdf.open_netcdf(output_file)
    plot.plot_results(config, result_dic, marker="o")


def _cleanup_files(file_name: str, output_dir: Path) -> None:
    """Closes logger handlers and moves the config/log files to the output
    dir.

    Args:
        file_name (str): Name of config file
        output_dir (str): Output directory
    """
    # clean up: close all logger handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.close()
        root_logger.removeHandler(handler)

    # move config and log files to results folder
    shutil.copy2(file_name, f"{output_dir}")
    if os.path.exists(f"{output_dir}/debug.log"):
        os.remove(f"{output_dir}/debug.log")
    shutil.move("debug.log", f"{output_dir}")
