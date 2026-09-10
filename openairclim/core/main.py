"""Main interface to the submodules and the user script."""

import datetime
import getpass
import logging
import os
import platform
import shutil
import sys
import time
from pathlib import Path

from .. import OAC_PREMIUM_AVAILABLE, __version__
from . import calc_ch4, calc_co2, plot, read_config, read_netcdf
from .attribution import apply_attribution
from .calc_cont import calc_contrails, check_cont_input
from .calc_dt import calc_dtemp
from .calc_metric import calc_climate_metrics
from .calc_response import calc_resp_all, calc_resp_sub
from .construct_conc import get_emissions, interp_bg_conc
from .interpolate_time import adjust_inventories, apply_evolution
from .parametric import adapt_co2_emission, adapt_rf
from .utils import convert_nested_to_series
from .write_output import (
    update_output_dict,
    write_climate_metrics,
    write_output_dict_to_netcdf,
)

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configures the root logger with a debug-log file and stdout handlers."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(module)s:%(lineno)d (%(funcName)s): "
        "%(message)s",
        level=logging.INFO,
        # TODO level=logging.DEBUG,
        handlers=[
            logging.FileHandler("debug.log", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def _prepare_inventories(config: dict) -> tuple[list[str], dict, dict]:
    """Reads, time-adjusts and splits emission inventories by aircraft.

    Args:
        config (dict): Configuration dictionary from config.

    Returns:
        tuple[list[str], dict, dict]: ``ac_lst`` (aircraft identifiers,
        including "TOTAL"); ``inv_dict`` (combined inventories, adjusted
        to the configured time evolution); ``full_inv_dict`` (inv_dict
        split by aircraft identifier).
    """
    inv_dict = read_netcdf.open_inventories(config)
    inv_dict = adjust_inventories(config, inv_dict)
    full_inv_dict = read_netcdf.split_inventory_by_aircraft(config, inv_dict)
    ac_lst = list(full_inv_dict.keys())
    return ac_lst, inv_dict, full_inv_dict


def _calc_emissions(
    config: dict,
    output_dict: dict,
    ac_lst: list[str],
    inv_dict: dict,
    full_inv_dict: dict,
) -> None:
    """Calculates and stores total and per-aircraft emissions.

    Applies the configured time evolution, and - if the parametric module is
    enabled - adapts per-aircraft CO2 emissions. The ``output_dict`` is
    modified in place.

    Args:
        config (dict): Configuration dictionary from config.
        output_dict (dict): Output dictionary, updated in-place.
        ac_lst (list[str]): Aircraft identifiers, including "TOTAL".
        inv_dict (dict): Combined (all-aircraft) emission inventories.
        full_inv_dict (dict): inv_dict split by aircraft identifier.
    """
    inv_species = config["species"]["inv"]
    parametric_enabled = config["parametric"]["enabled"]

    # total emissions
    _, emis_dict = get_emissions(inv_dict, inv_species)
    _, emis_interp_dict = apply_evolution(
        config, emis_dict, inv_dict, inventories_adjusted=True
    )
    update_output_dict(output_dict, "TOTAL", "emis", emis_interp_dict)

    # per-aircraft emissions
    for ac in ac_lst:
        ac_inv_dict = full_inv_dict[ac]
        _, ac_emis_dict = get_emissions(ac_inv_dict, inv_species)
        _, ac_emis_interp_dict = apply_evolution(
            config, ac_emis_dict, inv_dict, inventories_adjusted=True
        )
        # parametric scenario: adapt CO2 emissions
        if parametric_enabled:
            ac_emis_interp_dict = adapt_co2_emission(config, ac_emis_interp_dict)
        update_output_dict(output_dict, ac, "emis", ac_emis_interp_dict)


def _calc_co2_response(
    config: dict, output_dict: dict, ac_lst: list[str], species_0d: list[str]
) -> None:
    """Calculates CO2 concentration, RF and temperature change.

    CO2 is currently the only species using the 0D response method
    (see :func:`~openairclim.core.read_config.classify_species`).

    Args:
        config (dict): Configuration dictionary from config.
        output_dict (dict): Output dictionary, updated in-place. Must
            already contain "emis_CO2" for every aircraft identifier in
            ac_lst (see :func:`_calc_emissions`).
        ac_lst (list[str]): Aircraft identifiers, including "TOTAL".
        species_0d (list[str]): 0D species from
            :func:`~openairclim.core.read_config.classify_species`.
    """
    if not species_0d:
        logger.warning("No species defined in config with 0D response grid.")
        return
    if "CO2" not in species_0d:
        logger.warning(
            "Species CO2 is not set or response_grid option is not set to 0D in config."
        )
        return

    # total concentration (for attribution)
    emis_co2_dict = {"CO2": output_dict["TOTAL"]["emis_CO2"]}
    conc_co2_dict = calc_co2.calc_co2_concentration(config, emis_co2_dict)
    conc_co2_bg_dict = interp_bg_conc(config, "CO2")
    conc_co2_bg_dict["CO2"] -= calc_co2.CO2_0
    tot_conc_co2_dict = {"CO2": conc_co2_dict["CO2"] + conc_co2_bg_dict["CO2"]}
    co2_att_method = config["responses"]["CO2"]["rf"]["attr"]

    # concentration, RF and dT for each aircraft identifier
    for ac in ac_lst:
        ac_emis_co2_dict = {"CO2": output_dict[ac]["emis_CO2"]}
        ac_conc_co2_dict = calc_co2.calc_co2_concentration(config, ac_emis_co2_dict)
        update_output_dict(output_dict, ac, "conc", ac_conc_co2_dict)

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

        ac_dt_co2_dict = calc_dtemp(config, "CO2", ac_rf_co2_dict)
        update_output_dict(output_dict, ac, "dT", ac_dt_co2_dict)


def _calc_rf_response_species(  # noqa: PLR0913, PLR0917
    config: dict,
    output_dict: dict,
    ac_lst: list[str],
    inv_dict: dict,
    full_inv_dict: dict,
    species_rf: list[str],
) -> None:
    """Calculates RF and temperature change for direct RF-response species.

    These are species with a response surface that maps emissions directly
    to radiative forcing (no concentration step). Currently H2O and O3 (see
    :func:`~openairclim.core.read_config.classify_response_types`).

    Args:
        config (dict): Configuration dictionary from config.
        output_dict (dict): Output dictionary, updated in-place.
        ac_lst (list[str]): Aircraft identifiers, including "TOTAL".
        inv_dict (dict): Combined (all-aircraft) emission inventories.
        full_inv_dict (dict): inv_dict split by aircraft identifier.
        species_rf (list[str]): RF-response species from
            :func:`~openairclim.core.read_config.classify_response_types`.
    """
    if not species_rf:
        return
    parametric_enabled = config["parametric"]["enabled"]

    resp_rf_dict = read_netcdf.open_netcdf_from_config(
        config, "responses", species_rf, "rf"
    )
    # loop over aircraft identifiers and total
    for ac in ac_lst:
        ac_inv_dict = full_inv_dict[ac]
        rf_inv_years_dict = calc_resp_all(config, resp_rf_dict, ac_inv_dict)
        rf_series_dict = convert_nested_to_series(rf_inv_years_dict)
        _time_range, rf_interp_dict = apply_evolution(
            config, rf_series_dict, inv_dict, inventories_adjusted=True
        )
        # parametric scenario: adapt RF
        if parametric_enabled:
            rf_interp_dict = adapt_rf(config, rf_interp_dict, species_rf)
        update_output_dict(output_dict, ac, "RF", rf_interp_dict)

        # RF --> dT
        for spec in species_rf:
            dtemp_dict = calc_dtemp(config, spec, rf_interp_dict)
            update_output_dict(output_dict, ac, "dT", dtemp_dict)


def _calc_ch4_response(  # noqa: PLR0913, PLR0917
    config: dict,
    output_dict: dict,
    ac_lst: list[str],
    inv_dict: dict,
    full_inv_dict: dict,
    species_tau: list[str],
) -> None:
    """Calculates CH4 concentration, RF and temperature change.

    CH4 is currently the only species using the inverse-lifetime ("tau")
    response method. An inverse-lifetime response surface is converted to a
    concentration, then RF is calculated an attributed (see
    :func:`~openairclim.core.read_config.classify_response_types`).

    Args:
        config (dict): Configuration dictionary from config.
        output_dict (dict): Output dictionary, updated in-place.
        ac_lst (list[str]): Aircraft identifiers, including "TOTAL".
        inv_dict (dict): Combined (all-aircraft) emission inventories.
        full_inv_dict (dict): inv_dict split by aircraft identifier.
        species_tau (list[str]): tau-response species from
            :func:`~openairclim.core.read_config.classify_response_types`
            (currently always ``["CH4"]`` or empty).
    """
    if not species_tau:
        return
    parametric_enabled = config["parametric"]["enabled"]

    resp_tau_dict = read_netcdf.open_netcdf_from_config(
        config, "responses", ["CH4"], "tau"
    )

    # total concentration (for attribution)
    tau_inverse_dict = calc_resp_all(config, resp_tau_dict, inv_dict)
    tau_inverse_series_dict = convert_nested_to_series(tau_inverse_dict)
    _, tau_inverse_interp_dict = apply_evolution(
        config, tau_inverse_series_dict, inv_dict, inventories_adjusted=True
    )
    conc_ch4_dict = calc_ch4.calc_ch4_concentration(config, tau_inverse_interp_dict)
    conc_ch4_bg_dict = interp_bg_conc(config, "CH4")
    conc_ch4_bg_dict["CH4"] -= calc_ch4.CH4_0
    tot_conc_ch4_dict = {"CH4": conc_ch4_dict["CH4"] + conc_ch4_bg_dict["CH4"]}
    ch4_att_method = config["responses"]["CH4"]["rf"]["attr"]

    # concentration, RF and dT for each aircraft identifier
    for ac in ac_lst:
        ac_inv_dict = full_inv_dict[ac]
        ac_tau_inverse_dict = calc_resp_all(config, resp_tau_dict, ac_inv_dict)
        ac_tau_inverse_series_dict = convert_nested_to_series(ac_tau_inverse_dict)
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
            ac_rf_ch4_dict = adapt_rf(config, ac_rf_ch4_dict, species_tau)
        update_output_dict(output_dict, ac, "RF", ac_rf_ch4_dict)

        ac_dt_ch4_dict = calc_dtemp(config, "CH4", ac_rf_ch4_dict)
        update_output_dict(output_dict, ac, "dT", ac_dt_ch4_dict)

    # give warning until validation is complete
    logger.warning("CH4 response surface is not validated!")


def _calc_sub_species_response(
    config: dict, output_dict: dict, ac_lst: list[str], species_sub: list[str]
) -> None:
    """Calculates RF, concentration and temperature change for sub-species.

    Sub-species depend on the already-computed results of other (main)
    species - currently PMO and SWV. The per-species logic (e.g. SWV
    requiring a CH4 concentration) lives in
    :func:`~openairclim.core.calc_response.calc_resp_sub`.

    Args:
        config (dict): Configuration dictionary from config.
        output_dict (dict): Output dictionary, updated in-place. Must
            already contain the main species results sub-species depend on
            (e.g. SWV requires "conc_CH4").
        ac_lst (list[str]): Aircraft identifiers, including "TOTAL".
        species_sub (list[str]): Sub-species from
            :func:`~openairclim.core.read_config.classify_species`.
    """
    if not species_sub:
        logger.info("No subsequent species (PMO) defined in config.")
        return
    parametric_enabled = config["parametric"]["enabled"]

    for ac in ac_lst:
        rf_sub_dict, conc_sub_dict = calc_resp_sub(species_sub, config, output_dict, ac)
        # parametric scenario: adapt RF
        if parametric_enabled:
            rf_sub_dict = adapt_rf(config, rf_sub_dict, species_sub)
        update_output_dict(output_dict, ac, "RF", rf_sub_dict)
        update_output_dict(output_dict, ac, "conc", conc_sub_dict)

        # RF --> dT
        for spec in species_sub:
            dtemp_dict = calc_dtemp(config, spec, rf_sub_dict)
            update_output_dict(output_dict, ac, "dT", dtemp_dict)


def _calc_contrail_response(  # noqa: PLR0913, PLR0917
    config: dict,
    output_dict: dict,
    ac_lst: list[str],
    inv_dict: dict,
    full_inv_dict: dict,
    species_cont: list[str],
) -> None:
    """Calculates contrail RF and temperature change (species "cont").

    Args:
        config (dict): Configuration dictionary from config.
        output_dict (dict): Output dictionary, updated in-place.
        ac_lst (list[str]): Aircraft identifiers, including "TOTAL".
        inv_dict (dict): Combined (all-aircraft) emission inventories.
        full_inv_dict (dict): inv_dict split by aircraft identifier.
        species_cont (list[str]): Contrail "species" from
            :func:`~openairclim.core.read_config.classify_species`
            (currently always ``["cont"]`` or empty).
    """
    if not species_cont:
        logger.warning("No contrails defined in config.")
        return
    parametric_enabled = config["parametric"]["enabled"]

    # load contrail data
    ds_cont = read_netcdf.open_netcdf_from_config(
        config, "responses", ["cont"], "resp"
    )["cont"]
    check_cont_input(ds_cont)

    # calculate contrail RF
    rf_cont_dict = calc_contrails(ac_lst, config, inv_dict, full_inv_dict, ds_cont)

    # calculate contrail dT and save to output_dict
    for ac in ac_lst:
        ac_rf_cont_dict = {"cont": rf_cont_dict[ac]}
        # parametric scenario: adapt RF
        if parametric_enabled:
            ac_rf_cont_dict = adapt_rf(config, ac_rf_cont_dict, species_cont)
        update_output_dict(output_dict, ac, "RF", ac_rf_cont_dict)

        dtemp_cont_dict = calc_dtemp(config, "cont", ac_rf_cont_dict)
        update_output_dict(output_dict, ac, "dT", dtemp_cont_dict)


def _calc_and_write_metrics(config: dict) -> None:
    """Calculates and writes climate metrics, if enabled in config.

    Args:
        config (dict): Configuration dictionary from config.
    """
    if not config["output"]["run_metrics"]:
        return
    metrics_dict = calc_climate_metrics(config)
    write_climate_metrics(config, metrics_dict)


def _plot_results(config: dict, inv_dict: dict, output_dir: str) -> None:
    """Plots inventory vertical profiles and simulation results.

    Args:
        config (dict): Configuration dictionary from config.
        inv_dict (dict): Combined (all-aircraft) emission inventories.
        output_dir (str): Directory the results netCDF file was written to.
    """
    # Plot vertical profiles of inventories
    plot.plot_inventory_vertical_profiles(inv_dict, output_dir)

    # Plot results
    output_name = config["output"]["name"]
    output_file = Path(output_dir) / f"{output_name}.nc"
    result_dic = read_netcdf.open_netcdf(output_file)
    plot.plot_results(config, result_dic, marker="o")


def _finalize_output_files(file_name: str, output_dir: str) -> None:
    """Closes log handlers and moves the config/log files to output_dir.

    Args:
        file_name (str): Path to the config file that was run.
        output_dir (str): Directory to move the config and log files into.
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


def run(file_name: str) -> None:
    """Runs OpenAirClim.

    Reads the config, calculates emissions and species responses (see the
    ``_calc_*_response`` functions in this module for where each species is
    handled), writes results/metrics, and optionally plots.

    Args:
        file_name (str): Name of config file
    """
    _configure_logging()
    start = time.time()

    try:
        username = getpass.getuser()
    except OSError:
        username = "N/A"

    logger.info(
        "\n"
        "-------------- OPENAIRCLIM MAIN RUN LOG --------------\n"
        f"oac version: {__version__}\n"
        f"timestamp: {datetime.datetime.now()}\n"
        f"input file: {file_name}\n"
        f"user: {username}\n"
        f"platform: {platform.system()} {platform.release()}\n"
        f"oac_premium available: {OAC_PREMIUM_AVAILABLE}\n\n"
        "NOTE:\n"
        "OpenAirClim is currently in development phase.\n"
        "The computed output is not to be used for scientific\n"
        "purposes until the release of our publication.\n"
        "The NOx module (output species CH4, O3, PMO and SWV)\n"
        "are not validated and can produce misleading results.\n"
        "------------------------------------------------------\n"
        "START OF LOG\n"
    )

    config = read_config.get_config(file_name)
    run_oac = config["output"]["run_oac"]
    output_conc = config["output"]["concentrations"]
    output_dir = config["output"]["dir"]

    if run_oac:
        ac_lst, inv_dict, full_inv_dict = _prepare_inventories(config)
        output_dict: dict = {ac: {} for ac in ac_lst}

        species_0d, species_2d, species_cont, species_sub = (
            read_config.classify_species(config)
        )
        species_rf, species_tau = read_config.classify_response_types(
            config, species_2d
        )

        _calc_emissions(config, output_dict, ac_lst, inv_dict, full_inv_dict)

        # 0D species: CO2
        _calc_co2_response(config, output_dict, ac_lst, species_0d)

        # 2D species: H2O, O3 (direct RF response) and CH4 (tau response)
        if not species_2d:
            logger.warning("No species defined in config with 2D response_grid.")
        elif output_conc:
            logger.warning(
                "Computation of 2D concentration responses is not supported "
                "in this version. Change output settings to: concentrations = false"
            )
        _calc_rf_response_species(
            config, output_dict, ac_lst, inv_dict, full_inv_dict, species_rf
        )
        _calc_ch4_response(
            config, output_dict, ac_lst, inv_dict, full_inv_dict, species_tau
        )

        # Sub-species: PMO, SWV
        _calc_sub_species_response(config, output_dict, ac_lst, species_sub)

        # Contrails
        _calc_contrail_response(
            config, output_dict, ac_lst, inv_dict, full_inv_dict, species_cont
        )

        # save results
        write_output_dict_to_netcdf(config, output_dict, mode="w")

    # Calculate climate metrics
    _calc_and_write_metrics(config)

    # Record end time and log execution time
    end = time.time()
    logger.info("Execution time: %s sec", end - start)
    logger.info(
        "\n\nEND OF LOG\n------------------------------------------------------"
    )

    # PLOTS
    if config["output"]["run_plots"]:
        _plot_results(config, inv_dict, output_dir)

    _finalize_output_files(file_name, output_dir)
