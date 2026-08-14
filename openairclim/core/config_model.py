"""
Pydantic schema for the TOML configuration file. Single source of truth for
required/optional keys, defaults, and valid option strings (e.g. RF methods,
attribution methods).

Structural validation only: aircraft/contrail cross-checks, file
existence, and metrics/time-range consistency stay in read_config.py,
since they depend on I/O or on fields outside this file's ownership
(e.g. species.out gating which aircraft variables are required).

``Config``, ``AircraftEntry`` and ``AircraftCsvRow`` are the only public
classes. ``Config``'s sections are an implementation detail of how the
TOML tree is nested — callers outside this module should resolve them
by walking ``Config.model_fields`` (dotted TOML path) rather than
importing them directly, so they track the config's actual shape
instead of this file's internal class names. ``AircraftEntry``/
``AircraftCsvRow`` are the exception: they validate the
*dynamically-keyed* per-aircraft-identifier entries
(``config["aircraft"][<ac_id>]``), which aren't declared fields and so
can't be reached via ``model_fields`` — imported directly by
read_config.py (``AircraftCsvRow`` for bulk csv validation) and the
GUI's aircraft tab (``AircraftEntry``).
"""

import logging
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .calc_cont import calc_sac_slope
from .parametric import RATIO_DIC_D

# Maps deprecated config keys to their replacement, applied before
# validation so renamed keys keep working in old config files.
ALIAS_MAP = {
    "output.full_run": "output.run_oac",
}


def _apply_aliases(config: dict) -> dict:
    for old, new in ALIAS_MAP.items():
        cur = config
        parts = old.split(".")
        for p in parts[:-1]:
            if not isinstance(cur, dict) or p not in cur:
                break
            cur = cur[p]
        else:
            old_key = parts[-1]
            if old_key in cur:
                cur_new = config
                new_parts = new.split(".")
                for p in new_parts[:-1]:
                    cur_new = cur_new.setdefault(p, {})
                new_key = new_parts[-1]
                if new_key not in cur_new:
                    cur_new[new_key] = cur.pop(old_key)
                    logging.warning(
                        "Config key '%s' is deprecated; migrated to '%s'. "
                        "Please update your config file.",
                        old,
                        new,
                    )
                else:
                    logging.warning(
                        "Both deprecated key '%s' and new key '%s' exist; "
                        "keeping the new key. Please update your config file.",
                        old,
                        new,
                    )
    return config


# define which inventory species each output species' response calculation is
# actually driven by (e.g. O3/CH4 responses are computed from NOx emissions)
OUT_TO_INV_REQUIRED: dict[str, str] = {
    "CO2": "CO2", "H2O": "H2O", "O3": "NOx", "CH4": "NOx", "cont": "distance",
}

# define which output species are computed from another output species'
# results, rather than from an inventory directly
OUT_SPECIES_DEPENDENCIES: dict[str, list[str]] = {
    "PMO": ["CH4"],
    "SWV": ["CH4"],
}


class _SpeciesConfig(BaseModel):
    inv: list[Literal["CO2", "H2O", "NOx", "distance"]] = Field(
        description="Species defined in emission inventories."
    )
    out: list[Literal["CO2", "H2O", "O3", "CH4", "PMO", "cont", "SWV"]] = Field(
        description="Output / response species."
    )
    nox: Literal["NO", "NO2"] = Field(
        default="NO",
        description="Assumed NOx species in emission inventory."
    )

    @model_validator(mode="after")
    def _check_species_consistency(self) -> "_SpeciesConfig":
        for spec in self.out:
            required_inv = OUT_TO_INV_REQUIRED.get(spec)
            if required_inv and required_inv not in self.inv:
                raise ValueError(
                    f"'{spec}' in species.out requires '{required_inv}' in "
                    "species.inv."
                )
            for dep in OUT_SPECIES_DEPENDENCIES.get(spec, []):
                if dep not in self.out:
                    raise ValueError(
                        f"'{spec}' in species.out also requires '{dep}' in "
                        "species.out."
                    )
        return self


class _InventoriesBaseConfig(BaseModel):
    dir: Path = Path("")
    files: list[str] = Field(
        default_factory=list,
        description="Base emission inventories, describing all other air "
         "traffic (i.e. not included in the emission inventories). Only "
         "considered if `rel_to_base = True`. Only relevant for simulations "
         "with contrails."
    )


class _InventoriesConfig(BaseModel):
    dir: Path
    files: list[str]
    rel_to_base: bool = Field(
        default=False,
        description="Should the emission inventories be simulated on top of "
        "base air traffic? Only relevant simulations with contrails."
    )
    base: _InventoriesBaseConfig = Field(default_factory=_InventoriesBaseConfig)


class _OutputConfig(BaseModel):
    run_oac: bool = True
    run_metrics: bool = False
    run_plots: bool = False
    dir: Path
    name: str
    overwrite: bool = True
    concentrations: bool = False


class _TimeConfig(BaseModel):
    range: list[int]
    dir: Path = Field(
        default=Path(""),
        description="Path to the folder containing the time evolution file. "
        "Optional: only relevant if a time evolution file is used."
    )
    file: str | None = Field(
        default=None,
        description="Optional time evolution file (norm or scaling) for "
        "simulating beyond the range of the emission inventories. See also the "
        "[docs](https://openairclim.org/user_guide/02_evolution.html)."
    )

    @model_validator(mode="after")
    def _check_range(self) -> "_TimeConfig":
        if len(self.range) != 3:
            raise ValueError("time.range must have exactly 3 values: [start, end, step].")
        if self.range[1] <= self.range[0]:
            raise ValueError("Simulation end time must be after start time.")
        # calc_co2.py/calc_dt.py's impulse-response convolutions index
        # per-year arrays by the integer year offset (year - year_dash),
        # which only lines up with array positions for a 1-year step.
        if self.range[2] != 1:
            raise ValueError(
                "time.range step must be 1 — other step sizes are not yet "
                "supported by the core's response calculations."
            )
        return self


class _BackgroundSpeciesConfig(BaseModel):
    file: str
    scenario: str


class _BackgroundConfig(BaseModel):
    dir: Path
    CO2: _BackgroundSpeciesConfig
    CH4: _BackgroundSpeciesConfig
    N2O: _BackgroundSpeciesConfig


class _CO2ConcConfig(BaseModel):
    method: Literal["Sausen&Schumann"] = "Sausen&Schumann"


class _CO2RFConfig(BaseModel):
    method: Literal[
        "Etminan_2016", "IPCC_2001_1", "IPCC_2001_2", "IPCC_2001_3"
    ] = Field(
        default="Etminan_2016",
        description="The default RF method is based on " 
        "[Etminan et al. (2016)](https://doi.org/10.1002/2016gl071930). Other "
        "methods are from TAR [(IPCC, 2021)](https://www.ipcc.ch/report/ar3/wg1/)."
    )
    attr: Literal[
        "none", "residual", "marginal", "proportional", "differential"
    ] = Field(
        default="proportional",
        description="CO2 attribution method. See also the "
        "[docs](https://openairclim.org/background/attribution.html)."
    )


class _CO2ResponseConfig(BaseModel):
    response_grid: Literal["0D"] = "0D"
    conc: _CO2ConcConfig = Field(default_factory=_CO2ConcConfig)
    rf: _CO2RFConfig = Field(default_factory=_CO2RFConfig)


class _FileResponseConfig(BaseModel):
    file: str = ""


class _H2OResponseConfig(BaseModel):
    response_grid: Literal["2D"] = "2D"
    rf: _FileResponseConfig = Field(default_factory=_FileResponseConfig)


class _O3ResponseConfig(BaseModel):
    response_grid: Literal["2D"] = "2D"
    rf: _FileResponseConfig = Field(default_factory=_FileResponseConfig)


class _CH4RFConfig(BaseModel):
    method: Literal["Etminan_2016"] = Field(
        default="Etminan_2016",
        description="The default RF method is based on "
        "[Etminan et al. (2016)](https://doi.org/10.1002/2016gl071930)."
    )
    attr: Literal[
        "none", "residual", "marginal", "proportional", "differential"
    ] = Field(
        default="proportional",
        description="CH4 attribution method. See also the "
        "[docs](https://openairclim.org/background/attribution.html)."
    )


class _CH4ResponseConfig(BaseModel):
    response_grid: Literal["2D"] = "2D"
    tau: _FileResponseConfig = Field(default_factory=_FileResponseConfig)
    rf: _CH4RFConfig = Field(default_factory=_CH4RFConfig)


class _ContResponseConfig(BaseModel):
    response_grid: Literal["cont"] = "cont"
    resp: _FileResponseConfig = Field(default_factory=_FileResponseConfig)
    method: Literal["Megill_2026"] = Field(
        default="Megill_2026",
        description="Contrail module as described by Megill (2026)."
    )
    formation_method: Literal["Megill_2025"] = Field(
        default="Megill_2025",
        description="Persistent contrail formation method as described by "
        "[Megill et al. (2025)](https://doi.org/10.5194/acp-25-4131-2025)."
    )
    low_soot_case: Literal["case_low", "case_mid", "case_high"] = Field(
        default="case_mid",
        description="Low soot case as defined by Megill (2026). Requires "
        "OpenAirClim Premium."
    )


class _ResponsesConfig(BaseModel):
    dir: Path
    CO2: _CO2ResponseConfig = Field(default_factory=_CO2ResponseConfig)
    H2O: _H2OResponseConfig = Field(default_factory=_H2OResponseConfig)
    O3: _O3ResponseConfig = Field(default_factory=_O3ResponseConfig)
    CH4: _CH4ResponseConfig = Field(default_factory=_CH4ResponseConfig)
    cont: _ContResponseConfig = Field(default_factory=_ContResponseConfig)


class _CO2TemperatureConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    lambda_: float = Field(
        alias="lambda",
        default=0.73,
        description="Default climate sensitivity parameter from Table 1 of "
        "[Ponater et al. (2006)](https://doi.org/10.1016/j.atmosenv.2006.06.036)."
    )


class _EfficacyConfig(BaseModel):
    efficacy: float


class _TemperatureConfig(BaseModel):
    method: Literal["Boucher&Reddy"] = "Boucher&Reddy"
    CO2: _CO2TemperatureConfig = Field(default_factory=_CO2TemperatureConfig)
    H2O: _EfficacyConfig = Field(
        default_factory=lambda: _EfficacyConfig(efficacy=1.0),
        description="Expected range: [0.7, 1.3]. From Table 3.1 of 'Reference set "
        "of technical specifications for the MRV', report EC-CLIMA/2024/NP/0014, 2025, "
        "[link](https://climate.ec.europa.eu/document/download/735ae93d-d49d-46e0-b95b-48c36230ad57_en)."
    )
    O3: _EfficacyConfig = Field(
        default_factory=lambda: _EfficacyConfig(efficacy=1.05),
        description="Expected range: [0.74, 1.36]. From Table 3.1 of 'Reference set "
        "of technical specifications for the MRV', report EC-CLIMA/2024/NP/0014, 2025, "
        "[link](https://climate.ec.europa.eu/document/download/735ae93d-d49d-46e0-b95b-48c36230ad57_en)."
    )
    PMO: _EfficacyConfig = Field(
        default_factory=lambda: _EfficacyConfig(efficacy=1.0),
        description="Expected range: [0.7, 1.3]. From Table 3.1 of 'Reference set "
        "of technical specifications for the MRV', report EC-CLIMA/2024/NP/0014, 2025, "
        "[link](https://climate.ec.europa.eu/document/download/735ae93d-d49d-46e0-b95b-48c36230ad57_en)."
    )
    CH4: _EfficacyConfig = Field(
        default_factory=lambda: _EfficacyConfig(efficacy=1.04),
        description="Expected range: [0.84, 1.26]. From Table 3.1 of 'Reference set "
        "of technical specifications for the MRV', report EC-CLIMA/2024/NP/0014, 2025, "
        "[link](https://climate.ec.europa.eu/document/download/735ae93d-d49d-46e0-b95b-48c36230ad57_en)."
    )
    SWV: _EfficacyConfig = Field(
        default_factory=lambda: _EfficacyConfig(efficacy=1.0),
    )
    cont: _EfficacyConfig = Field(
        default_factory=lambda: _EfficacyConfig(efficacy=0.21),
        description="Expected range: [0.21, 0.59]. From Table 3.1 of 'Reference set "
        "of technical specifications for the MRV', report EC-CLIMA/2024/NP/0014, 2025, "
        "[link](https://climate.ec.europa.eu/document/download/735ae93d-d49d-46e0-b95b-48c36230ad57_en). "
        "Default value from Table 3 of [Bickel et al. (2025)](https://doi.org/10.1175/JCLI-D-24-0245.1)."
    )


class _MetricsConfig(BaseModel):
    types: list[Literal["AGWP", "ATR", "AGTP"]] = []
    t_0: list[int] = []
    H: list[int] = []


# sub-values that each derivable AircraftEntry field can be computed from
AIRCRAFT_DERIVATION_MAP: dict[str, list[str]] = {
    "G_250": ["SAC_eq", "Q_h", "eta", "eta_elec", "EIH2O", "R"],
    "PMrel": ["PM"],
}


class AircraftEntry(BaseModel):
    """Aircraft-specific parameters, defined in-line in the config 
    (`config["aircraft"][<ac_id>]`) or in a row of a linked csv file.
    
    Aircraft identifiers are required if the emission inventories have an `ac`
    data variable. In addition, the parameters `b` (wingspan [m]), `PMrel`
    (nvPM emissions relative to 1.5e15 kg⁻¹) and `G_250` (slope of the 
    Schmidt-Appleman mixing line at 250 hPa [Pa/K]) are required if contrails
    are being calculated (i.e. `"cont" in config["species"]["out"]`). These
    parameters can also be calculated using sub-values, defined in
    `AIRCRAFT_DERIVATION_MAP`.

    Field declaration order matches the aircraft csv column order, but is not
    important.
    """

    b: Annotated[float, Field(ge=20.0, le=80.0)] | None = Field(
        default=None,
        description="Aircraft wingspan [m], must be within [20, 80]. Not used "
        "within the low-soot regime."
    )
    PMrel: float | None = Field(
        default=None,
        description="Non-volatile particulate matter (nvPM) emissions, "
        "relative to 1.5e15 kg⁻¹. Can be derived from `PM` if left undefined."
    )
    G_250: float | None = Field(
        default=None,
        description="Slope of the Schmidt-Appleman mixing line at a reference " 
        "pressure of 250 hPa. Can be derived online from sub-values (`SAC_eq`, "
        "`Q_h`, `eta`, `eta_elec`, `EIH2O`, `R`) if left undefined."
    )
    SAC_eq: Literal["CON", "HYB", "H2C", "H2FC"] | None = Field(
        default=None,
        description="SAC equation used to derive G_250: 'CON' (conventional "
        "jet fuel), 'HYB' (hybrid-electric), 'H2C' (hydrogen combustion), "
        "'H2FC' (hydrogen fuel cell). For the equations used, see "
        "[Megill et al. (2025)](https://doi.org/10.5194/acp-25-4131-2025)."
    )
    Q_h: float | None = Field(
        default=None,
        description="Lower heating value of the fuel (Q) [J/kg], or, if "
        "`SAC_eq ='H2FC'`, formation enthalpy of water vapour (Δh) [J/mol]."
    )
    eta: float | None = Field(
        default=None,
        description="Overall propulsion system efficiency [-]."
    )
    eta_elec: float | None = Field(
        default=None,
        description="Overall propulsion efficiency of the electric/fuel-cell "
        "system [-] (for `SAC_eq = 'HYB' | 'H2FC'`)."
    )
    EIH2O: float | None = Field(
        default=None,
        description="Emission index of water vapour [kg/kg]."
    )
    R: float | None = Field(
        default=None,
        description="Degree of hybridisation: 1 = pure liquid fuel, 0 = pure "
        "electric. Only used for `SAC_eq = 'HYB'`."
    )
    PM: float | None = Field(
        default=None,
        description="Absolute non-volatile particulate matter (nvPM) number "
        "emission index, used to derive PMrel (PMrel = PM / 1.5e15 kg⁻¹)."
    )

    @model_validator(mode="after")
    def _derive(self) -> "AircraftEntry":
        if self.G_250 is None and any(
            getattr(self, c) is not None for c in AIRCRAFT_DERIVATION_MAP["G_250"]
        ):
            try:
                g_250 = calc_sac_slope(
                    250e2,
                    sac_eq=self.SAC_eq,
                    q_h=self.Q_h,
                    eta=self.eta,
                    eta_elec=self.eta_elec,
                    ei_h2o=self.EIH2O,
                    r=self.R,
                )
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Could not derive G_250 from sub-values: {exc}"
                ) from exc
            self.G_250 = round(g_250, 3)

        if self.PMrel is None and self.PM is not None:
            self.PMrel = round(self.PM / 1.5e15, 3)

        return self


class AircraftCsvRow(AircraftEntry):
    """One row of the aircraft csv file: AircraftEntry's fields plus the
    aircraft identifier column, and pandas' NaN/blank-string "missing"
    convention mapped to `None` before AircraftEntry's own field
    validation/derivation runs (which expects `None`.)
    """

    ac: str = Field(description="Aircraft identifier.")

    def _is_blank_csv_cell(value) -> bool:
        """True for a csv cell that should be read as 'not provided'."""
        if isinstance(value, float) and value != value:
            return True
        if isinstance(value, str) and not value.strip():
            return True
        return False

    @model_validator(mode="before")
    @classmethod
    def _blank_to_none(cls, data):
        if not isinstance(data, dict):
            return data
        return {
            key: None if cls._is_blank_csv_cell(value) else value
            for key, value in data.items()
        }


class _AircraftConfig(BaseModel):
    """Aircraft section of the config file. Since the per-identifier entries
    (config["aircraft"]["<ac_id>"]) are dynamic keys, rather than declared
    fields, they are typed using __pydantic_extra__, so that they are still
    validated.
    """
    model_config = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, AircraftEntry]
    types: list[str]
    dir: Path = Path("")
    file: str = ""


class _ParametricConfig(BaseModel):
    enabled: bool = Field(
        default=False,
        description="Default parametric values from Saleh Walie (2025) for the "
        "ATR20 metric, calculated using the results of "
        "[Castino et al. (2024)](https://doi.org/10.5194/gmd-17-4031-2024)."
    )
    CO2: float = RATIO_DIC_D["CO2"]
    H2O: float = RATIO_DIC_D["H2O"]
    O3: float = RATIO_DIC_D["O3"]
    CH4: float = RATIO_DIC_D["CH4"]
    cont: float = RATIO_DIC_D["cont"]


class Config(BaseModel):
    species: _SpeciesConfig
    inventories: _InventoriesConfig
    output: _OutputConfig
    time: _TimeConfig
    background: _BackgroundConfig
    responses: _ResponsesConfig
    temperature: _TemperatureConfig = Field(default_factory=_TemperatureConfig)
    metrics: _MetricsConfig = Field(default_factory=_MetricsConfig)
    aircraft: _AircraftConfig
    parametric: _ParametricConfig = Field(default_factory=_ParametricConfig)

    @model_validator(mode="before")
    @classmethod
    def _apply_aliases(cls, data: dict) -> dict:
        if isinstance(data, dict):
            return _apply_aliases(data)
        return data

    @model_validator(mode="after")
    def _check_metrics(self) -> "Config":
        """If metrics are being calculated, metrics.types/t_0/H must be
        complete, and every (t_0, H) combination must fall within time.range.
        """
        if not self.output.run_metrics:
            return self

        metrics = self.metrics
        if not (metrics.types and metrics.t_0 and metrics.H):
            raise ValueError(
                "metrics.types, metrics.t_0 and metrics.H must all be "
                "defined (non-empty) when output.run_metrics is true."
            )

        start, end, _ = self.time.range
        for t_0 in metrics.t_0:
            for horizon in metrics.H:
                if t_0 < start or t_0 + horizon > end:
                    raise ValueError(
                        f"Metrics time settings with t_0={t_0} and H={horizon} "
                        f"fall outside the simulation time range {self.time.range}."
                    )
        return self


def validate_config(config: dict) -> dict:
    """Validate a raw config dict against the schema and fill in defaults.

    Args:
        config (dict): Configuration dictionary, as loaded from TOML.

    Returns:
        dict: Configuration dictionary with structure/types validated,
            deprecated keys migrated, and defaults filled in.
    """
    return Config.model_validate(config).model_dump(by_alias=True, exclude_none=True)
