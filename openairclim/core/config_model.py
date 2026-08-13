"""
Pydantic schema for the TOML configuration file. Single source of truth for
required/optional keys, defaults, and valid option strings (e.g. RF methods,
attribution methods).

Structural validation only: aircraft/contrail cross-checks, file
existence, and metrics/time-range consistency stay in read_config.py,
since they depend on I/O or on fields outside this file's ownership
(e.g. species.out gating which aircraft variables are required).

``Config`` is the only public class — everything else is an
implementation detail of how its sections are nested. Callers outside
this module should resolve submodels by walking ``Config.model_fields``
(dotted TOML path) rather than importing them directly, so they track
the config's actual shape instead of this file's internal class names.
"""

import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class _AircraftConfig(BaseModel):
    # Per-identifier entries (config["aircraft"]["<ac_id>"]) are dynamic
    # keys, not declared fields — validated later in
    # read_config._aircraft_identifier_validation.
    model_config = ConfigDict(extra="allow")
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


def validate_config(config: dict) -> dict:
    """Validate a raw config dict against the schema and fill in defaults.

    Args:
        config (dict): Configuration dictionary, as loaded from TOML.

    Returns:
        dict: Configuration dictionary with structure/types validated,
            deprecated keys migrated, and defaults filled in.
    """
    return Config.model_validate(config).model_dump(by_alias=True, exclude_none=True)
