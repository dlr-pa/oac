"""Provides tests for module config_model

Covers the custom validation logic in config_model.py (alias migration,
species consistency, time range checks, AircraftEntry derivation,
AircraftCsvRow blank-cell handling, and the metrics/time-range cross-check) -
not pydantic's own type/required-field machinery, which is exercised
implicitly by every test that builds a Config.

`valid_config` comes from the top-level tests/conftest.py - pytest injects
it automatically, no import needed. config_model's validate_config never
touches the filesystem, so its (real) referenced files don't need to exist
for these tests either way.
"""

# accessing private classes/functions directly is the point of these tests;
# pylint doesn't recognise pytest's fixture injection
# pylint: disable=protected-access,redefined-outer-name

import math

import pytest
from pydantic import ValidationError, TypeAdapter

from openairclim.core import config_model
from openairclim.core.calc_cont import calc_sac_slope
from openairclim.core.config_model import (
    AircraftCsvRow,
    AircraftEntry,
    Config,
    validate_config,
)


class TestApplyAliases:
    """Tests function _apply_aliases(config)"""

    def test_migrates_deprecated_key(self):
        """Tests deprecated key is replaced."""
        config = {"output": {"full_run": True}}
        result = config_model._apply_aliases(config)
        assert result["output"]["run_oac"] is True
        assert "full_run" not in result["output"]

    def test_new_key_untouched_when_old_absent(self):
        """Tests that new keys remain untouched."""
        config = {"output": {"run_oac": False}}
        result = config_model._apply_aliases(config)
        assert result["output"]["run_oac"] is False

    def test_new_key_wins_when_both_present(self):
        """Tests that a new key remains untouched when both new and old are
        present."""
        config = {"output": {"full_run": True, "run_oac": False}}
        result = config_model._apply_aliases(config)
        assert result["output"]["run_oac"] is False

    def test_applied_through_full_validate_config(self, valid_config):
        """Tests that Config._apply_aliases works within the full validation."""
        # valid_config already sets "run_oac" explicitly - drop it so only
        # the deprecated "full_run" key is present
        output = {k: v for k, v in valid_config["output"].items() if k != "run_oac"}
        config = {**valid_config, "output": {**output, "full_run": False}}
        result = validate_config(config)
        assert result["output"]["run_oac"] is False


class TestSpeciesConsistency:
    """Tests _SpeciesConfig._check_species_consistency, via Config"""

    def test_consistent_species_ok(self, valid_config):
        """Tests valid data."""
        config = {**valid_config, "species": {"inv": ["CO2"], "out": ["CO2"]}}
        validate_config(config)  # does not raise

    def test_out_species_missing_required_inv_species_raises(self, valid_config):
        """Tests that output species require corresponding input species (e.g.
        O3 output requires NOx input)."""
        config = {**valid_config, "species": {"inv": ["CO2"], "out": ["O3"]}}
        with pytest.raises(ValidationError, match="NOx"):
            validate_config(config)

    def test_out_species_missing_dependency_raises(self, valid_config):
        """Tests that response species require corresponding input species
        (e.g. PMO requires CH4 output)."""
        config = {
            **valid_config,
            "species": {"inv": ["CO2", "NOx"], "out": ["PMO"]},
        }
        with pytest.raises(ValidationError, match="CH4"):
            validate_config(config)


class TestTimeConfigRange:
    """Tests _TimeConfig._check_range, via Config"""

    def test_valid_range_ok(self, valid_config):
        """Tests valid configuration."""
        config = {
            **valid_config,
            "time": {"range": [2020, 2030, 1]},
            "output": {**valid_config["output"], "run_metrics": False},
        }
        validate_config(config)  # does not raise

    def test_wrong_length_raises(self, valid_config):
        """Tests incorrect input length."""
        config = {**valid_config, "time": {"range": [2020, 2030]}}
        with pytest.raises(ValidationError, match="exactly 3 values"):
            validate_config(config)

    def test_end_before_start_raises(self, valid_config):
        """Tests end < start."""
        config = {**valid_config, "time": {"range": [2030, 2020, 1]}}
        with pytest.raises(ValidationError, match="end time must be after"):
            validate_config(config)

    def test_step_other_than_one_raises(self, valid_config):
        """Tests step other than 1 (currently required)."""
        config = {**valid_config, "time": {"range": [2020, 2030, 5]}}
        with pytest.raises(ValidationError, match="step must be 1"):
            validate_config(config)


class TestAircraftEntryDerive:
    """Tests AircraftEntry._derive"""

    CON_SUBVALUES = {"SAC_eq": "CON", "Q_h": 43e6, "eta": 0.3, "EIH2O": 1.25}

    def test_explicit_g250_not_overridden_by_subvalues(self):
        """Tests that G_250 is not over-ridden by subvalues."""
        entry = AircraftEntry(G_250=99.0, **self.CON_SUBVALUES)
        assert entry.G_250 == 99.0

    def test_g250_derived_from_subvalues_matches_calc_cont(self):
        """Tests that G_250 derived from subvalues matches the value calculated
        online."""
        entry = AircraftEntry(**self.CON_SUBVALUES)
        expected = round(
            calc_sac_slope(250e2, sac_eq="CON", q_h=43e6, eta=0.3, ei_h2o=1.25), 3
        )
        assert entry.G_250 == expected

    def test_g250_not_attempted_without_any_subvalues(self):
        """Tests that G_250 is not calculated without sub-values."""
        entry = AircraftEntry(b=45.0)
        assert entry.G_250 is None

    def test_g250_incomplete_subvalues_raises(self):
        """Tests incomplete sub-values."""
        with pytest.raises(ValidationError, match="Could not derive G_250"):
            AircraftEntry(Q_h=43e6)

    def test_explicit_pmrel_not_overridden_by_pm(self):
        """Tests PMrel not over-ridden by PM."""
        entry = AircraftEntry(PMrel=0.1, PM=3.0e15)
        assert entry.PMrel == 0.1

    def test_pmrel_derived_from_pm(self):
        """Tests correct functioning of PM -> PMrel calculation."""
        entry = AircraftEntry(PM=3.0e15)
        assert entry.PMrel == pytest.approx(2.0)

    def test_pmrel_not_attempted_without_pm(self):
        """Tests that PMrel is not calculated without PM."""
        entry = AircraftEntry(b=45.0)
        assert entry.PMrel is None


class TestAircraftCsvRowBlankHandling:
    """Tests AircraftCsvRow._blank_to_none / _is_blank_csv_cell."""

    def test_real_numeric_value_preserved(self):
        """The bug this guards against: pandas gives a real value in a
        column that also has NaN elsewhere the same Python type (float) as
        the NaN itself - isinstance(value, float) alone can't tell them
        apart."""
        row = TypeAdapter(AircraftCsvRow).validate_python(
            {"ac": "AC1", "b": 45.0}
        )
        assert row.b == 45.0

    def test_nan_float_mapped_to_none(self):
        """Tests that NaN is mapped to None."""
        row = TypeAdapter(AircraftCsvRow).validate_python(
            {"ac": "AC1", "b": math.nan}
        )
        assert row.b is None

    def test_blank_string_mapped_to_none(self):
        """Tests that a blank string is mapped to None."""
        row = TypeAdapter(AircraftCsvRow).validate_python(
            {"ac": "AC1", "SAC_eq": "  "}
        )
        assert row.SAC_eq is None


class TestConfigCheckMetrics:
    """Tests Config._check_metrics"""

    def test_run_metrics_off_ignores_missing_metrics(self, valid_config):
        """Tests that `run_metrics=False` doesn't require the definition of
        climate metrics."""
        config = {
            **valid_config,
            "output": {**valid_config["output"], "run_metrics": False},
        }
        validate_config(config)  # does not raise

    def test_run_metrics_on_requires_complete_metrics(self, valid_config):
        """Tests that `run_metrics=True` requires valid combination of
        climate metrics."""
        config = {
            **valid_config,
            "output": {**valid_config["output"], "run_metrics": True},
            "metrics": {"types": ["ATR"]},
        }
        with pytest.raises(ValidationError, match="must all be"):
            validate_config(config)

    def test_run_metrics_on_with_valid_metrics_ok(self, valid_config):
        """Tests valid metrics setup."""
        config = {
            **valid_config,
            "output": {**valid_config["output"], "run_metrics": True},
            "metrics": {"types": ["ATR"], "t_0": [2020], "H": [5]},
        }
        validate_config(config)  # does not raise

    def test_metrics_outside_time_range_raises(self, valid_config):
        """Tests a climate metric setup that goes beyond the time range."""
        config = {
            **valid_config,
            "output": {**valid_config["output"], "run_metrics": True},
            "metrics": {"types": ["ATR"], "t_0": [2025], "H": [100]},
        }
        with pytest.raises(
            ValidationError, match="outside the simulation time range"
        ):
            validate_config(config)


class TestConfigInlineAircraft:
    """Tests _AircraftConfig's dynamic [aircraft.<id>] entries, via Config"""

    def test_valid_inline_entry_derived(self, valid_config):
        """Tests a valid entry."""
        config = {
            **valid_config,
            "aircraft": {"types": ["AC1"], "AC1": {"PM": 3.0e15}},
        }
        result = validate_config(config)
        assert result["aircraft"]["AC1"]["PMrel"] == pytest.approx(2.0)

    def test_invalid_inline_entry_raises(self, valid_config):
        """Tests an invalid entry."""
        config = {
            **valid_config,
            "aircraft": {
                "types": ["AC1"],
                "AC1": {"Q_h": 43e6},  # SAC_eq missing - can't derive G_250
            },
        }
        with pytest.raises(ValidationError, match="Could not derive G_250"):
            validate_config(config)


class TestValidateConfig:
    """Tests function validate_config(config)"""

    def test_returns_plain_dict(self, valid_config):
        """Tests the return."""
        result = validate_config(dict(valid_config))
        assert isinstance(result, dict)
        assert not isinstance(result, Config)

    def test_fills_in_defaults(self, valid_config):
        """Tests that defaults are added."""
        result = validate_config(dict(valid_config))
        assert "temperature" in result
        assert "metrics" in result
        assert "parametric" in result

    def test_invalid_config_raises_validation_error(self):
        """Tests that an invalid entry raises a ValidationError."""
        with pytest.raises(ValidationError):
            validate_config({"species": {"inv": "not-a-list"}})
