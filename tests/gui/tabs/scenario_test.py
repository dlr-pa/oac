"""Provides tests for module openairclim.gui.tabs.scenario"""

# since we are testing private helpers within the module, we ignore the
# corresponding pylint warning in this file
# pylint: disable=protected-access

import pytest

from openairclim.gui.tabs import scenario


class TestUnitStr:
    """Tests function _unit_str(raw)"""

    def test_blank_becomes_dimensionless(self):
        """Tests that a blank unit becomes dimensionless."""
        assert scenario._unit_str("") == "1"
        assert scenario._unit_str(None) == "1"

    def test_non_blank_unchanged(self):
        """Tests that a non-blank unit remains unchanged."""
        assert scenario._unit_str("kg") == "kg"


class TestConvertValue:
    """Tests function _convert_value(value, src_units, target_units, per_year)"""

    def test_simple_conversion(self):
        """Tests conversion between km and m."""
        assert scenario._convert_value(1, "km", "m") == pytest.approx(1000.0)

    def test_identity_conversion(self):
        """Tests conversion from and to the same unit."""
        assert scenario._convert_value(5, "kg", "kg") == pytest.approx(5.0)

    def test_blank_units_treated_as_dimensionless(self):
        """Tests that blank units are treated as dimensionless."""
        assert scenario._convert_value(2, "", "1") == pytest.approx(2.0)

    def test_per_year_rate_cancels_to_total(self):
        """Tests converting a rate (e.g. Tg yr-1) to a yearly total (Tg)."""
        result = scenario._convert_value(5, "Tg yr-1", "Tg", per_year=True)
        assert result == pytest.approx(5.0)

    def test_incompatible_units_raise(self):
        """Tests an incompatible unit conversion."""
        with pytest.raises(ValueError):
            scenario._convert_value(1, "kg", "m")


class TestConvertRatio:
    """Tests function _convert_ratio(value, numerator_units, denominator_units,
    target_units)"""

    def test_same_units_cancel_to_dimensionless(self):
        """Tests the unit conversion to dimensionless."""
        result = scenario._convert_ratio(3, "kg", "kg", "1")
        assert result == pytest.approx(3.0)

    def test_unit_conversion_within_ratio(self):
        """Tests that a unit conversion to dimensionless still works for 
        different units of the same type (e.g. mass)."""
        result = scenario._convert_ratio(1, "g", "kg", "1")
        assert result == pytest.approx(0.001)

    def test_blank_units_treated_as_dimensionless(self):
        """Tests that blank units are treated as dimensionless."""
        result = scenario._convert_ratio(4, "", "", "1")
        assert result == pytest.approx(4.0)
