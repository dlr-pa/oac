"""Provides tests for module openairclim.core.utils"""

import pytest

from openairclim.core import utils


class TestToPintUnits:
    """Tests function to_pint_units(unit_str)"""

    def test_blank_becomes_dimensionless(self):
        """Tests that a blank unit becomes dimensionless."""
        assert utils.to_pint_units("") == "1"
        assert utils.to_pint_units(None) == "1"

    def test_simple_unit_unchanged(self):
        """Tests that a simple unit is left unchanged."""
        assert utils.to_pint_units("kg") == "kg"

    def test_compound_rate_rewritten(self):
        """Tests that a UDUNITS-style rate is rewritten into pint syntax."""
        assert utils.to_pint_units("Tg yr-1") == "Tg*yr**-1"

    def test_compound_multi_exponent_rewritten(self):
        """Tests a multi-term compound unit with negative exponents."""
        assert utils.to_pint_units("kg m-2 s-1") == "kg*m**-2*s**-1"

    def test_pint_syntax_input_raises(self):
        """Tests that an already-pint-syntax string (containing '**') is
        rejected with a clear error, rather than being mangled into
        invalid syntax (e.g. "kg**-1" -> "kg****-1")."""
        with pytest.raises(ValueError):
            utils.to_pint_units("kg**-1")


class TestQuantity:
    """Tests function quantity(value, unit_str)"""

    def test_builds_quantity(self):
        """Tests that a plain value/unit pair builds a pint Quantity."""
        qty = utils.quantity(5, "kg")
        assert qty.magnitude == pytest.approx(5.0)

    def test_unparseable_unit_raises_value_error(self):
        """Tests that a garbage unit string raises ValueError, not an
        internal pint/TypeError."""
        with pytest.raises(ValueError):
            utils.quantity(1, "incorrect-unit")


class TestToValue:
    """Tests function to_value(qty, target_units)"""

    def test_converts_to_target(self):
        """Tests conversion of a Quantity to a target unit."""
        qty = utils.quantity(1, "km")
        assert utils.to_value(qty, "m") == pytest.approx(1000.0)

    def test_incompatible_units_raise(self):
        """Tests an incompatible unit conversion raises ValueError."""
        qty = utils.quantity(1, "kg")
        with pytest.raises(ValueError):
            utils.to_value(qty, "m")


class TestConvertUnits:
    """Tests function convert_units(value, src_units, target_units)"""

    def test_simple_conversion(self):
        """Tests conversion between km and m."""
        assert utils.convert_units(1, "km", "m") == pytest.approx(1000.0)

    def test_identity_conversion(self):
        """Tests conversion from and to the same unit."""
        assert utils.convert_units(5, "kg", "kg") == pytest.approx(5.0)

    def test_kg_tg_roundtrip(self):
        """Tests the kg <-> Tg conversion previously covered by
        kg_to_tg/tg_to_kg."""
        assert utils.convert_units(1.0, "kg", "Tg") == pytest.approx(1.0e-9)
        assert utils.convert_units(1.0, "Tg", "kg") == pytest.approx(1.0e9)
