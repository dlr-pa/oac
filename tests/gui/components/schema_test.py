"""Provides tests for module openairclim.gui.components.schema"""

import pytest
from pydantic import BaseModel

from openairclim.gui.components import schema


class TestSubmodel:
    """Tests function submodel(path)"""

    def test_top_level_field(self):
        """Tests top-level fields, e.g. within the "species" submodel."""
        model = schema.submodel("species")
        assert issubclass(model, BaseModel)
        assert "inv" in model.model_fields

    def test_nested_field(self):
        """Tests nested fields loaded through a dotted path (e.g.
        "responses.CO2.rf")."""
        model = schema.submodel("responses.CO2.rf")
        assert "method" in model.model_fields
        assert "attr" in model.model_fields

    def test_unknown_field_raises(self):
        """Tests that a unknown field raises a KeyError."""
        with pytest.raises(KeyError):
            schema.submodel("does_not_exist")


class TestLiteralChoices:
    """Tests function literal_choices(model, field)"""

    def test_plain_literal_field(self):
        """Tests correct functionality using NOx choices."""
        species_model = schema.submodel("species")
        choices = schema.literal_choices(species_model, "nox")
        assert isinstance(choices, list)
        assert "NO" in choices

    def test_list_of_literal_field(self):
        """Tests correct functionality using output species."""
        species_model = schema.submodel("species")
        choices = schema.literal_choices(species_model, "out")
        assert isinstance(choices, list)
        assert "CO2" in choices

    def test_optional_literal_field(self):
        """Tests correct functionality of optional fields."""
        model = schema.submodel("responses.cont")
        choices = schema.literal_choices(model, "low_soot_case")
        assert isinstance(choices, list)


class TestIsStringLikeField:
    """Tests function is_string_like_field(model, field)"""

    def test_str_field_is_string_like(self):
        """Tests that a string field is correctly identified."""
        from openairclim.core.config_model import AircraftEntry
        assert schema.is_string_like_field(AircraftEntry, "SAC_eq") is True

    def test_float_field_is_not_string_like(self):
        """Tests that a float field is correctly identified."""
        from openairclim.core.config_model import AircraftEntry
        assert schema.is_string_like_field(AircraftEntry, "PMrel") is False
