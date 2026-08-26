"""Provides tests for module openairclim.gui.tabs.config_text"""

# since we are testing private helpers within the module, we ignore the
# corresponding pylint warning in this file
# pylint: disable=protected-access

from openairclim.gui import config_io
from openairclim.gui.state import AppState
from openairclim.gui.tabs import config_text


class TestSerialize:
    """Tests function _serialize(state)"""

    def test_no_config_returns_empty_string(self):
        """Tests that a blank/missing config returns an empty string."""
        state = AppState()
        assert config_text._serialize(state) == ""

    def test_serializes_edited_config_to_toml(self, loaded_state):
        """Tests the serialisation of the edited config - ensures that it is
        in the format we desire (i.e. with sections like `[species]`)."""
        text = config_text._serialize(loaded_state)
        prepared = config_io.prepare_for_save(
            loaded_state.edited_config, loaded_state.working_dir
        )
        assert text == config_io.to_toml_string(prepared)
        assert "[species]" in text
