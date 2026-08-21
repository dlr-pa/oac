"""Provides tests for module openairclim.gui.components.file_picker"""

from openairclim.gui.components.file_picker import FilePicker


class TestFilePicker:
    """Tests widget FilePicker"""

    def test_set_path_updates_path_param(self):
        """Tests `set_path` function for string input."""
        fp = FilePicker(label="Folder", directory=True)
        fp.set_path("/some/dir")
        assert fp.path == "/some/dir"

    def test_initial_value_reflected_in_text_input(self):
        """Tests `path` initial variable for `FilePicker` class."""
        fp = FilePicker(label="Folder", path="/preset")
        assert fp._text_input.value == "/preset"  # pylint: disable=protected-access
