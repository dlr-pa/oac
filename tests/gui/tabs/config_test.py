"""Provides tests for module openairclim.gui.tabs.config

The per-card required-field checks (_check_*, CARD_CHECKS,
check_required_fields, and their _is_blank/_get_path/_required_fields_status
helpers) now live in openairclim.gui.config_io - see config_io_test.py -
since they're pure config-dict logic with no Panel dependency, reused by
config_io.run_full_validation. Only the widget-facing helpers stay here.
"""

# since we are testing private helpers within the module, we ignore the
# corresponding pylint warning in this file
# pylint: disable=protected-access

import pytest

from openairclim.gui import config_io
from openairclim.gui.components.file_picker import FilePicker
from openairclim.gui.state import AppState
from openairclim.gui.tabs import config


class TestMissingLabel:
    """Tests functions _missing_label(filename) and
    _strip_missing_label(value).
    """

    @pytest.mark.parametrize(
        "value", [config._missing_label("file.nc"), "file.nc"],
        ids=["decorated", "plain"]
    )
    def test_strip_returns_real_filename(self, value):
        """Tests both the strip and pass-through branches of the function."""
        assert config._strip_missing_label(value) == "file.nc"


class TestResolveDirOrDefault:
    """Tests function _resolve_dir_or_default(working_dir, dir_str)"""

    def test_blank_falls_back_to_default_repository_dir(self, monkeypatch, tmp_path):
        """A blank dir_str resolves to config_io.default_repository_dir(),
        unlike _resolve_dir_or_none which would stay None."""
        monkeypatch.setattr(
            config.config_io, "default_repository_dir", lambda: tmp_path
        )
        assert config._resolve_dir_or_default("/working/dir", "") == tmp_path

    def test_explicit_dir_resolved_against_working_dir(self, tmp_path):
        """A non-blank dir_str is resolved via config_io.resolve_dir
        against working_dir, not the default."""
        sub = tmp_path / "custom"
        sub.mkdir()
        result = config._resolve_dir_or_default(str(tmp_path), "custom")
        assert result == sub.resolve()


def _capture_file_pickers(monkeypatch):
    """Patch FilePicker.__init__ to record every instance created, so
    tests can reach the folder-picker widget a section builder creates."""
    created = []
    orig_init = FilePicker.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(FilePicker, "__init__", patched_init)
    return created


class TestBackgroundResponsesDirFileSync:
    """A file dropdown's auto-suggested or cleared value must be written into
    the config dict directly, not solely via its "value changed" watcher."""

    def test_background_auto_suggested_file_persists(self, monkeypatch, tmp_path):
        """A file auto-suggested from BACKGROUND_FILE_DEFAULTS when the
        dir is (re)resolved is written into the config dict, even when
        the widget's displayed value doesn't itself change."""
        monkeypatch.setattr(
            config.config_io, "default_repository_dir", lambda: tmp_path / "cache"
        )
        (tmp_path / "co2_bg.nc").touch()
        pickers = _capture_file_pickers(monkeypatch)

        state = AppState()
        state.working_dir = str(tmp_path)
        edited = config_io.blank_config()

        config._build_background_section(state, edited, lambda: None)
        dir_picker = pickers[0]
        dir_picker.path = str(tmp_path)

        assert edited["background"]["CO2"]["file"] == "co2_bg.nc"

    def test_background_stale_file_cleared_when_missing(self, monkeypatch, tmp_path):
        """A previously-selected file no longer found in the newly
        resolved dir is cleared from the config dict too, not just from
        the widget."""
        monkeypatch.setattr(
            config.config_io, "default_repository_dir", lambda: tmp_path / "cache"
        )
        pickers = _capture_file_pickers(monkeypatch)

        state = AppState()
        state.working_dir = str(tmp_path)
        edited = config_io.blank_config()
        edited["background"]["CO2"] = {"file": "co2_bg.nc", "scenario": "SSP2-4.5"}

        config._build_background_section(state, edited, lambda: None)
        dir_picker = pickers[0]
        dir_picker.path = str(tmp_path)  # real, but empty -> co2_bg.nc not there

        assert edited["background"]["CO2"]["file"] == ""

    def test_responses_auto_suggested_file_persists(self, monkeypatch, tmp_path):
        """A file auto-suggested from RESPONSES_FILE_DEFAULTS when the
        dir is (re)resolved is written into the config dict, even when
        the widget's displayed value doesn't itself change."""
        monkeypatch.setattr(
            config.config_io, "default_repository_dir", lambda: tmp_path / "cache"
        )
        (tmp_path / "resp_RF.nc").touch()
        pickers = _capture_file_pickers(monkeypatch)

        state = AppState()
        state.working_dir = str(tmp_path)
        edited = config_io.blank_config()

        config._build_responses_section(state, edited, lambda: None)
        dir_picker = pickers[0]
        dir_picker.path = str(tmp_path)

        assert edited["responses"]["H2O"]["rf"]["file"] == "resp_RF.nc"

    def test_responses_stale_file_cleared_when_missing(self, monkeypatch, tmp_path):
        """A previously-selected response file no longer found in the
        newly resolved dir is cleared from the config dict too."""
        monkeypatch.setattr(
            config.config_io, "default_repository_dir", lambda: tmp_path / "cache"
        )
        pickers = _capture_file_pickers(monkeypatch)

        state = AppState()
        state.working_dir = str(tmp_path)
        edited = config_io.blank_config()
        edited["responses"]["H2O"]["rf"]["file"] = "resp_RF.nc"

        config._build_responses_section(state, edited, lambda: None)
        dir_picker = pickers[0]
        dir_picker.path = str(tmp_path)  # real, but empty -> resp_RF.nc not there

        assert edited["responses"]["H2O"]["rf"]["file"] == ""
