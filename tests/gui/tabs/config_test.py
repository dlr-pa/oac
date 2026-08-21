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
