"""
Provides tests for module openairclim.utils.create_test_files
"""

import tomllib

import pytest

from openairclim.utils import create_test_files as ctf


class TestCreateTestConfigFiles:
    """Tests function create_test_config_files(repo_path, valid_name, invalid_name)"""

    def test_valid_file_is_parseable_toml(self, tmp_path):
        """The 'valid' file is actually valid TOML with the expected
        key/value pair."""
        ctf.create_test_config_files(str(tmp_path), "valid.toml", "invalid.toml")
        with open(tmp_path / "valid.toml", "rb") as valid_file:
            config = tomllib.load(valid_file)
        assert config == {"key": "value"}

    def test_invalid_file_is_not_parseable_toml(self, tmp_path):
        """The 'invalid' file genuinely fails to parse as TOML."""
        ctf.create_test_config_files(str(tmp_path), "valid.toml", "invalid.toml")
        with open(tmp_path / "invalid.toml", "rb") as invalid_file:
            with pytest.raises(tomllib.TOMLDecodeError):
                tomllib.load(invalid_file)
