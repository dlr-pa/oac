"""Provides tests for module openairclim.gui.tabs.aircraft"""

# since we are testing private helpers within the module, we ignore the
# corresponding pylint warning in this file
# pylint: disable=protected-access

import pandas as pd
import pytest

from openairclim.gui.tabs import aircraft

# A minimal, valid set of sub-values that derives G_250 for a conventional
# (SAC_eq="CON") aircraft - mirrors core.calc_cont.calc_sac_slope's
# requirements for that case (ei_h2o, eta, q_h).
CON_SUBVALUES = {"SAC_eq": "CON", "Q_h": 43e6, "eta": 0.3, "EIH2O": 1.25}


class TestDeriveEntry:
    """Tests function _derive_entry(row)"""

    def test_blank_row_returns_none(self):
        """Tests that a blank row returns `None`."""
        row = {c: None for c in aircraft._DATA_FIELDS}
        assert aircraft._derive_entry(row) is None

    def test_valid_subvalues_derive_g250(self):
        """Tests valid combination of sub-values."""
        entry = aircraft._derive_entry(CON_SUBVALUES)
        assert entry is not None
        assert entry.G_250 is not None

    def test_incomplete_subvalues_return_none(self):
        """Tests that incomplete sub-values returns `None`."""
        row = {"Q_h": 43e6}
        assert aircraft._derive_entry(row) is None

    def test_pm_only_derives_pmrel_not_g250(self):
        """Tests the derivation of only PMrel."""
        entry = aircraft._derive_entry({"PM": 1.5e15})
        assert entry is not None
        assert entry.PMrel == pytest.approx(1.0)
        assert entry.G_250 is None


class TestComputeG250Preview:
    """Tests function _compute_g250_preview(row)"""

    def test_derivable(self):
        """Tests that a G_250 value is calculated for valid sub-values."""
        assert aircraft._compute_g250_preview(CON_SUBVALUES) is not None

    def test_not_derivable_returns_none(self):
        """Tests a blank input."""
        assert aircraft._compute_g250_preview({}) is None


class TestComputePMrelPreview:
    """Tests function _compute_pmrel_preview(row)"""

    def test_derivable(self):
        """Tests that a PMrel value is calculated for valid sub-values."""
        result = aircraft._compute_pmrel_preview({"PM": 3.0e15})
        assert result == pytest.approx(2.0)

    def test_not_derivable_returns_none(self):
        """Tests a blank input."""
        assert aircraft._compute_pmrel_preview({}) is None


class TestBuildTableDf:
    """Tests function _build_table_df(edited, csv_df)"""

    def test_config_sourced_row(self):
        """Tests inline config data."""
        edited = {"aircraft": {"types": ["AC1"], "AC1": {"b": 30.0}}}
        df = aircraft._build_table_df(edited, aircraft._empty_csv_df())
        row = df[df["ac"] == "AC1"].iloc[0]
        assert row["source"] == "config"
        assert row["b"] == 30.0

    def test_csv_sourced_row(self):
        """Tests data from external csv file."""
        edited = {"aircraft": {"types": ["AC2"]}}
        csv_df = pd.DataFrame([{"ac": "AC2", "b": 40.0}])
        df = aircraft._build_table_df(edited, csv_df)
        row = df[df["ac"] == "AC2"].iloc[0]
        assert row["source"] == "csv"
        assert row["b"] == 40.0

    def test_bare_identifier_with_no_data(self):
        """Tests empty csv file input."""
        edited = {"aircraft": {"types": ["AC3"]}}
        df = aircraft._build_table_df(edited, aircraft._empty_csv_df())
        row = df[df["ac"] == "AC3"].iloc[0]
        assert row["source"] == "config"
        assert aircraft._is_blank(row["b"])

    def test_config_wins_over_csv_when_ac_in_both(self):
        """Tests that data defined in the config overwrites data defined in
        an external csv file."""
        edited = {"aircraft": {"types": ["AC4"], "AC4": {"b": 50.0}}}
        csv_df = pd.DataFrame([{"ac": "AC4", "b": 99.0}])
        df = aircraft._build_table_df(edited, csv_df)
        row = df[df["ac"] == "AC4"].iloc[0]
        assert row["source"] == "config"
        assert row["b"] == 50.0

    def test_no_rows_returns_empty_df(self):
        """Tests that an empty aircraft.types produces an empty dataframe."""
        df = aircraft._build_table_df(
            {"aircraft": {"types": []}}, aircraft._empty_csv_df()
        )
        assert df.empty
        assert list(df.columns) == aircraft.TABLE_COLUMNS

    def test_none_edited_returns_empty_df(self):
        """Tests that a missing aircraft.types produces an empty dataframe."""
        df = aircraft._build_table_df(None, aircraft._empty_csv_df())
        assert df.empty

    def test_leftover_config_entry_not_in_types_still_included(self):
        """Tests that aircraft data is loaded into the dataframe even if the
        identifier is not included in aircraft.types. This could happen if the
        config file is hand-edited. The GUI automatically fixes this issue."""
        edited = {"aircraft": {"types": [], "AC5": {"b": 60.0}}}
        df = aircraft._build_table_df(edited, aircraft._empty_csv_df())
        row = df[df["ac"] == "AC5"].iloc[0]
        assert row["source"] == "config"
        assert row["b"] == 60.0

    def test_leftover_csv_entry_not_in_types_still_included(self):
        """Tests that aircraft data in a csv file is loaded into the dataframe
        even if the identifier is not included in aircraft.types. This could
        happen if the csv or config files are hand-edited. The GUI
        automatically fixes this issue."""
        edited = {"aircraft": {"types": []}}
        csv_df = pd.DataFrame([{"ac": "AC6", "b": 70.0}])
        df = aircraft._build_table_df(edited, csv_df)
        row = df[df["ac"] == "AC6"].iloc[0]
        assert row["source"] == "csv"
        assert row["b"] == 70.0
