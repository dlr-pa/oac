"""Aircraft tab: define aircraft identifiers and per-aircraft contrail
parameters. This data can live in two places:

- inline in the config, under `config["aircraft"][<ac_id>]`; or
- in a separate CSV file (`config["aircraft"]["dir"]`/`["file"]`.

Both sources support deriving `G_250` from sub-values (`SAC_eq`, `Q_h`, `eta`,
`eta_elec`, `EIH2O`, `R`) and `PMrel` from `PM`, when the direct value is
missing: csv rows via `load_ac_data`, config-inline entries via
`_derive_missing_ac_params`. This tab lets either be edited directly or
derived: the"Calculate G_250/PMrel from sub-values" button can be used to fill
in any blank G_250/PMrel cells from their sub-values.

A direct value and its sub-values may both be present at once (e.g. right after
"Calculate", or because a user keeps sub-values on hand for reference) - core
always silently prefers the direct value and ignores the sub-values. This tab
never deletes sub-values on the user's behalf; the completeness check only
warns when they're both present *and* actually disagree (compares the
sub-value-derived value against the stored direct value, within the same
rounding tolerance core uses when deriving one itself).

Table edits are synced into `edited_config`/the local CSV state immediately,
but the completeness/consistency check ("Check completeness" button) and the
sub-value calculation only run on button-press, rather than on every keystroke,
to keep large tables responsive.

Core does not tolerate an aircraft identifier defined in-line in the config
*and* in a linked csv file. Therefore, this tab keeps the two mutually
exclusive per identifier: one row per aircraft in a single table, with a
"source" column. Changing a row's source moves its data between `edited_config`
and a locally-held CSV DataFrame, which is only written to disk on Save
(mirroring the sidebar's config Save). `edited_config["aircraft"]["types"]` is
kept in sync automatically.
"""

import math
from pathlib import Path
from typing import Any

import pandas as pd
import panel as pn
from pydantic import ValidationError

from ..components.schema import literal_choices, is_string_like_field
from ..components.utils import load_inventory
from ..state import AppState
from ...core.config_model import AircraftEntry, AIRCRAFT_DERIVATION_MAP

# Field names/order, SAC_eq's valid values, and the G_250/PMrel sub-value
# groupings all come from config_model.AircraftEntry - the same schema core
# validates aircraft data against - rather than being duplicated here.
_DATA_FIELDS = list(AircraftEntry.model_fields)
G250_SUBCOLS = AIRCRAFT_DERIVATION_MAP["G_250"]
PMREL_SUBCOLS = AIRCRAFT_DERIVATION_MAP["PMrel"]
SAC_EQ_OPTIONS = ["", *literal_choices(AircraftEntry, "SAC_eq")]

CSV_COLUMNS = ["ac", *_DATA_FIELDS]
TABLE_COLUMNS = [*CSV_COLUMNS, "source"]
SOURCE_OPTIONS = ["config", "csv"]

_STR_COLS = ["ac", *(f for f in _DATA_FIELDS if is_string_like_field(AircraftEntry, f))]
_FLOAT_COLS = [f for f in _DATA_FIELDS if f not in _STR_COLS]

TITLE = """
### Edit aircraft data
This is an **editing** tab - changes you make here update the shared working
configuration immediately.

Use this tab to view and edit the aircraft data. If the main or base emission
inventories include different aircraft identifiers (`ac` data variable), then
all aircraft IDs must be defined below. The corresponding variables `b`,
`PMrel` and `G_250` are required only if contrails are to be calculated. This
data can be provided in-line in the config file itself, or in an accompanying
csv file — the "source" column controls which, per aircraft. An appropriate
csv file can be created using the buttons below.

For more information, see Megill (2026) and the OpenAirClim
[docs](https://openairclim.org/user_guide/contrails.html).
"""


def _empty_table_df() -> pd.DataFrame:
    """Return an empty, correctly-typed table DataFrame."""
    cols = {
        c: pd.Series(dtype="object" if c in _STR_COLS else "float64")
        for c in TABLE_COLUMNS
        if c != "source"
    }
    cols["source"] = pd.Series(dtype="object")
    return pd.DataFrame(cols)[TABLE_COLUMNS]


def _empty_csv_df() -> pd.DataFrame:
    """Return an empty, correctly-typed CSV DataFrame."""
    cols = {
        c: pd.Series(dtype="object" if c in _STR_COLS else "float64")
        for c in CSV_COLUMNS
    }
    return pd.DataFrame(cols)[CSV_COLUMNS]


def _is_blank(value: Any) -> bool:
    """Return True if a table/CSV cell counts as "not filled in"."""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _derive_entry(row: Any) -> AircraftEntry | None:
    """Validate a row's G250_SUBCOLS/PMREL_SUBCOLS through AircraftEntry,
    deriving G_250/PMrel the same way core does (config_model.AircraftEntry),
    so the preview always matches what core would actually compute.

    Args:
        row (pandas.Series or dict): Row with sub-value columns.

    Returns:
        AircraftEntry or None: The validated entry, or None if the given
            sub-values are absent or aren't enough to derive anything.
    """
    kwargs = {
        c: row.get(c)
        for c in (*G250_SUBCOLS, *PMREL_SUBCOLS)
        if not _is_blank(row.get(c))
    }
    if not kwargs:
        return None
    try:
        return AircraftEntry.model_validate(kwargs)
    except ValidationError:
        return None


def _compute_g250_preview(row: Any) -> float | None:
    """Return G_250 computed from sub-values, or None if not derivable.

    Args:
        row (pandas.Series or dict): Row with G250_SUBCOLS values.

    Returns:
        float or None: Computed G_250, or None if not derivable.
    """
    entry = _derive_entry(row)
    return entry.G_250 if entry is not None else None


def _compute_pmrel_preview(row: Any) -> float | None:
    """Return PMrel computed from PM (PMrel = PM / 1.5e15), or None if not
    derivable.

    Args:
        row (pandas.Series or dict): Row with a "PM" value.

    Returns:
        float or None: Computed PMrel, or None if not derivable.
    """
    entry = _derive_entry(row)
    return entry.PMrel if entry is not None else None


def _build_table_df(edited: dict, csv_df: pd.DataFrame) -> pd.DataFrame:
    """Merge config-sourced and csv-sourced aircraft data into one table.

    Args:
        edited (dict): Working configuration dict.
        csv_df (pandas.DataFrame): Locally-held CSV data (CSV_COLUMNS).

    Returns:
        pandas.DataFrame: One row per aircraft identifier, TABLE_COLUMNS.
    """
    aircraft = edited.get("aircraft", {}) if edited else {}
    types = [t for t in aircraft.get("types", []) if not _is_blank(t)]
    config_rows = {
        k: v
        for k, v in aircraft.items()
        if k not in ("types", "dir", "file") and isinstance(v, dict)
    }
    csv_lookup = {}
    if csv_df is not None and not csv_df.empty:
        for _, row in csv_df.iterrows():
            if not _is_blank(row.get("ac")):
                csv_lookup[str(row["ac"])] = row

    data_cols = _DATA_FIELDS
    seen = set()
    rows = []

    def _row_from_config(ac: str) -> dict:
        d = config_rows.get(ac, {})
        row = {c: d.get(c) for c in data_cols}
        row["ac"] = ac
        row["source"] = "config"
        return row

    def _row_from_csv(ac: str) -> dict:
        r = csv_lookup[ac]
        row = {c: r.get(c) for c in data_cols}
        row["ac"] = ac
        row["source"] = "csv"
        return row

    for ac in types:
        if ac in seen:
            continue
        seen.add(ac)
        if ac in config_rows:
            rows.append(_row_from_config(ac))
        elif ac in csv_lookup:
            rows.append(_row_from_csv(ac))
        else:
            # A bare identifier with no data anywhere yet - valid as
            # long as contrails aren't being computed.
            blank_row: dict = {c: None for c in data_cols}
            blank_row["ac"] = ac
            blank_row["source"] = "config"
            rows.append(blank_row)

    # Data present in a source but not (yet) listed in "types" - shouldn't
    # happen once this tab is the one maintaining "types", but a
    # hand-edited or externally-loaded config could have this.
    for ac in config_rows:
        if ac not in seen:
            seen.add(ac)
            rows.append(_row_from_config(ac))
    for ac in csv_lookup:
        if ac not in seen:
            seen.add(ac)
            rows.append(_row_from_csv(ac))

    if not rows:
        return _empty_table_df()
    return pd.DataFrame(rows)[TABLE_COLUMNS]


# pylint: disable-next=too-many-statements,too-many-locals
def panel(state: AppState) -> pn.Column:
    """Return the aircraft tab content.

    Args:
        state (AppState): Shared application state.
    """
    _csv = {"df": _empty_csv_df()}
    _inv_ac_cache: dict[str, Any] = {"key": None, "values": set()}
    _rebuilding = {"flag": False}
    pending_action: dict[str, Any] = {"type": None}

    status_pane = pn.pane.Markdown("")
    csv_status = pn.pane.Markdown("")
    check_status = pn.pane.Markdown("")

    open_btn = pn.widgets.Button(name="Open CSV…", button_type="primary")
    new_btn = pn.widgets.Button(name="New CSV", button_type="default")
    save_btn = pn.widgets.Button(name="Save CSV", button_type="success")
    save_as_btn = pn.widgets.Button(name="Save CSV As…", button_type="default")
    unlink_btn = pn.widgets.Button(name="Unlink CSV from config", button_type="warning")
    add_row_btn = pn.widgets.Button(name="Add row", button_type="default")
    delete_row_btn = pn.widgets.Button(
        name="Delete selected row(s)", button_type="danger"
    )
    calculate_btn = pn.widgets.Button(
        name="Calculate G_250/PMrel from sub-values", button_type="primary"
    )
    check_btn = pn.widgets.Button(name="Check completeness", button_type="default")

    confirm_msg = pn.pane.Markdown(
        "⚠️ You have unsaved edits to the aircraft CSV. Continuing will "
        "discard them."
    )
    confirm_yes = pn.widgets.Button(name="Discard and continue", button_type="danger")
    confirm_no = pn.widgets.Button(name="Cancel", button_type="default")
    confirm_row = pn.Column(confirm_msg, pn.Row(confirm_yes, confirm_no), visible=False)

    table = pn.widgets.Tabulator(
        _empty_table_df(),
        show_index=False,
        selectable="checkbox",
        editors={
            "ac": "input",
            # Numeric columns are deliberately absent here
            "SAC_eq": {"type": "list", "values": SAC_EQ_OPTIONS},
            "source": {"type": "list", "values": SOURCE_OPTIONS},
        },
        titles={
            "ac": "Aircraft ID",
            "b": "b [m]",
            "PMrel": "PMrel [-]",
            "G_250": "G_250 [Pa/K]",
            "SAC_eq": "SAC eq.",
            "Q_h": "Q or Δh",
            "eta": "eta [-]",
            "eta_elec": "eta elec. [-]",
            "EIH2O": "EIH2O [kg/kg]",
            "R": "R",
            "PM": "PM [1/kg]",
            "source": "Source",
        },
        frozen_columns=["ac"],
        sizing_mode="stretch_width",
    )

    # ── helpers ───────────────────────────────────────────────────────

    def _current_csv_path() -> str | None:
        config = state.edited_config
        if not config:
            return None
        aircraft = config.get("aircraft", {})
        f = aircraft.get("file")
        if not f:
            return None
        d = aircraft.get("dir", "")
        return str(Path(state.working_dir) / d / f)

    def _load_inventory_ac_values() -> set:
        """Unique "ac" values across loaded inventories, cached until the
        inventory dir/file list actually changes. Inventories without an
        "ac" variable contribute "DEFAULT", matching read_netcdf.py's
        split_inventory_by_aircraft."""
        config = state.edited_config
        if not config:
            return set()
        inv_cfg = config.get("inventories", {})
        inv_dir = inv_cfg.get("dir", "")
        inv_files = tuple(inv_cfg.get("files", []))
        key = (state.working_dir, inv_dir, inv_files)
        if _inv_ac_cache["key"] == key:
            return _inv_ac_cache["values"]

        ac_values: set = set()
        any_no_ac = False
        if inv_files and state.working_dir:
            for f in inv_files:
                try:
                    ds = load_inventory(state.working_dir, inv_dir, f)
                except Exception:  # pylint: disable=broad-exception-caught
                    continue
                if "ac" in ds.data_vars:
                    ac_values.update(str(v) for v in ds["ac"].values.tolist())
                else:
                    any_no_ac = True
            if any_no_ac:
                ac_values.add("DEFAULT")

        _inv_ac_cache["key"] = key
        _inv_ac_cache["values"] = ac_values
        return ac_values

    def _run_check() -> None:  # pylint: disable=too-many-locals
        """Update check_status with, in order:

        1. Ambiguous rows - G_250 set directly *and* via sub-values (core
           silently keeps the direct value and ignores the sub-values),
           same for PMrel/PM. Checked regardless of "cont"/inventories.
        2. Aircraft identifiers found in inventories but not defined here.
        3. Only if "cont" is in species.out: do those identifiers have
           complete G_250/b/PMrel data?
        """
        config = state.edited_config
        if not config:
            check_status.object = ""
            return

        messages = []
        df = table.value

        # a direct value *and* its sub-values can both legitimately be
        # present at once. Only warn when the two disagree.
        conflicting_g250 = [
            row["ac"]
            for _, row in df.iterrows()
            if not _is_blank(row.get("ac"))
            and not _is_blank(row.get("G_250"))
            and any(not _is_blank(row.get(c)) for c in G250_SUBCOLS)
            and (derived := _compute_g250_preview(row)) is not None
            and not math.isclose(
                derived, float(row["G_250"]), rel_tol=1e-3, abs_tol=1e-3
            )
        ]
        conflicting_pmrel = [
            row["ac"]
            for _, row in df.iterrows()
            if not _is_blank(row.get("ac"))
            and not _is_blank(row.get("PMrel"))
            and not _is_blank(row.get("PM"))
            and (derived := _compute_pmrel_preview(row)) is not None
            and not math.isclose(
                derived, float(row["PMrel"]), rel_tol=1e-3, abs_tol=1e-3
            )
        ]
        if conflicting_g250:
            messages.append(
                "⚠️ G_250 is set directly to a value that doesn't match what "
                "its sub-values (SAC_eq/Q_h/...) compute for: "
                f"{', '.join(conflicting_g250)} - the direct value will be "
                "used, the sub-values ignored."
            )
        if conflicting_pmrel:
            messages.append(
                "⚠️ PMrel is set directly to a value that doesn't match "
                f"PM/1.5e15 for: {', '.join(conflicting_pmrel)} - the direct "
                "value will be used, PM ignored."
            )

        inv_ac = _load_inventory_ac_values()
        if inv_ac:
            types = set(config.get("aircraft", {}).get("types", []))
            missing_types = inv_ac - types
            if missing_types:
                messages.append(
                    "⚠️ Aircraft identifier(s) found in emission inventories "
                    f"but not defined here: {', '.join(sorted(missing_types))}"
                )
            elif "cont" in config.get("species", {}).get("out", []):
                incomplete = []
                for ac in sorted(inv_ac):
                    match = df[df["ac"] == ac]
                    if match.empty:
                        incomplete.append(ac)
                        continue
                    row = match.iloc[0]
                    has_b = not _is_blank(row.get("b"))
                    has_g250 = (
                        not _is_blank(row.get("G_250"))
                        or _compute_g250_preview(row) is not None
                    )
                    has_pmrel = (
                        not _is_blank(row.get("PMrel"))
                        or _compute_pmrel_preview(row) is not None
                    )
                    if not (has_b and has_g250 and has_pmrel):
                        incomplete.append(ac)
                if incomplete:
                    messages.append(
                        "⚠️ Contrails are enabled, but G_250/b/PMrel are "
                        f"incomplete for: {', '.join(incomplete)}"
                    )

        if not messages and inv_ac:
            messages.append(
                "✅ All aircraft identifiers used in the inventories are "
                "fully defined."
            )

        check_status.object = "\n\n".join(messages)

    def _sync_types() -> None:
        config = state.edited_config
        if not config:
            return
        acs = [a for a in table.value["ac"].tolist() if not _is_blank(a)]
        config.setdefault("aircraft", {})["types"] = sorted(dict.fromkeys(acs))

    def _clean_value(col: str, value: Any) -> Any:
        """Coerce a raw cell value to what should be stored for `col`."""
        if _is_blank(value):
            return None
        return str(value) if col == "SAC_eq" else float(value)

    def _sync_row(row: Any, source: str, old_ac: str | None = None) -> None:
        """Write one row's data into whichever store matches `source`,
        removing it from the other store (and from `old_ac` if renamed).

        Args:
            row (pandas.Series): The row's current data (TABLE_COLUMNS).
            source (str): "config" or "csv".
            old_ac (str, optional): Previous aircraft ID, if renamed.
        """
        config = state.edited_config
        if not config:
            return
        ac = row["ac"]
        aircraft = config.setdefault("aircraft", {})
        data_cols = _DATA_FIELDS

        for name in {ac, old_ac} - {None}:
            aircraft.pop(name, None)
            _csv["df"] = _csv["df"][_csv["df"]["ac"] != name]

        if source == "config":
            data = {
                c: _clean_value(c, row.get(c))
                for c in data_cols
                if not _is_blank(row.get(c))
            }
            aircraft[ac] = data
            state.dirty = True
        else:
            new_row = pd.DataFrame(
                [
                    {
                        "ac": ac,
                        **{c: _clean_value(c, row.get(c)) for c in data_cols},
                    }
                ]
            ).astype(_csv["df"].dtypes.to_dict())
            _csv["df"] = pd.concat([_csv["df"], new_row], ignore_index=True)
            state.aircraft_csv_dirty = True

    def _rebuild_table() -> None:
        config = state.edited_config
        _rebuilding["flag"] = True
        try:
            table.value = (
                _build_table_df(config, _csv["df"]) if config else _empty_table_df()
            )
        finally:
            _rebuilding["flag"] = False
        check_status.object = ""

    # ── table edits ──────────────────────────────────────────────────

    def _on_table_edit(event: Any) -> None:
        if _rebuilding["flag"]:
            return
        row = table.value.loc[event.row]
        ac = row["ac"]
        if _is_blank(ac):
            return  # nothing to sync yet - aircraft not named

        old_ac = (
            event.old if event.column == "ac" and not _is_blank(event.old) else None
        )
        _sync_row(row, row["source"], old_ac=old_ac)
        _sync_types()
        state.param.trigger("edited_config")

    table.on_edit(_on_table_edit)

    def _on_add_row(_event: Any = None) -> None:
        blank = {
            c: ("" if c == "ac" else ("config" if c == "source" else None))
            for c in TABLE_COLUMNS
        }
        new_row = pd.DataFrame([blank]).astype(
            {c: dt for c, dt in table.value.dtypes.items() if c in blank}
        )
        table.value = pd.concat([table.value, new_row], ignore_index=True)

    def _on_delete_rows(_event: Any = None) -> None:
        if not table.selection:
            return
        config = state.edited_config
        rows = table.value.loc[table.selection]
        for _, row in rows.iterrows():
            ac = row["ac"]
            if _is_blank(ac):
                continue
            if row["source"] == "config" and config:
                config.get("aircraft", {}).pop(ac, None)
                state.dirty = True
            else:
                _csv["df"] = _csv["df"][_csv["df"]["ac"] != ac]
                state.aircraft_csv_dirty = True
        table.value = table.value.drop(index=table.selection).reset_index(drop=True)
        table.selection = []
        _sync_types()
        if config:
            state.param.trigger("edited_config")

    add_row_btn.on_click(_on_add_row)
    delete_row_btn.on_click(_on_delete_rows)

    def _on_calculate_click(_event: Any = None) -> None:
        """Fill blank G_250/PMrel cells from their sub-values, using the
        same core calculation as the completeness check, then sync only
        the rows that actually changed."""
        df = table.value.copy()
        changed_idx = []
        for idx, row in df.iterrows():
            if _is_blank(row.get("ac")):
                continue
            row_changed = False
            if _is_blank(row.get("G_250")):
                g250 = _compute_g250_preview(row)
                if g250 is not None:
                    df.at[idx, "G_250"] = g250
                    row_changed = True
            if _is_blank(row.get("PMrel")):
                pmrel = _compute_pmrel_preview(row)
                if pmrel is not None:
                    df.at[idx, "PMrel"] = pmrel
                    row_changed = True
            if row_changed:
                changed_idx.append(idx)

        if not changed_idx:
            check_status.object = "ℹ️ Nothing to calculate."
            return

        _rebuilding["flag"] = True
        try:
            table.value = df
        finally:
            _rebuilding["flag"] = False

        config = state.edited_config
        for idx in changed_idx:
            row = df.loc[idx]
            _sync_row(row, row["source"])
        _sync_types()
        if config:
            state.param.trigger("edited_config")
        _run_check()

    calculate_btn.on_click(_on_calculate_click)
    check_btn.on_click(lambda event=None: _run_check())

    # ── CSV open / new / save ───────────────────────────────────────

    def _do_open_csv(path: str) -> None:
        try:
            df = pd.read_csv(path)
        except Exception as e:  # pylint: disable=broad-exception-caught
            csv_status.object = f"❌ Failed to read CSV: {e}"
            return
        df.columns = df.columns.str.strip()
        if "ac" not in df.columns:
            csv_status.object = "❌ CSV must have an 'ac' column."
            return
        for col in CSV_COLUMNS:
            if col not in df.columns:
                df[col] = None
        _csv["df"] = df[CSV_COLUMNS].copy()

        config = state.edited_config
        if config:
            aircraft = config.setdefault("aircraft", {})
            aircraft["dir"] = str(Path(path).parent)
            aircraft["file"] = Path(path).name
            state.dirty = True
            state.param.trigger("edited_config")

        state.aircraft_csv_dirty = False
        csv_status.object = f"ℹ️ Loaded `{Path(path).name}`."
        _rebuild_table()

    def _on_open_click(_event: Any = None) -> None:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(
            title="Select aircraft CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=state.working_dir or None,
        )
        root.destroy()
        if selected:
            _do_open_csv(selected)

    def _on_new_click(_event: Any = None) -> None:
        _csv["df"] = _empty_csv_df()
        config = state.edited_config
        if config:
            aircraft = config.setdefault("aircraft", {})
            aircraft["dir"] = ""
            aircraft["file"] = ""
            state.dirty = True
            state.param.trigger("edited_config")
        state.aircraft_csv_dirty = False
        csv_status.object = "ℹ️ Started a new blank aircraft CSV."
        _rebuild_table()

    def _request_open(_event: Any = None) -> None:
        if state.aircraft_csv_dirty:
            pending_action["type"] = "open"
            confirm_row.visible = True
        else:
            _on_open_click()

    def _request_new(_event: Any = None) -> None:
        if state.aircraft_csv_dirty:
            pending_action["type"] = "new"
            confirm_row.visible = True
        else:
            _on_new_click()

    def _on_confirm_yes(_event: Any = None) -> None:
        confirm_row.visible = False
        action = pending_action["type"]
        pending_action["type"] = None
        if action == "open":
            _on_open_click()
        elif action == "new":
            _on_new_click()

    def _on_confirm_no(_event: Any = None) -> None:
        confirm_row.visible = False
        pending_action["type"] = None

    open_btn.on_click(_request_open)
    new_btn.on_click(_request_new)
    confirm_yes.on_click(_on_confirm_yes)
    confirm_no.on_click(_on_confirm_no)

    def _prompt_save_path() -> str:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.asksaveasfilename(
            title="Save aircraft CSV as",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=state.working_dir or None,
        )
        root.destroy()
        return selected

    def _save_csv_to(path: str) -> None:
        out_df = _csv["df"][~_csv["df"]["ac"].apply(_is_blank)]
        if out_df.empty:
            csv_status.object = (
                "⚠️ Nothing to save - no aircraft are currently sourced from "
                'the CSV. Use "Unlink CSV from config" instead if you no '
                "longer need a separate file."
            )
            return

        try:
            out_df.to_csv(path, index=False, columns=CSV_COLUMNS)
        except Exception as e:  # pylint: disable=broad-exception-caught
            csv_status.object = f"❌ Failed to save: {e}"
            return

        config = state.edited_config
        if config:
            aircraft = config.setdefault("aircraft", {})
            aircraft["dir"] = str(Path(path).parent)
            aircraft["file"] = Path(path).name
            state.dirty = True
            state.param.trigger("edited_config")

        state.aircraft_csv_dirty = False
        csv_status.object = f"✅ Saved to `{path}`."

    def _on_save_click(_event: Any = None) -> None:
        path = _current_csv_path()
        if not path:
            path = _prompt_save_path()
            if not path:
                return
        _save_csv_to(path)

    def _on_save_as_click(_event: Any = None) -> None:
        path = _prompt_save_path()
        if path:
            _save_csv_to(path)

    save_btn.on_click(_on_save_click)
    save_as_btn.on_click(_on_save_as_click)

    def _on_unlink_click(_event: Any = None) -> None:
        """Remove the CSV file reference from the config, leaving any
        inline (source="config") aircraft data untouched.

        Refuses if any row is still sourced from the CSV - unlinking
        would silently orphan that data, since it only lives in the
        locally-held CSV DataFrame and isn't written anywhere once the
        config no longer points at a file.
        """
        config = state.edited_config
        if not config:
            return
        aircraft = config.get("aircraft", {})
        if not (aircraft.get("dir") or aircraft.get("file")):
            csv_status.object = "ℹ️ No CSV file is currently linked."
            return

        still_csv = sorted(
            row["ac"]
            for _, row in table.value.iterrows()
            if row["source"] == "csv" and not _is_blank(row.get("ac"))
        )
        if still_csv:
            csv_status.object = (
                "⚠️ Cannot unlink - still sourced from the CSV: "
                f"{', '.join(still_csv)}. Switch their source to "
                '"config" first.'
            )
            return

        aircraft["dir"] = ""
        aircraft["file"] = ""
        _csv["df"] = _empty_csv_df()
        state.aircraft_csv_dirty = False
        state.dirty = True
        state.param.trigger("edited_config")
        csv_status.object = "✅ CSV file reference removed from the configuration."
        _rebuild_table()

    unlink_btn.on_click(_on_unlink_click)

    # ── config change watchers ───────────────────────────────────────

    def _on_config_generation_changed(_event: Any = None) -> None:
        """A fresh config was loaded/created - reset the local CSV state
        and rebuild the table from scratch, auto-loading the config's
        referenced aircraft CSV file if it already points at one."""
        _csv["df"] = _empty_csv_df()
        state.aircraft_csv_dirty = False
        csv_status.object = ""

        path = _current_csv_path()
        if path and Path(path).exists():
            try:
                df = pd.read_csv(path)
                df.columns = df.columns.str.strip()
                for col in CSV_COLUMNS:
                    if col not in df.columns:
                        df[col] = None
                if "ac" in df.columns:
                    _csv["df"] = df[CSV_COLUMNS].copy()
                    csv_status.object = f"ℹ️ Loaded `{Path(path).name}`."
            except Exception as e:  # pylint: disable=broad-exception-caught
                csv_status.object = f"⚠️ Could not auto-load aircraft CSV: {e}"

        if state.edited_config is None:
            status_pane.object = "⚠️ Create or load a configuration first."
        else:
            status_pane.object = ""

        _rebuild_table()

    state.param.watch(_on_config_generation_changed, "config_generation")

    if state.edited_config is None:
        status_pane.object = "⚠️ Create or load a configuration first."
    else:
        _on_config_generation_changed()

    # ── layout ────────────────────────────────────────────────────────

    card_table = pn.Card(
        status_pane,
        pn.Row(add_row_btn, delete_row_btn, calculate_btn, check_btn),
        table,
        check_status,
        title="Aircraft data",
        collapsible=False,
        sizing_mode="stretch_width",
    )

    return pn.Column(
        pn.pane.Markdown(TITLE),
        pn.Row(open_btn, new_btn),
        pn.Row(save_btn, save_as_btn, unlink_btn),
        confirm_row,
        csv_status,
        card_table,
        sizing_mode="stretch_width",
        styles={"gap": "10px", "margin-top": "15px"},
    )
