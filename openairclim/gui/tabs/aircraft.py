"""Aircraft tab: define per-aircraft contrail parameters.

Aircraft data can live in two places, matching how core reads it (see
core/read_config.py):

* Inline in the config, under ``config["aircraft"][<ac_id>]`` — a dict
  with ``G_250``/``b``/``PMrel`` — the "config" source.
* In a separate CSV file (``config["aircraft"]["dir"]``/``["file"]``,
  read by ``load_ac_data``) — the "csv" source. Columns: ac, b, PMrel,
  G_250 (core's CSV loader also supports deriving G_250/PMrel from
  other columns, but this tab always writes them directly).

Core tolerates an aircraft identifier defined in both places at once
(config wins per-key, with a warning) — this tab instead keeps the two
mutually exclusive per identifier: one row per aircraft in a single
table, with a "source" column. Changing a row's source moves its data
between ``edited_config`` and a locally-held CSV DataFrame, which is
only written to disk on Save (mirroring the sidebar's config Save).
``edited_config["aircraft"]["types"]`` is kept in sync automatically,
as the union of every row's aircraft identifier.
"""

from pathlib import Path

import pandas as pd
import panel as pn

from ..components.utils import load_inventory

CSV_COLUMNS = ["ac", "b", "PMrel", "G_250"]
TABLE_COLUMNS = ["ac", "b", "PMrel", "G_250", "source"]
SOURCE_OPTIONS = ["config", "csv"]


def _empty_table_df():
    """Return an empty, correctly-typed table DataFrame."""
    return pd.DataFrame({
        "ac": pd.Series(dtype="object"),
        "b": pd.Series(dtype="float64"),
        "PMrel": pd.Series(dtype="float64"),
        "G_250": pd.Series(dtype="float64"),
        "source": pd.Series(dtype="object"),
    })


def _empty_csv_df():
    """Return an empty, correctly-typed CSV DataFrame."""
    return pd.DataFrame({
        "ac": pd.Series(dtype="object"),
        "b": pd.Series(dtype="float64"),
        "PMrel": pd.Series(dtype="float64"),
        "G_250": pd.Series(dtype="float64"),
    })


def _is_blank(value):
    """Return True if a table/CSV cell counts as "not filled in"."""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _build_table_df(edited, csv_df):
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
        k: v for k, v in aircraft.items()
        if k not in ("types", "dir", "file") and isinstance(v, dict)
    }
    csv_lookup = {}
    if csv_df is not None and not csv_df.empty:
        for _, row in csv_df.iterrows():
            if not _is_blank(row.get("ac")):
                csv_lookup[str(row["ac"])] = row

    seen = set()
    rows = []

    def _row_from_config(ac):
        d = config_rows.get(ac, {})
        return {"ac": ac, "b": d.get("b"), "PMrel": d.get("PMrel"), "G_250": d.get("G_250"), "source": "config"}

    def _row_from_csv(ac):
        r = csv_lookup[ac]
        return {"ac": ac, "b": r.get("b"), "PMrel": r.get("PMrel"), "G_250": r.get("G_250"), "source": "csv"}

    for ac in types:
        if ac in seen:
            continue
        seen.add(ac)
        if ac in config_rows:
            rows.append(_row_from_config(ac))
        elif ac in csv_lookup:
            rows.append(_row_from_csv(ac))
        else:
            # A bare identifier with no data anywhere yet — valid as
            # long as contrails aren't being computed.
            rows.append({"ac": ac, "b": None, "PMrel": None, "G_250": None, "source": "config"})

    # Data present in a source but not (yet) listed in "types" — shouldn't
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


def panel(state):
    """Return the aircraft tab content.

    Args:
        state (AppState): Shared application state.
    """
    _csv = {"df": _empty_csv_df()}
    _inv_ac_cache = {"key": None, "values": set()}
    _rebuilding = {"flag": False}
    pending_action = {"type": None}

    status_pane = pn.pane.Markdown("")
    csv_status = pn.pane.Markdown("")
    check_status = pn.pane.Markdown("")

    open_btn = pn.widgets.Button(name="Open CSV…", button_type="primary")
    new_btn = pn.widgets.Button(name="New CSV", button_type="default")
    save_btn = pn.widgets.Button(name="Save CSV", button_type="success")
    save_as_btn = pn.widgets.Button(name="Save CSV As…", button_type="default")
    add_row_btn = pn.widgets.Button(name="Add row", button_type="default")
    delete_row_btn = pn.widgets.Button(name="Delete selected row(s)", button_type="danger")

    confirm_msg = pn.pane.Markdown(
        "⚠️ You have unsaved edits to the aircraft CSV. Continuing will discard them."
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
            "b": "number",
            "PMrel": "number",
            "G_250": "number",
            "source": {"type": "list", "values": SOURCE_OPTIONS},
        },
        titles={
            "ac": "Aircraft ID", "b": "b [m]", "PMrel": "PMrel",
            "G_250": "G_250", "source": "Source",
        },
        sizing_mode="stretch_width",
    )

    # ── helpers ───────────────────────────────────────────────────────

    def _current_csv_path():
        config = state.edited_config
        if not config:
            return None
        aircraft = config.get("aircraft", {})
        f = aircraft.get("file")
        if not f:
            return None
        d = aircraft.get("dir", "")
        return str(Path(state.working_dir) / d / f)

    def _load_inventory_ac_values():
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

        ac_values = set()
        any_no_ac = False
        if inv_files and state.working_dir:
            for f in inv_files:
                try:
                    ds = load_inventory(state.working_dir, inv_dir, f)
                except Exception:
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

    def _run_check():
        """Update check_status: are all inventory aircraft defined, and —
        only if contrails are being computed — do they have complete
        G_250/b/PMrel data (in either source)?"""
        config = state.edited_config
        if not config:
            check_status.object = ""
            return

        inv_ac = _load_inventory_ac_values()
        if not inv_ac:
            check_status.object = ""
            return

        types = set(config.get("aircraft", {}).get("types", []))
        missing_types = inv_ac - types
        if missing_types:
            check_status.object = (
                "⚠️ Aircraft identifier(s) found in emission inventories "
                f"but not defined here: {', '.join(sorted(missing_types))}"
            )
            return

        if "cont" not in config.get("species", {}).get("out", []):
            check_status.object = "✅ All aircraft identifiers used in the inventories are defined."
            return

        df = table.value
        incomplete = []
        for ac in sorted(inv_ac):
            match = df[df["ac"] == ac]
            if match.empty:
                incomplete.append(ac)
                continue
            row = match.iloc[0]
            if _is_blank(row.get("G_250")) or _is_blank(row.get("b")) or _is_blank(row.get("PMrel")):
                incomplete.append(ac)
        if incomplete:
            check_status.object = (
                "⚠️ Contrails are enabled, but G_250/b/PMrel are incomplete for: "
                f"{', '.join(incomplete)}"
            )
        else:
            check_status.object = "✅ All aircraft identifiers have complete contrail data."

    def _sync_types():
        config = state.edited_config
        if not config:
            return
        acs = [a for a in table.value["ac"].tolist() if not _is_blank(a)]
        config.setdefault("aircraft", {})["types"] = sorted(dict.fromkeys(acs))

    def _sync_row(ac, b, pmrel, g250, source, old_ac=None):
        """Write one row's data into whichever store matches `source`,
        removing it from the other store (and from `old_ac` if renamed)."""
        config = state.edited_config
        if not config:
            return
        aircraft = config.setdefault("aircraft", {})

        for name in {ac, old_ac} - {None}:
            aircraft.pop(name, None)
            _csv["df"] = _csv["df"][_csv["df"]["ac"] != name]

        if source == "config":
            data = {}
            if not _is_blank(g250):
                data["G_250"] = float(g250)
            if not _is_blank(b):
                data["b"] = float(b)
            if not _is_blank(pmrel):
                data["PMrel"] = float(pmrel)
            aircraft[ac] = data
            state.dirty = True
        else:
            new_row = pd.DataFrame([{
                "ac": ac,
                "b": None if _is_blank(b) else float(b),
                "PMrel": None if _is_blank(pmrel) else float(pmrel),
                "G_250": None if _is_blank(g250) else float(g250),
            }]).astype(_csv["df"].dtypes.to_dict())
            _csv["df"] = pd.concat([_csv["df"], new_row], ignore_index=True)
            state.aircraft_csv_dirty = True

    def _rebuild_table():
        config = state.edited_config
        _rebuilding["flag"] = True
        try:
            table.value = _build_table_df(config, _csv["df"]) if config else _empty_table_df()
        finally:
            _rebuilding["flag"] = False
        _run_check()

    # ── table edits ──────────────────────────────────────────────────

    def _on_table_edit(event):
        if _rebuilding["flag"]:
            return
        row = table.value.loc[event.row]
        ac = row["ac"]
        if _is_blank(ac):
            return  # nothing to sync yet — aircraft not named

        old_ac = event.old if event.column == "ac" and not _is_blank(event.old) else None
        _sync_row(ac, row["b"], row["PMrel"], row["G_250"], row["source"], old_ac=old_ac)
        _sync_types()
        state.param.trigger("edited_config")
        _run_check()

    table.on_edit(_on_table_edit)

    def _on_add_row(event=None):
        new_row = pd.DataFrame(
            [{"ac": "", "b": None, "PMrel": None, "G_250": None, "source": "config"}]
        ).astype(table.value.dtypes.to_dict())
        table.value = pd.concat([table.value, new_row], ignore_index=True)

    def _on_delete_rows(event=None):
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
        _run_check()

    add_row_btn.on_click(_on_add_row)
    delete_row_btn.on_click(_on_delete_rows)

    # ── CSV open / new / save ───────────────────────────────────────

    def _do_open_csv(path):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            csv_status.object = f"❌ Failed to read CSV: {e}"
            return
        df.columns = df.columns.str.strip()
        if "ac" not in df.columns:
            csv_status.object = "❌ CSV must have an 'ac' column."
            return
        for col in ("b", "PMrel", "G_250"):
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

    def _on_open_click(event=None):
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

    def _on_new_click(event=None):
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

    def _request_open(event=None):
        if state.aircraft_csv_dirty:
            pending_action["type"] = "open"
            confirm_row.visible = True
        else:
            _on_open_click()

    def _request_new(event=None):
        if state.aircraft_csv_dirty:
            pending_action["type"] = "new"
            confirm_row.visible = True
        else:
            _on_new_click()

    def _on_confirm_yes(event=None):
        confirm_row.visible = False
        action = pending_action["type"]
        pending_action["type"] = None
        if action == "open":
            _on_open_click()
        elif action == "new":
            _on_new_click()

    def _on_confirm_no(event=None):
        confirm_row.visible = False
        pending_action["type"] = None

    open_btn.on_click(_request_open)
    new_btn.on_click(_request_new)
    confirm_yes.on_click(_on_confirm_yes)
    confirm_no.on_click(_on_confirm_no)

    def _prompt_save_path():
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

    def _save_csv_to(path):
        try:
            out_df = _csv["df"][~_csv["df"]["ac"].apply(_is_blank)]
            out_df.to_csv(path, index=False, columns=CSV_COLUMNS)
        except Exception as e:
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

    def _on_save_click(event=None):
        path = _current_csv_path()
        if not path:
            path = _prompt_save_path()
            if not path:
                return
        _save_csv_to(path)

    def _on_save_as_click(event=None):
        path = _prompt_save_path()
        if path:
            _save_csv_to(path)

    save_btn.on_click(_on_save_click)
    save_as_btn.on_click(_on_save_as_click)

    # ── config change watchers ───────────────────────────────────────

    def _on_config_generation_changed(event=None):
        """A fresh config was loaded/created — reset the local CSV state
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
                for col in ("b", "PMrel", "G_250"):
                    if col not in df.columns:
                        df[col] = None
                if "ac" in df.columns:
                    _csv["df"] = df[CSV_COLUMNS].copy()
                    csv_status.object = f"ℹ️ Loaded `{Path(path).name}`."
            except Exception as e:
                csv_status.object = f"⚠️ Could not auto-load aircraft CSV: {e}"

        if state.edited_config is None:
            status_pane.object = "⚠️ Create or load a configuration first."
        else:
            status_pane.object = ""

        _rebuild_table()

    def _on_edited_config_changed(event=None):
        # A lighter-weight refresh for edits elsewhere (e.g. adding/removing
        # "cont" from species.out changes whether the completeness check
        # below applies) — doesn't rebuild the table or touch CSV state.
        _run_check()

    state.param.watch(_on_config_generation_changed, "config_generation")
    state.param.watch(_on_edited_config_changed, "edited_config")

    if state.edited_config is None:
        status_pane.object = "⚠️ Create or load a configuration first."
    else:
        _on_config_generation_changed()

    # ── layout ────────────────────────────────────────────────────────

    card_file = pn.Card(
        pn.Row(open_btn, new_btn),
        confirm_row,
        pn.Row(save_btn, save_as_btn),
        csv_status,
        title="Aircraft CSV file",
        collapsible=False,
        sizing_mode="stretch_width",
    )
    card_table = pn.Card(
        status_pane,
        pn.Row(add_row_btn, delete_row_btn),
        table,
        check_status,
        title="Aircraft data",
        collapsible=False,
        sizing_mode="stretch_width",
    )

    return pn.Column(
        card_file,
        card_table,
        sizing_mode="stretch_width",
        styles={"gap": "10px", "margin-top": "15px"},
    )
