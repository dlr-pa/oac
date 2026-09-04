"""Shared reactive application state.

All tabs read from and write to a single ``AppState`` instance,
which keeps them loosely coupled while sharing data like the
validated configuration dictionary and file paths.
"""

from typing import Any

import param


class AppState(param.Parameterized):
    """Observable application state shared across tabs."""

    working_dir = param.String(default="", doc="Project working directory.")
    config_path = param.String(default="", doc="Path to the config file.")
    config = param.Dict(default=None, doc="Validated configuration dictionary.")
    edited_config = param.Dict(
        default=None,
        doc="Working copy of the configuration, mutated by the editor UI.",
    )
    results_path = param.String(default="", doc="Path to results NetCDF file.")
    dirty = param.Boolean(
        default=False, doc="Whether there are unsaved edits to the config."
    )
    config_generation = param.Integer(
        default=0,
        doc="Bumped each time a new config object is loaded/created, so the "
        "Config tab knows to rebuild its form from scratch (as opposed to "
        "the edited_config trigger, which fires on every field edit too).",
    )
    aircraft_csv_dirty = param.Boolean(
        default=False,
        doc="Whether the aircraft CSV table has unsaved edits. Separate from "
        "`dirty` because CSV row data (for csv-sourced aircraft) lives "
        "outside edited_config until explicitly saved to disk.",
    )
    config_text_dirty = param.Boolean(
        default=False,
        doc="Whether the Config (Expert) tab's text box has edits that "
        "haven't been applied to the working configuration yet. Separate "
        "from `dirty` because text typed there is a private scratchpad "
        "until 'Apply to Config' is clicked.",
    )
    needs_revalidation = param.Boolean(
        default=False,
        doc="Whether edited_config has changed since the last "
        "run_full_validation() call, Set True automatically whenever "
        "edited_config changes; callers of run_full_validation are "
        "responsible for clearing it again once they've reported a "
        "fresh result.",
    )

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.param.watch(self._mark_needs_revalidation, "edited_config")

    def _mark_needs_revalidation(self, _event: Any) -> None:
        self.needs_revalidation = True
