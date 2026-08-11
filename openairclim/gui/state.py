"""Shared reactive application state.

All tabs read from and write to a single ``AppState`` instance,
which keeps them loosely coupled while sharing data like the
validated configuration dictionary and file paths.
"""

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
