"""Helpers for driving GUI widgets off core.config_model's pydantic schema.
"""

import types
from typing import Literal, Union, get_args, get_origin, cast
from pydantic import BaseModel

from ...core.config_model import Config


def _unwrap_optional(annotation):
    """Strip an `Optional[...]`/`X | None` wrapper off an annotation, if
    present, returning the inner type. Leaves other annotations unchanged.
    """
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def submodel(path: str) -> type[BaseModel]:
    """Resolve a dotted config path to its pydantic submodel class. Walks
    through `Config.model_fields` (e.g. "responses.CO2.rf") rather than
    importing the per-section classes directly, so that the data can be
    referenced in the actual toml shape.

    Args:
        path (str): Dotted field path, starting from Config.

    Returns:
        The pydantic model class at that path.
    """
    model: type[BaseModel] = Config
    for part in path.split("."):
        field_info = model.model_fields[part]  # pylint: disable=unsubscriptable-object
        model = cast(type[BaseModel], field_info.annotation)
    return model


def literal_choices(model: type[BaseModel], field: str) -> list:
    """Return the allowed values of a `Literal[...]` field — optionally
    wrapped in `list[...]` (e.g. `species.out`) and/or made optional
    (e.g. `Literal[...] | None`).

    Args:
        model: A pydantic model class, e.g. `submodel("species")`.
        field (str): Name of one of its fields.

    Returns:
        list: Allowed values for that field. A list, not a tuple —
            Panel's SelectBase.options only accepts dict/list.
    """
    annotation = _unwrap_optional(model.model_fields[field].annotation)
    if get_origin(annotation) is list:
        inner = _unwrap_optional(get_args(annotation)[0])
        return list(get_args(inner))
    return list(get_args(annotation))


def field_description(model: type[BaseModel], field: str) -> str | None:
    """Return a pydantic model field's `Field(description=...)`. This can
    be passed straight through to a panel widget's own `description`
    kwarg to provide a tooltip.

    Args:
        model: A pydantic model class, e.g. `submodel("temperature")`.
        field (str): Name of one of its fields.

    Returns:
        str or None: The field's description, if one is set.
    """
    return model.model_fields[field].description


def is_string_like_field(model: type[BaseModel], field: str) -> bool:
    """Return True if `field`'s values should be treated as strings/objects
    (e.g. for a pandas dtype or a Tabulator "input"/"list" editor), rather
    than numeric — i.e. its annotation is (optionally) `str` or a
    `Literal[...]` of strings.

    Args:
        model: A pydantic model class.
        field (str): Name of one of its fields.

    Returns:
        bool: True if string-like, False if numeric.
    """
    annotation = _unwrap_optional(model.model_fields[field].annotation)
    return annotation is str or get_origin(annotation) is Literal
