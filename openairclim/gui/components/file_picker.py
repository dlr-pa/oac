"""File/folder picker component: text input + native browse dialog."""

from pathlib import Path

import panel as pn
import param


def _shorten_path(value, keep=2):
    """Return the last `keep` path segments, prefixed with "..." if longer.

    Args:
        value (str): Path string.
        keep (int): Number of trailing path segments to keep.

    Returns:
        str: Shortened path, e.g. ".../oac/example" — or the original
            string unchanged if it already has `keep` segments or fewer.
    """
    parts = Path(value).parts
    if len(parts) <= keep:
        return value
    return ".../" + "/".join(parts[-keep:])


class FilePicker(pn.viewable.Viewer):
    """A picker with a text field, browse button, and status indicator.

    Args:
        label (str): Display label shown above the text input.
        file_types (list): File type filters for the browse dialog.
        directory (bool): If True, browse for a folder instead of a file.
        path (str): Currently selected path (readable and watchable).
    """

    label = param.String(default="File")
    file_types = param.List(
        default=[("All files", "*.*")],
        doc="File type filters as (description, pattern) tuples.",
    )
    directory = param.Boolean(
        default=False,
        doc="If True, select a directory instead of a file.",
    )
    path = param.String(default="", doc="Currently selected path.")
    description = param.String(
        default=None,
        allow_None=True,
        doc="Tooltip text (Markdown supported) shown as an (i) icon next "
        "to the label — same convention as Panel widgets' own "
        "`description`, passed straight through to the underlying "
        "TextInput.",
    )

    def __init__(self, **params):
        super().__init__(**params)

        self._text_input = pn.widgets.TextInput(
            name=self.label,
            value=self.path,
            placeholder=(
                "Enter a folder path or click the folder icon..."
                if self.directory
                else "Enter a file path or click the folder icon..."
            ),
            description=self.description,
        )
        self._browse_btn = pn.widgets.Button(
            icon="folder",
            icon_size="1.1em",
            button_type="default",
            width=44,
            margin=(24, 10, 0, 6),
        )
        self._status = pn.pane.Markdown(
            "", styles={"font-size": "0.9em"}, margin=(0, 5, 0, 5),
        )

        self._browse_btn.on_click(self._on_browse)
        self._text_input.param.watch(self._on_text_changed, "value")

    def set_path(self, value: str) -> None:
        """Programmatically set the selected path. Updates the text input,
        which in turn syncs `self.path` and the status indicator via the
        existing `_on_text_changed` watcher."""
        self._text_input.value = value

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_browse(self, _event):
        """Open a native file/folder dialog and update the path."""
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        if self.directory:
            selected = filedialog.askdirectory(
                title=f"Select {self.label}",
            )
        else:
            selected = filedialog.askopenfilename(
                title=f"Select {self.label}",
                filetypes=self.file_types,
            )
        root.destroy()

        if selected:
            self._text_input.value = selected

    def _on_text_changed(self, event):
        """Sync the text input value to the public *path* param."""
        self.path = event.new
        self._update_status(event.new)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_status(self, value):
        """Show a short status hint below the input.

        The path itself is shortened to its last two segments (e.g.
        ".../oac/example") so the status stays on one line even for
        long, deeply-nested paths.
        """
        if not value:
            self._status.object = ""
            return

        target = "Folder" if self.directory else "File"
        short = _shorten_path(value)
        exists = Path(value).is_dir() if self.directory else Path(value).is_file()
        if exists:
            self._status.object = f'✅ {target} "{short}" exists.'
        else:
            self._status.object = f'⚠️ {target} not found: "{short}"'

    # ------------------------------------------------------------------
    # Panel rendering
    # ------------------------------------------------------------------

    def __panel__(self):
        return pn.Column(
            pn.Row(self._text_input, self._browse_btn),
            self._status,
            sizing_mode="stretch_width",
        )
