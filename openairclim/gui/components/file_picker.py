"""File/folder picker component: text input + native browse dialog."""

from pathlib import Path

import panel as pn
import param


class FilePicker(pn.viewable.Viewer):
    """A picker with a text field, browse button, and status indicator.

    Parameters
    ----------
    label : str
        Display label shown above the text input.
    file_types : list of (description, pattern) tuples
        File type filters for the browse dialog (ignored in directory mode).
    directory : bool
        If True, browse for a folder instead of a file.
    path : str
        The currently selected path (readable and watchable).
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

    def __init__(self, **params):
        super().__init__(**params)

        self._text_input = pn.widgets.TextInput(
            name=self.label,
            value=self.path,
            placeholder=(
                "Enter a folder path or click the folder icon\u2026"
                if self.directory
                else "Enter a file path or click the folder icon\u2026"
            ),
        )
        self._browse_btn = pn.widgets.Button(
            icon="folder",
            icon_size="1.1em",
            button_type="default",
            width=44,
            margin=(18, 0, 0, 6),
        )
        self._status = pn.pane.Markdown(
            "",
            styles={"margin-top": "0px", "font-size": "0.9em"},
        )

        self._browse_btn.on_click(self._on_browse)
        self._text_input.param.watch(self._on_text_changed, "value")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_browse(self, event):
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
        """Show a short status hint below the input."""
        if not value:
            self._status.object = ""
        elif self.directory and Path(value).is_dir():
            self._status.object = f"\u2705 {value}"
        elif not self.directory and Path(value).is_file():
            self._status.object = f"\u2705 {value}"
        else:
            target = "Directory" if self.directory else "File"
            self._status.object = f"\u26a0\ufe0f {target} not found: {value}"

    # ------------------------------------------------------------------
    # Panel rendering
    # ------------------------------------------------------------------

    def __panel__(self):
        return pn.Column(
            pn.Row(self._text_input, self._browse_btn),
            self._status,
            sizing_mode="stretch_width",
        )
