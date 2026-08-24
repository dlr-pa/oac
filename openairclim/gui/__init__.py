"""
Configure the OpenAirClim GUI.
"""

def launch(
    config_path=None, results_path=None, show=True, port=5006, theme="default"
):
    """Launch the OpenAirClim GUI in the browser.

    Args:
        config_path (str or Path, optional): Path to an existing config file to
            load on startup. Defaults to None.
        results_path (str or Path, optional): Path to an existing output file
            to view on startup. Defaults to None.
        show (bool, optional): Open a browser automatically. Defaults to True.
        port (int, optional): Port for the Panel server. Defaults to 5006.
        theme (str, optional): Colour theme, either "default" (light) or
            "dark". Set once at launch rather than toggled in the GUI, because
            switching theme requires a full page reload which would reset any
            unsaved edits. Defaults to "default".
    """
    import panel as pn
    from .app import build_app

    pn.extension(sizing_mode="stretch_width")

    def _make_app():
        return build_app(
            config_path=config_path, results_path=results_path, theme=theme
        )

    # Pass a factory to give each session its own independent app and AppState
    pn.serve(_make_app, port=port, show=show, title="OpenAirClim")
