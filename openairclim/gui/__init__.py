"""
Configure the OpenAirClim GUI.
"""

def launch(config_path=None, results_path=None, show=True, port=5006):
    """Launch the OpenAirClim GUI in the browser.

    Args:
        config_path (str or Path, optional): Path to an existing config file to
            load on startup. Defaults to None.
        results_path (str or Path, optional): Path to an existing output file
            to view on startup. Defaults to None.
        show (bool, optional): Open a browser automatically. Defaults to True.
        port (int, optional): Port for the Panel server. Defaults to 5006.
    """
    import panel as pn
    from .app import build_app

    pn.extension(sizing_mode="stretch_width")
    app = build_app(config_path=config_path, results_path=results_path)
    pn.serve(app, port=port, show=show, title="OpenAirClim")
