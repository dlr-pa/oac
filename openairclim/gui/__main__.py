"""Allows launching the GUI via ``oac-gui`` (or, equivalently,
``python -m openairclim.gui``).
"""

import argparse

from . import launch


def main() -> None:
    """Parse command-line arguments and launch the GUI."""
    parser = argparse.ArgumentParser(
        prog="oac-gui",
        description="Launch the OpenAirClim graphical user interface.",
    )
    parser.add_argument("--config", type=str, help="Path to config file.")
    parser.add_argument("--results", type=str, help="Path to results NetCDF file.")
    parser.add_argument(
        "--port", type=int, default=5006, help="Port for the Panel server."
    )
    parser.add_argument(
        "--theme",
        type=str,
        choices=["default", "dark"],
        default="default",
        help="Colour theme for the GUI. Choice of 'default' (light) or 'dark'.",
    )
    args = parser.parse_args()

    launch(
        config_path=args.config,
        results_path=args.results,
        port=args.port,
        theme=args.theme,
    )


if __name__ == "__main__":
    main()
