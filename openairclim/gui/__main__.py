"""Allows launching the GUI via ``python -m openairclim.gui``."""

import argparse

from . import launch


def main():
    """Parse command-line arguments and launch the GUI."""
    parser = argparse.ArgumentParser(
        prog="python -m openairclim.gui",
        description="Launch the OpenAirClim graphical user interface.",
    )
    parser.add_argument(
        "--config", type=str, help="Path to config file."
    )
    parser.add_argument(
        "--port", type=int, default=5006, help="Port for the Panel server."
    )
    args = parser.parse_args()

    launch(config_path=args.config, port=args.port,)


if __name__ == "__main__":
    main()
