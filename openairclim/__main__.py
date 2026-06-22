"""
Allows launching OpenAirClim via ``python -m openairclim file.toml``.
"""

import argparse
from .core.main import run


def main():
    """Parse command-line arguments and run OpenAirClim."""
    parser = argparse.ArgumentParser(
        prog="python -m openairclim",
        description="Run OpenAirClim.",
    )
    parser.add_argument(
        "config", type=str, help="Path to OpenAirClim config file."
    )
    args = parser.parse_args()

    run(file_name=args.config,)


if __name__ == "__main__":
    main()
