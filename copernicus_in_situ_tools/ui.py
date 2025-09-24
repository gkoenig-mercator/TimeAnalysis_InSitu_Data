import argparse
import pyfiglet
from colorama import Fore, Style


def print_banner():
    """
    Prints a colored ASCII art banner for the tool.
    """
    banner = pyfiglet.figlet_format("Copernicus Tool")
    print(Fore.CYAN + banner + Style.RESET_ALL)


def parse_arguments():
    """
    Handles command-line arguments for the tool.

    Returns:
        argparse.Namespace: parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Analyze Copernicus In Situ data and plot variable trends."
    )
    parser.add_argument(
        "variable",
        type=str,
        help="Variable to analyze (e.g., TEMP, PSAL).",
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Directory containing the NetCDF files.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="config/layers.yaml",
        help="YAML file with depth layer definitions (default: config/layers.yaml).",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="config/style.yaml",
        help="YAML file with plotting styles (default: config/style.yaml).",
    )
    return parser.parse_args()
