#!/usr/bin/env python
"""
Main script to process Copernicus in-situ data and plot results.
"""

import yaml
from copernicus_in_situ_tools.io import load_station_directory
from copernicus_in_situ_tools.processing import (
    compute_layer_averages_per_station,
    compute_layer_statistics_across_stations
)
from copernicus_in_situ_tools.plotting import StationPlotter
from copernicus_in_situ_tools.ui import print_banner, parse_arguments
import matplotlib.pyplot as plt

def main():
    # -------------------- Banner + CLI --------------------
    print_banner()
    args = parse_arguments()

    # Load layers from YAML (dictionary format)
    with open("config/layers.yaml") as f:
        layers_config = yaml.safe_load(f)
    LAYERS = layers_config["layers"]  # e.g., {"surface": [0,50], "thermocline": [50,200], "deep": [200,1000]}

    with open("config/style.yaml") as f:
        style_config = yaml.safe_load(f)
    STYLE = style_config["style"]

    # -------------------- Load Station Data --------------------
    station_data = load_station_directory(args.data_dir, variable=args.variable)

    # -------------------- Compute Per-Station Layer Averages --------------------
    station_layers = compute_layer_averages_per_station(
        station_data, variable=args.variable, layers=LAYERS
    )

    # -------------------- Compute Aggregated Statistics --------------------
    stats = compute_layer_statistics_across_stations(
        station_data, variable=args.variable, layers=LAYERS
    )

    # -------------------- Plot --------------------
    plotter = StationPlotter(
        station_layers=station_layers,
        stats=stats,
        layers=LAYERS,
        variable=args.variable,
        style=STYLE
    )

    # 1️⃣ Spaghetti plot for a single layer
    plotter.plot_spaghetti_for_layer("surface")

    # 2️⃣ Mean ± std plot for a single layer
    plotter.plot_mean_std_for_layer("surface")

    # 3️⃣ Combined figure for all layers
    fig, axs = plotter.plot_all_layers()

    plt.show()

if __name__ == "__main__":
    main()
