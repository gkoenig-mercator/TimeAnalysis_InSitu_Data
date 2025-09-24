#!/usr/bin/env python
"""
Main script to process Copernicus in-situ data and plot results.
"""

import yaml
from copernicus_in_situ_tools.io import load_station_directory
from copernicus_in_situ_tools.processing import compute_layer_averages_per_station, compute_layer_statistics_across_stations
from copernicus_in_situ_tools.plotting import plot_spaghetti, plot_global_average

# -------------------- Configuration --------------------

DATA_DIR = "/homelocal/gkoenig/datatests/INSITU_IBI_PHYBGCWAV_DISCRETE_MYNRT_013_033/cmems_obs-ins_ibi_phybgcwav_mynrt_na_irr_202311/history/XX/"  # path to your NetCDF files
VARIABLE = "TEMP"   # variable to analyze, e.g., TEMP, SAL
MAX_STATIONS_TO_PLOT = 50
TIME_DOWNSAMPLE = 1

# Load layer definitions from YAML
with open("config/layers.yaml") as f:
    layers_config = yaml.safe_load(f)
LAYERS = layers_config["layers"]

# -------------------- Step 1: Load Data --------------------

station_data = load_station_directory(DATA_DIR, variable=VARIABLE)

if not station_data:
    print("No station data loaded. Exiting.")
    exit()

# -------------------- Step 2: Compute Layer Averages --------------------

# Per-station layer averages for spaghetti plots
station_layers = compute_layer_averages_per_station(station_data, variable=VARIABLE, layers=LAYERS)

# -------------------- Step 3: Compute Global Averages --------------------

stats = compute_layer_statistics_across_stations(station_data, variable=VARIABLE, layers=LAYERS)

# -------------------- Step 4: Plot --------------------

# Spaghetti plot per station
plot_spaghetti(station_data, variable=VARIABLE, max_stations=MAX_STATIONS_TO_PLOT, time_downsample=TIME_DOWNSAMPLE)

# Global average ± std
plot_global_average(global_ds, variable=VARIABLE)
