import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from typing import List

def plot_spaghetti(
    station_data: List[xr.Dataset],
    variable: str,
    max_stations: int = 50,
    time_downsample: int = 1
):
    """
    Plot spaghetti plot for each layer of the variable.
    """
    if not station_data:
        print("No station data to plot.")
        return

    # Determine layers from first station
    layer_names = [name for name in station_data[0].data_vars if name.endswith(f"_{variable}")]
    num_layers = len(layer_names)

    fig, axs = plt.subplots(num_layers, 1, figsize=(12, 4*num_layers), sharex=True)

    if num_layers == 1:
        axs = [axs]

    num_stations = len(station_data)
    if num_stations > max_stations:
        stations_to_plot = np.random.choice(num_stations, max_stations, replace=False)
    else:
        stations_to_plot = np.arange(num_stations)

    for i, layer_var in enumerate(layer_names):
        for s in stations_to_plot:
            ds = station_data[s]
            if layer_var not in ds:
                continue
            axs[i].plot(ds["TIME"][::time_downsample], ds[layer_var][::time_downsample],
                        color="tab:blue", alpha=0.4, lw=1)
        axs[i].set_ylabel(layer_var)
        axs[i].set_title(f"Spaghetti plot for {layer_var}")

    axs[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.show()


def plot_global_average(
    global_ds: xr.Dataset,
    variable: str
):
    """
    Plot global mean ± std for all layers of the variable.
    """
    if not global_ds:
        print("No global dataset to plot.")
        return

    layer_names = [name.replace("_mean", "") for name in global_ds.data_vars if name.endswith("_mean")]
    num_layers = len(layer_names)

    fig, axs = plt.subplots(num_layers, 1, figsize=(12, 4*num_layers), sharex=True)

    if num_layers == 1:
        axs = [axs]

    for i, layer_var in enumerate(layer_names):
        mean_var = f"{layer_var}_mean"
        std_var = f"{layer_var}_std"
        if mean_var not in global_ds or std_var not in global_ds:
            continue
        axs[i].plot(global_ds["TIME"], global_ds[mean_var], color="tab:blue", lw=2, label="Mean")
        axs[i].fill_between(global_ds["TIME"].values,
                            global_ds[mean_var] - global_ds[std_var],
                            global_ds[mean_var] + global_ds[std_var],
                            color="tab:blue", alpha=0.3, label="Std dev")
        axs[i].set_ylabel(layer_var)
        axs[i].legend()
        axs[i].set_title(f"Global average ± std for {layer_var}")

    axs[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.show()
