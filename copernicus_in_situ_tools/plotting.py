import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from typing import List, Dict, Tuple, Optional

class StationPlotter:
    """
    Class to plot station-based data, including:
      - Spaghetti plots per station
      - Aggregated mean ± std plots
      - Flexible styling options
    """
    def __init__(self,
                 station_layers: List[xr.Dataset],
                 stats: Dict[str, xr.Dataset],
                 layers: Dict[str, Tuple[float, float]],
                 variable: str,
                 style: Optional[Dict] = None):
        """
        Parameters
        ----------
        station_layers : list of xr.Dataset
            Per-station layer averages (from compute_layer_averages_per_station)
        stats : dict
            Aggregated mean ± std per layer (from compute_layer_statistics_across_stations)
        layers : dict
            Dictionary of depth_layer_name -> (min_depth, max_depth)
        variable : str
            Name of the variable being plotted (e.g., "TEMP")
        style : dict, optional
            Plotting style options:
              - colors : dict of depth_layer_name -> color
              - alpha : float
              - lw_station : float
              - lw_mean : float
              - max_stations : int
              - time_downsample : int
              - figsize : tuple
        """
        self.station_layers = station_layers
        self.stats = stats
        self.layers = layers
        self.variable = variable
        self.style = style or {}
        self._set_defaults()

    def _set_defaults(self):
        """Set default plotting styles if not provided."""
        default_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
        self.colors = self.style.get("colors",
                                     {layer: default_colors[i % len(default_colors)]
                                      for i, layer in enumerate(self.layers.keys())})
        self.alpha = self.style.get("alpha", 0.4)
        self.lw_station = self.style.get("lw_station", 1.0)
        self.lw_mean = self.style.get("lw_mean", 2.5)
        self.max_stations = self.style.get("max_stations", 50)
        self.time_downsample = self.style.get("time_downsample", 1)
        self.figsize = self.style.get("figsize", (12, 10))

    # -------------------- Low-level helpers --------------------
    def _select_stations(self) -> np.ndarray:
        """Return indices of stations to plot based on max_stations."""
        num_stations = len(self.station_layers)
        if num_stations <= self.max_stations:
            return np.arange(num_stations)
        return np.random.choice(num_stations, self.max_stations, replace=False)

    def _get_da_for_station(self, station_idx: int, depth_layer_name: str) -> xr.DataArray:
        """Return the DataArray for a single station and layer, optionally downsampled."""
        da = self.station_layers[station_idx][depth_layer_name]
        return da[::self.time_downsample]

    def _plot_single_da(self, ax, da: xr.DataArray, color: str, lw: float, alpha: float):
        """Plot a single DataArray on the given axis."""
        ax.scatter(da["TIME"], da, color=color, lw=lw, alpha=alpha)

    def _fill_mean_std(self, ax, mean_da: xr.DataArray, std_da: xr.DataArray, color: str, alpha: float):
        """Fill the mean ± std band on the axis."""
        ax.fill_between(mean_da["TIME"],
                        mean_da - std_da,
                        mean_da + std_da,
                        color=color,
                        alpha=alpha)

    # -------------------- Public plotting methods --------------------
    def plot_spaghetti_for_layer(self, depth_layer_name: str, ax=None):
        """Plot spaghetti plot of all stations for a single depth layer."""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        stations_to_plot = self._select_stations()
        color = self.colors.get(depth_layer_name, "tab:blue")

        for i in stations_to_plot:
            da = self._get_da_for_station(i, depth_layer_name)
            self._plot_single_da(ax, da, color=color, lw=self.lw_station, alpha=self.alpha)

        ax.set_title(f"{self.variable} by Station: {depth_layer_name}")
        ax.set_ylabel(self.variable)
        return ax

    def plot_mean_std_for_layer(self, depth_layer_name: str, ax=None):
        """Plot mean ± std of a depth layer across stations."""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        mean_da = self.stats[depth_layer_name]["mean"]
        std_da = self.stats[depth_layer_name]["std"]
        color = self.colors.get(depth_layer_name, "tab:blue")

        self._plot_single_da(ax, mean_da, color=color, lw=self.lw_mean, alpha=1.0)
        self._fill_mean_std(ax, mean_da, std_da, color=color, alpha=self.alpha)

        ax.set_title(f"{self.variable} mean ± std: {depth_layer_name}")
        ax.set_ylabel(self.variable)
        return ax

    def plot_all_layers(self):
        """Plot spaghetti and mean ± std for all layers in a single figure."""
        n_layers = len(self.layers)
        fig, axs = plt.subplots(n_layers, 2,
                                figsize=(2*self.figsize[0], n_layers*self.figsize[1]//2),
                                sharex=True)
        if n_layers == 1:
            axs = np.array([axs])  # ensure 2D array

        for i, depth_layer_name in enumerate(self.layers.keys()):
            self.plot_spaghetti_for_layer(depth_layer_name, ax=axs[i, 0])
            self.plot_mean_std_for_layer(depth_layer_name, ax=axs[i, 1])

        for ax in axs.flatten():
            ax.set_xlabel("TIME")
        plt.tight_layout()
        return fig, axs

def plot_station_map(station_data, variable="TEMP"):
    """
    Plot a world map with one point per station.
    Marker size is proportional to number of measurements.

    Args:
        station_data (list of xr.Dataset]): List of station datasets.
        variable (str): Variable to count measurements for.

    Returns:
        matplotlib.figure.Figure
    """
    lats, lons, sizes = [], [], []
    for ds in station_data:
        if "LATITUDE" not in ds or "LONGITUDE" not in ds:
            continue
        lat = float(ds["LATITUDE"].values)
        lon = float(ds["LONGITUDE"].values)
        n_obs = ds[variable].count().item() if variable in ds else 1

        lats.append(lat)
        lons.append(lon)
        sizes.append(n_obs)

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(lons, lats, s=np.array(sizes) / 50, c="tab:blue", alpha=0.6, edgecolors="k")
    ax.set_title("Station distribution")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return fig

