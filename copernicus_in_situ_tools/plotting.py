import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from typing import List, Dict, Tuple, Optional
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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

class ProfilePlotter:
    """
    Class to plot time-averaged vertical profiles per station
    with spaghetti lines and mean ± std, reusing the StationPlotter style architecture.
    """

    def __init__(self, profiles: xr.DataArray, variable="TEMP", style=None):
        """
        Args:
            profiles: xr.DataArray (station, DEPTH)
            variable: str, variable name for labeling
            style: dict with optional keys:
                color_spaghetti, alpha, lw_station, color_mean, lw_mean, max_stations
        """
        self.profiles = profiles
        self.variable = variable

        # Default style
        default_style = dict(
            color_spaghetti="tab:blue",
            alpha=0.3,
            lw_station=1.0,
            color_mean="tab:red",
            lw_mean=2.5,
            max_stations=50,
        )
        self.style = default_style if style is None else {**default_style, **style}

    # ------------------ Helper methods ------------------

    def _select_stations(self):
        """Randomly select a subset of stations if too many"""
        n_stations = self.profiles.sizes["station"]
        max_s = self.style.get("max_stations", 50)
        if n_stations > max_s:
            return np.random.choice(n_stations, max_s, replace=False)
        return np.arange(n_stations)

    def _compute_statistics(self):
        """Compute mean and std profiles"""
        mean_profile = self.profiles.mean(dim="station", skipna=True)
        std_profile = self.profiles.std(dim="station", skipna=True)
        return mean_profile, std_profile

    def _plot_spaghetti(self, ax, stations_idx):
        """Plot individual station profiles"""
        for i in stations_idx:
            ax.plot(
                self.profiles.isel(station=i),
                self.profiles["DEPTH"],
                color=self.style["color_spaghetti"],
                alpha=self.style["alpha"],
                lw=self.style["lw_station"],
            )

    def _plot_mean_std(self, ax, mean_profile, std_profile):
        """Plot mean ± std shaded area"""
        ax.plot(
            mean_profile,
            self.profiles["DEPTH"],
            color=self.style["color_mean"],
            lw=self.style["lw_mean"],
            label="Mean profile",
        )
        ax.fill_betweenx(
            self.profiles["DEPTH"],
            mean_profile - std_profile,
            mean_profile + std_profile,
            color=self.style["color_mean"],
            alpha=0.2,
            label="Std dev",
        )

    # ------------------ Public method ------------------

    def plot(self):
        """Generate the spaghetti + mean ± std profile plot"""
        fig, ax = plt.subplots(figsize=(8, 6))
        stations_idx = self._select_stations()
        self._plot_spaghetti(ax, stations_idx)
        mean_profile, std_profile = self._compute_statistics()
        self._plot_mean_std(ax, mean_profile, std_profile)

        ax.invert_yaxis()
        ax.set_xlabel(self.variable)
        ax.set_ylabel("Depth (m)")
        ax.set_title(f"Time-averaged vertical profiles ({self.variable})")
        ax.legend()
        plt.tight_layout()
        return fig, ax


class StationMapPlotter:
    """
    Plot station locations on a world map with marker size proportional
    to the number of measurements.
    """

    def __init__(self, station_data, variable="TEMP", style=None):
        """
        Args:
            station_data: list of xr.Dataset
            variable: str, used to count number of observations per station
            style: dict with optional keys:
                color: marker color
                alpha: transparency
                max_marker_size: max point size
                figsize: figure size
        """
        self.station_data = station_data
        self.variable = variable

        # Default style
        default_style = dict(
            color="tab:blue",
            alpha=0.6,
            max_marker_size=200,
            figsize=(12, 6),
        )
        self.style = default_style if style is None else {**default_style, **style}

    # ------------------ Helper methods ------------------

    def _extract_coords_and_sizes(self):
        """
        Extract latitude, longitude, and marker size for each station
        """
        lats, lons, sizes = [], [], []
        for ds in self.station_data:
            if "LATITUDE" not in ds or "LONGITUDE" not in ds:
                print("Not present")
                continue
            lat = float(ds["LATITUDE"].values)
            lon = float(ds["LONGITUDE"].values)

            if self.variable in ds:
                n_obs = ds[self.variable].count().item()
            else:
                n_obs = 1
            # scale marker size
            sizes.append(min(n_obs, self.style["max_marker_size"]))
            lats.append(lat)
            lons.append(lon)
        return lons, lats, sizes

    # ------------------ Public method ------------------

    def plot(self):
        """
        Plot the stations on a Cartopy map.
        Returns: fig, ax
        """
        fig = plt.figure(figsize=self.style.get("figsize", (12, 6)))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.gridlines(draw_labels=True)

        lons, lats, sizes = self._extract_coords_and_sizes()
        scatter = ax.scatter(
            lons,
            lats,
            s=sizes,
            color=self.style.get("color", "tab:blue"),
            alpha=self.style.get("alpha", 0.6),
            edgecolors="k",
            transform=ccrs.PlateCarree(),
        )

        ax.set_title("Station Distribution")
        return fig, ax

