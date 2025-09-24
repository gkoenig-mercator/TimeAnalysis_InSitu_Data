import xarray as xr
from typing import List, Dict, Tuple
import numpy as np
import scipy

# -------------------- Helper functions --------------------

def compute_layer_average(ds: xr.Dataset, variable: str, depth_range: Tuple[float, float]) -> xr.DataArray:
    """Compute mean of `variable` over a given depth range for a single station."""
    if "DEPTH" not in ds.dims:
        raise ValueError("Dataset does not have a DEPTH dimension")
    min_depth, max_depth = depth_range
    return ds[variable].sel(DEPTH=slice(min_depth, max_depth)).mean(dim="DEPTH", skipna=True)


def compute_station_layer_averages(ds: xr.Dataset, variable: str, layers: Dict[str, Tuple[float, float]]) -> xr.Dataset:
    """Compute multiple layer averages for one station dataset."""
    averaged = {layer_name: compute_layer_average(ds, variable, depth_range)
                for layer_name, depth_range in layers.items()}
    result = xr.Dataset(averaged)
    # Preserve station metadata
    result.attrs["station_id"] = ds.attrs.get("station_id", "unknown")
    result.attrs["file_path"] = ds.attrs.get("file_path", "")
    return result


def stack_layer_across_stations(station_layers: List[xr.Dataset], layer_name: str) -> xr.DataArray:
    """Stack one layer from all stations along a new 'station' dimension."""
    return xr.concat([ds[layer_name] for ds in station_layers], dim="station")


def statistical_treatment(stacked_da: xr.DataArray) -> xr.Dataset:
    """Compute mean and standard deviation along 'station' dimension."""
    return xr.Dataset({
        "mean": stacked_da.mean(dim="station", skipna=True),
        "std": stacked_da.std(dim="station", skipna=True)
    })


# -------------------- Stateful class (internal) --------------------

class _StationProcessor:
    """
    Internal processor: keeps station datasets, computes layer averages and stats.
    """
    def __init__(self, variable: str, layers: Dict[str, Tuple[float, float]]):
        self.variable = variable
        self.layers = layers
        self.station_data: List[xr.Dataset] = []
        self.station_layers: List[xr.Dataset] = []

    def load_station_data(self, station_data: List[xr.Dataset]):
        self.station_data = station_data

    def compute_layers(self):
        self.station_layers = [
            compute_station_layer_averages(ds, self.variable, self.layers)
            for ds in self.station_data
        ]

    def compute_statistics(self) -> Dict[str, xr.Dataset]:
        return {
            layer_name: statistical_treatment(stack_layer_across_stations(self.station_layers, layer_name))
            for layer_name in self.layers.keys()
        }


# -------------------- High-level user function --------------------

def compute_layer_statistics_across_stations(station_data: List[xr.Dataset], variable: str, layers: Dict[str, Tuple[float, float]]) -> Dict[str, xr.Dataset]:
    """
    User-friendly function to process station datasets.

    Steps:
        1. Computes layer averages per station.
        2. Computes mean and std across stations for each layer.
        3. Returns a dictionary of layer_name -> Dataset with 'mean' and 'std'.

    Parameters
    ----------
    station_data : list of xr.Dataset
        List of station datasets loaded from io.py
    variable : str
        Variable to process (e.g., "TEMP", "SAL")
    layers : dict
        Dictionary of layer_name -> (min_depth, max_depth)

    Returns
    -------
    dict
        layer_name -> Dataset with DataArrays 'mean' and 'std'
    """
    processor = _StationProcessor(variable, layers)
    processor.load_station_data(station_data)
    processor.compute_layers()
    return processor.compute_statistics()

def compute_layer_averages_per_station(station_data: List[xr.Dataset], variable: str, layers: Dict[str, Tuple[float, float]]) -> List[xr.Dataset]:
    """
    Compute the layer averages for each station (per-station data).

    Parameters
    ----------
    station_data : list of xr.Dataset
        Raw station datasets.
    variable : str
        Variable to process (e.g., "TEMP").
    layers : dict
        Dictionary of layer_name -> (min_depth, max_depth)

    Returns
    -------
    List[xr.Dataset]
        List of datasets, one per station, containing the averaged layers.
    """
    return [compute_station_layer_averages(ds, variable, layers) for ds in station_data]

def _is_valid_profile_station(ds, variable, min_depths=2):
    """
    Check if station dataset can produce a valid vertical profile.
    """
    if variable not in ds:
        return False
    if "DEPTH" not in ds.dims:
        return False
    if len(np.unique(ds["DEPTH"].values)) < min_depths:
        return False
    return True


def _compute_station_profile(ds, variable):
    """
    Compute the time-averaged profile for one station.
    """
    return ds[variable].mean(dim="TIME", skipna=True)


def compute_time_averaged_profiles(
    station_data, variable="TEMP", min_depths=2, depth_grid=None
):
    """
    Compute time-averaged vertical profiles for each station,
    interpolated onto a common depth grid.

    Args:
        station_data (list of xr.Dataset): List of station datasets.
        variable (str): Variable to analyze (e.g., "TEMP").
        min_depths (int): Minimum number of unique depths required.
        depth_grid (array-like, optional): Common depth levels to interpolate to.
            If None, an automatic grid is created from 0 → max depth (step 10 m).

    Returns:
        xr.DataArray: Dimensions (station, DEPTH) with averaged profiles.
    """
    # If no grid is provided, make one
    if depth_grid is None:
        max_depth = 0
        for ds in station_data:
            if "DEPTH" in ds.dims:
                max_depth = max(max_depth, float(np.nanmax(ds["DEPTH"].values)))
        depth_grid = np.arange(0, max_depth + 1, 10)  # 10 m resolution

    profiles = []
    for i, ds in enumerate(station_data):
        if not _is_valid_profile_station(ds, variable, min_depths):
            continue

        profile = _compute_station_profile(ds, variable)

        # Interpolate to common depth grid
        profile_interp = profile.interp(DEPTH=depth_grid)

        # Add station dimension
        profile_interp = profile_interp.expand_dims({"station": [i]})
        profiles.append(profile_interp)

    if not profiles:
        raise ValueError("No valid profiles could be computed.")

    return xr.concat(profiles, dim="station")


