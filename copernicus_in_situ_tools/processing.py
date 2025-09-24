import xarray as xr
from typing import List, Dict, Tuple

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

def process_station_data(station_data: List[xr.Dataset], variable: str, layers: Dict[str, Tuple[float, float]]) -> Dict[str, xr.Dataset]:
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
