import xarray as xr
import numpy as np
import pandas as pd
from typing import List, Optional, Dict


def compute_layer_averages(
    station_data: List[xr.Dataset],
    layers: List[Dict],
    variable: str = "TEMP"
) -> List[xr.Dataset]:
    """
    Compute depth-averaged values for multiple layers.

    Parameters
    ----------
    station_data : List[xr.Dataset]
        List of per-station datasets.
    layers : List[dict]
        Each dict must have:
            - 'name': name of the layer
            - 'depth': tuple (min_depth, max_depth)
    variable : str
        Variable to average (e.g., "TEMP", "SAL").

    Returns
    -------
    List[xr.Dataset]
        Updated datasets with a variable for each layer, e.g. surface_TEMP, deep_TEMP.
    """
    processed_data = []

    for ds in station_data:
        if variable not in ds.variables:
            continue

        averaged_vars = {}
        for layer in layers:
            min_depth, max_depth = layer["depth"]
            layer_name = layer["name"]
            averaged_vars[f"{layer_name}_{variable}"] = ds[variable].sel(
                DEPTH=slice(min_depth, max_depth)
            ).mean(dim="DEPTH", skipna=True)

        averaged_ds = xr.Dataset(averaged_vars)

        # Preserve station metadata
        if "station_index" in ds.attrs:
            averaged_ds.attrs["station_index"] = ds.attrs["station_index"]

        processed_data.append(averaged_ds)

    return processed_data


def compute_global_average(
    station_data: List[xr.Dataset],
    variable: str = "TEMP",
    common_time: Optional[pd.DatetimeIndex] = None
) -> xr.Dataset:
    """
    Compute global mean and standard deviation across stations for all layers.

    Parameters
    ----------
    station_data : List[xr.Dataset]
        List of per-station datasets with layer averages.
    variable : str
        Base variable name (e.g., "TEMP") for which to compute averages.
    common_time : pd.DatetimeIndex, optional
        Optional common time axis to reindex all stations.

    Returns
    -------
    xr.Dataset
        Dataset with global mean and std for each layer.
        Variables: {layer}_mean, {layer}_std
    """
    if not station_data:
        return xr.Dataset()

    # Determine layers from first station dataset
    layer_names = [name for name in station_data[0].data_vars if name.endswith(f"_{variable}")]

    layer_mean_vars = {}
    layer_std_vars = {}

    for layer_var in layer_names:
        layer_list = []

        for ds in station_data:
            if layer_var not in ds.variables:
                continue
            if common_time is not None:
                # Reindex to common time
                layer_list.append(ds[layer_var].reindex(TIME=common_time, method="nearest"))
            else:
                layer_list.append(ds[layer_var])

        if layer_list:
            stack = xr.concat(layer_list, dim="station", coords="minimal")
            layer_mean_vars[f"{layer_var}_mean"] = stack.mean(dim="station", skipna=True)
            layer_std_vars[f"{layer_var}_std"] = stack.std(dim="station", skipna=True)

    return xr.Dataset({**layer_mean_vars, **layer_std_vars})
