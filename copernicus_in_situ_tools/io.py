import xarray as xr
import numpy as np
from pathlib import Path
from typing import List, Tuple


# -------------------- Helper functions --------------------

def load_single_file(file_path: str, variable: str) -> xr.Dataset:
    """
    Load a single NetCDF file and keep only the requested variable
    and coordinates.

    Parameters
    ----------
    file_path : str
        Path to a NetCDF file.
    variable : str
        Variable to load (e.g., "TEMP", "SAL").

    Returns
    -------
    xr.Dataset or None
        Dataset with only the requested variable.
        Returns None if the variable is not present.
    """
    ds = xr.open_dataset(file_path)

    if variable not in ds.variables:
        print(f"Skipping {file_path} (no {variable})")
        return None

    # Keep only the variable, drop extra coordinates for simplicity
    ds = ds[[variable]].reset_coords(drop=True)

    return ds


def remove_duplicate_times(ds: xr.Dataset) -> xr.Dataset:
    """
    Remove duplicate TIME entries in a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset that contains a TIME dimension.

    Returns
    -------
    xr.Dataset
        Dataset with unique TIME entries.
    """
    if "TIME" not in ds.dims:
        return ds

    _, idx = np.unique(ds["TIME"], return_index=True)
    return ds.isel(TIME=idx)


def assign_station_id(ds: xr.Dataset, station_index: int, file_path: str) -> xr.Dataset:
    """
    Assign a unique station ID to the dataset and store as an attribute.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to annotate.
    station_index : int
        Index of the station in the file list.
    file_path : str
        Original file path, for reference.

    Returns
    -------
    xr.Dataset
        Same dataset with 'station_id' attribute.
    """
    station_id = f"station_{station_index}"
    ds.attrs["station_id"] = station_id
    ds.attrs["file_path"] = str(file_path)
    return ds


# -------------------- Main functions --------------------

def load_files(file_paths: List[str], variable: str) -> List[xr.Dataset]:
    """
    Load multiple NetCDF files as per-station datasets, each with a unique station ID.

    Parameters
    ----------
    file_paths : list of str
        List of NetCDF file paths to load.
    variable : str
        Variable to extract from each file (e.g., "TEMP").

    Returns
    -------
    List[xr.Dataset]
        List of cleaned datasets, each corresponding to a station.
    """
    station_data = []

    for i, f in enumerate(file_paths):
        ds = load_single_file(f, variable)
        if ds is None:
            continue
        ds = remove_duplicate_times(ds)
        ds = assign_station_id(ds, i, f)
        station_data.append(ds)

    print(f"Loaded {len(station_data)} stations from {len(file_paths)} files.")
    return station_data


def load_station_directory(directory_path: str, variable: str, file_pattern: str = "*.nc") -> List[xr.Dataset]:
    """
    Load all NetCDF files in a directory matching the pattern.

    Each dataset will have a unique station ID and cleaned TIME dimension.

    Parameters
    ----------
    directory_path : str
        Path to the folder containing NetCDF files.
    variable : str
        Variable to extract from each file (e.g., "TEMP").
    file_pattern : str, optional
        Glob pattern to select files (default "*.nc").

    Returns
    -------
    List[xr.Dataset]
        List of cleaned datasets, one per station.
    """
    directory = Path(directory_path)
    files = sorted(directory.glob(file_pattern))
    return load_files([str(f) for f in files], variable)

