import xarray as xr
import numpy as np
from pathlib import Path
from typing import List, Optional

def load_station_files(
    file_paths: List[str],
    variables: Optional[List[str]] = None,
    remove_duplicates: bool = True
) -> List[xr.Dataset]:
    """
    Load multiple NetCDF files as per-station datasets.

    Parameters
    ----------
    file_paths : List[str]
        List of NetCDF file paths to load.
    variables : List[str], optional
        Variables to keep from the datasets (e.g., ["TEMP", "SAL"]).
        If None, keep all variables.
    remove_duplicates : bool, default True
        Remove duplicate TIME entries within each dataset.

    Returns
    -------
    station_data : List[xr.Dataset]
        List of datasets, one per station, ready for processing.
    """
    station_data = []

    for i, f in enumerate(file_paths):
        ds = xr.open_dataset(f)

        # Filter variables if requested
        if variables is not None:
            missing_vars = [v for v in variables if v not in ds.variables]
            if len(missing_vars) == len(variables):
                # Skip file if none of the requested variables are present
                print(f"Skipping {f} (no requested variables: {variables})")
                continue
            ds = ds[[v for v in variables if v in ds.variables]]

        # Drop any coordinates we don't need
        ds = ds.reset_coords(drop=True)

        # Remove duplicate TIME entries
        if remove_duplicates and "TIME" in ds.dims:
            _, idx = np.unique(ds["TIME"], return_index=True)
            ds = ds.isel(TIME=idx)

        # Keep track of the station index as a dataset attribute
        ds.attrs["station_index"] = i

        station_data.append(ds)

    print(f"Loaded {len(station_data)} stations from {len(file_paths)} files.")
    return station_data


def load_station_directory(
    directory_path: str,
    file_pattern: str = "*.nc",
    variables: Optional[List[str]] = None,
    remove_duplicates: bool = True
) -> List[xr.Dataset]:
    """
    Convenience function to load all NetCDF files in a directory.

    Parameters
    ----------
    directory_path : str
        Path to the folder containing NetCDF files.
    file_pattern : str, default "*.nc"
        Glob pattern to select files in the folder.
    variables : List[str], optional
        Variables to keep from the datasets.
    remove_duplicates : bool, default True
        Remove duplicate TIME entries.

    Returns
    -------
    List[xr.Dataset]
        List of per-station datasets.
    """
    directory = Path(directory_path)
    files = sorted(directory.glob(file_pattern))
    return load_station_files([str(f) for f in files], variables=variables, remove_duplicates=remove_duplicates)
