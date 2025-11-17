from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import sys
sys.path.append('/Users/zmhoppinen/Documents/spicy-snow/spicy_snow/utils')
from raster import tif_to_dataarray, combine_close_images

def file_paths_to_pol_dataarray(file_list, search_df, area, pol='VV'):
    """
    Create an xarray.Dataset for a single polarization ('VV' or 'VH') from a list of files.

    Args:
        file_list (list[Path]): List of local files.
        search_df (pandas dataframe): Search results used
        area (shapely.geometry.Polygon): AOI for clipping/padding.
        pol (str): Polarization to select ('VV' or 'VH').

    Returns:
        xarray.Dataset: Dataset with a single variable named 'VV' or 'VH', time dimension.
    """
    file_to_row = map_files_to_asf_properties(file_list, search_df)
    pol = pol.upper()
    da_list = []

    for fp in file_list:
        if fp.stem.upper().endswith(pol):
            row = file_to_row.get(fp)
            if row is None:
                continue  # skip files without ASF properties mapping
            time = row['properties.startTime']
            
            if len(da_list) == 0: da = tif_to_dataarray(fp, time, area)
            else: da = tif_to_dataarray(fp, time, area, ref_da = da_list[0].isel(time = 0))
            
            if da is None: continue
            da_list.append(da)

    if not da_list:
        return None

    # concatenate along time dimension
    stacked = xr.concat(da_list, dim='time').sortby('time')
    return stacked


def map_files_to_asf_properties(file_list, search_df):
    """
    Map local files to the corresponding ASF search results row based on filename stem.

    Usage:
        fps_properties = map_files_to_asf_properties(fps, search_df)
    Args:
        file_list (list[Path]): List of local downloaded files.
        search_df (pd.DataFrame): ASF search results.

    Returns:
        dict: mapping Path -> row (as Series) from search_df
    """
    # precompute list of URL stems for faster lookup
    url_stems_list = [
        [Path(url).stem for url in row.get('properties.additionalUrls', [])] +
        [Path(row.get('properties.url')).stem]  # also include main URL
        for _, row in search_df.iterrows()
    ]

    file_to_row = {}
    for fp in file_list:
        stem = fp.stem
        # find matching row
        for i, stems in enumerate(url_stems_list):
            if stem in stems:
                file_to_row[fp] = search_df.iloc[i]
                break
        else:
            # no match
            file_to_row[fp] = None

    return file_to_row
