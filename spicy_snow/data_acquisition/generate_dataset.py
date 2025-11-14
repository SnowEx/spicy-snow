from pathlib import Path
from functools import reduce

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxa

def tif_to_dataarray(fp, time, area, ref_da = None):
    """
    Open a single band, reproject to EPSG:4326, clip and pad to AOI, assign band name.
    """
    import rioxarray as rxa
    name = fp.stem
    if not any(name.endswith(suffix) for suffix in ['_VV', '_VH', '_mask', '_inc']):
        return None
    
    img = xr.open_dataarray(fp, masked=True)[0]
    
    if ref_da is None:
        img = img.rio.reproject('EPSG:4326')
        img = img.rio.clip_box(*area.bounds)
        img = img.rio.pad_box(*area.bounds)
    else:
        img = img.rio.reproject_match(ref_da)

    dt = pd.to_datetime(time)
    return img.expand_dims(time = [dt])
    
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

def mosaic_group(sub):
    # sub is a DataArray with 'time' dimension
    merged = reduce(lambda a, b: a.combine_first(b), [sub.isel(time=i) for i in range(sub.sizes['time'])])
    merged = merged.expand_dims(time=[pd.to_datetime(sub['time']).mean()])  # assign average time
    merged = merged.dropna('x', how = 'all').dropna('y', how = 'all')
    return merged

def combine_close_images(da, time_tol = pd.Timedelta('2min')):
    # Define tolerance
    time_tol = pd.Timedelta('2min')

    time_diff = da['time'].diff('time', label='upper')

    # Convert to NumPy, prepend zero along the 'time' axis
    data_padded = np.concatenate([[0], time_diff.values], axis=0)

    # rebuild DataArray with same 'time' coordinate
    time_diff = xr.DataArray(
        data_padded,
        dims=['time'],
        coords={'time': da['time']},
        name='time_diff'
    )

    # cumulative sum adds when over time tolerance
    groups = (time_diff > time_tol).cumsum(dim='time')

    # group images closer than time difference
    return da.groupby(groups).map(mosaic_group)

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
