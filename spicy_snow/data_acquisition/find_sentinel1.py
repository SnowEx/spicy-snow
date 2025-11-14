"""
Functions to search and download Sentinel-1 RTC images for specific geometries and dates
"""
from pathlib import Path
from functools import reduce
from datetime import date

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxa

import asf_search as asf

from shapely.geometry import box, Polygon

from tqdm import tqdm

import logging
log = logging.getLogger(__name__)


def get_asf_search_results(start_date, end_date, aoi, source = 'opera'):
    """
    
    """
    aoi = validate_aoi(aoi)
    start_date, end_date = validate_dates(start_date, end_date)

    platform = asf.PLATFORM.SENTINEL1
    if source == 'opera':
        product_type = asf.PRODUCT_TYPE.RTC
    elif source == 'hyp3':
        product_type = asf.PRODUCT_TYPE.GRD_HD
    else:
        raise ValueError(f"Unknown source: {source}")

    results = asf.geo_search(intersectsWith = aoi.wkt, 
                   start = start_date, 
                   end = end_date, 
                   processingLevel = product_type, 
                   platform = platform)
    
    results_df = pd.json_normalize(results.geojson(), record_path = ['features'])

    return results_df

def subset_asf_search_results(
    results_df, 
    aoi=None, 
    path_numbers=None, 
    direction=None, 
    polarization=None, 
    start_time=None, 
    stop_time=None, 
    scene_name=None
):
    """
    Subset ASF search results with optional filters and AOI intersection.

    Args:
        results_df (pd.DataFrame): ASF search results.
        aoi (list/tuple/shapely.geometry): AOI as [xmin, ymin, xmax, ymax] or shapely geometry.
        path_numbers (list[int], optional): Filter by multiple path numbers.
        direction (str, optional): Filter by flightDirection.
        polarization (str, optional): Filter by polarization.
        start_time (str/pd.Timestamp, optional): Filter results after this time.
        stop_time (str/pd.Timestamp, optional): Filter results before this time.
        scene_name (str, optional): Filter by sceneName.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    df = results_df.copy()

    # -- Convert AOI to shapely box if provided as list --
    if aoi is not None:
        if isinstance(aoi, (list, tuple)) and len(aoi) == 4:
            aoi = box(*aoi)
        gdf = gpd.GeoDataFrame(
            df,
            geometry=df['geometry.coordinates'].apply(lambda coords: Polygon(coords[0])),
            crs="EPSG:4326"
        )
        gdf = gdf[gdf.intersects(aoi)]
        df = pd.DataFrame(gdf.drop(columns='geometry'))

    # -- Path number filtering (support multiple) --
    if path_numbers is not None:
        df = df[df['properties.pathNumber'].isin(path_numbers)]
    
    # -- Other optional filters --
    if direction is not None:
        df = df[df['properties.flightDirection'] == direction]
    if polarization is not None:
        df = df[df['properties.polarization'] == polarization]
    if start_time is not None:
        start_time = pd.to_datetime(start_time)
        df = df[pd.to_datetime(df['properties.startTime']) >= start_time]
    if stop_time is not None:
        stop_time = pd.to_datetime(stop_time)
        df = df[pd.to_datetime(df['properties.stopTime']) <= stop_time]
    if scene_name is not None:
        df = df[df['properties.sceneName'] == scene_name]

    return df    

def get_urls_from_asf_search(results_df):

    urls = []

    for _, row in results_df.iterrows():
        main = row.get('properties.url')
        if main:
            urls.append(main)

        extras = row.get('properties.additionalUrls', [])
        if extras:
            urls.extend(extras)

    return urls

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

def validate_dates(start_date, end_date):
    """
    Validate a start and end date for Sentinel-1 SAR availability.

    Rules:
    - Convert both dates to pandas Timestamps.
    - Dates must be >= 2014 (Sentinel-1A launch).
    - Dates must be <= today.
    - start_date must be < end_date.
    - Warn if dates overlap Sentinel-1B outage (Dec 2021 → present).
    - Warn if dates fall before Sentinel-1C becomes operational (May 20, 2025).
    """
    # ---- Convert to Timestamps ----
    if start_date is not None:
        start = pd.to_datetime(start_date)
    else:
        raise ValueError("start_date cannot be None")

    if end_date is not None:
        end = pd.to_datetime(end_date)
    else:
        raise ValueError("end_date cannot be None")

    # ---- Basic range checks ----
    if start.year < 2014 or end.year < 2014:
        raise ValueError("Dates must be in or after 2014 (Sentinel-1A launch).")

    today = pd.to_datetime(date.today())

    if start > today or end > today:
        raise ValueError("Dates cannot be in the future.")

    if start >= end:
        raise ValueError("start_date must be earlier than end_date.")

    # ---- Special Sentinel mission warnings ----
    # S1B failed: Dec 23, 2021 →
    # S1C operational: May 20, 2025

    s1b_fail = pd.to_datetime("2021-12-23")
    s1c_start = pd.to_datetime("2025-05-20")
    if end >= s1b_fail and end < s1c_start:
        warnings.warn(
            "Date range intersects the Sentinel-1B outage period (Dec 2021 → present). "
            "Only S1A data will be available."
        )

    return start, end


def validate_aoi(aoi):
    """
    Validate and normalize an AOI to a shapely Box geometry.

    Accepts:
    - Iterable of four floats [xmin, ymin, xmax, ymax]
    - Dicts using common key conventions:
        {'xmin','ymin','xmax','ymax'} or
        {'west','south','east','north'} or
        {'minx','miny','maxx','maxy'}
    - Existing shapely geometry (Box, Polygon, etc.)

    Returns:
        shapely.geometry.Polygon (a box)
    """
    if isinstance(aoi, Polygon): return aoi

    if isinstance(aoi, list) or isinstance(aoi, np.ndarray):
        if len(aoi) == 4:

            # AOI given as [xmin, ymin, xmax, ymax] (or any four floats)
            xmin, ymin, xmax, ymax = aoi  # your 4-element iterable

            return box(xmin, ymin, xmax, ymax)
        
    if isinstance(aoi, dict):
        key_sets = [
            ("xmin", "ymin", "xmax", "ymax"),
            ("west", "south", "east", "north"),
            ("minx", "miny", "maxx", "maxy"),
        ]

        for keys in key_sets:
            if all(k in aoi for k in keys):
                xmin, ymin, xmax, ymax = (float(aoi[k]) for k in keys)

                # auto-fix reversed ranges
                if xmin > xmax:
                    xmin, xmax = xmax, xmin
                if ymin > ymax:
                    ymin, ymax = ymax, ymin

                return box(xmin, ymin, xmax, ymax)

    raise ValueError(
        f"AOI dict must contain one of these key sets: {key_sets} "
        f"but received keys: {list(aoi.keys())}"
    )