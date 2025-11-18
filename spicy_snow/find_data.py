"""
Functions to search and download Sentinel-1 RTC images for specific geometries and dates
"""
from pathlib import Path
from itertools import chain

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxa
import geopandas as gpd

# opera s1
import asf_search as asf
from asf_search import download_url
# viirs snow cover
import earthaccess
from tqdm import tqdm

from shapely.geometry import box, Polygon

from spicy_snow.utils.checks import validate_aoi, validate_dates, within_conus
from spicy_snow.utils.spicy_logging import temp_silence_logger
import logging
log = logging.getLogger(__name__)

def find_snowcover_urls(aoi, start_date = None, stop_date = None, date_list = None):
    
    aoi = validate_aoi(aoi)
    start_date, stop_date = validate_dates(start_date, stop_date)
    
    # https://nsidc.org/data/vj110a1f/versions/2
    results = earthaccess.search_data(
        short_name = "VJ110A1F",
        downloadable = True,
        bounding_box = aoi.bounds,
        temporal = (start_date, stop_date),
    )
    
    # Flatten all URLs
    snowcover_urls = list(chain.from_iterable([r.data_links() for r in results]))
    
    if date_list is not None:
        date_list = pd.to_datetime(date_list)
        # extract dates from filenames
        url_dates = pd.to_datetime([Path(url).stem.split('.')[1] for url in snowcover_urls], format="A%Y%j")
        snowcover_urls = [url for url, dt in zip(snowcover_urls, url_dates) if dt in date_list]

    return snowcover_urls

def get_sentinel1_urls(start_date, stop_date, aoi, source = 'opera'):
    """
    Query ASF (Alaska Satellite Facility) for Sentinel-1 SAR products over a given AOI and date range.

    Args:
        start_date (str or datetime-like): Start of the date range for the search.
        stop_date (str or datetime-like): End of the date range for the search.
        aoi (list, np.ndarray, or shapely.geometry.Box): Area of interest. Can be a 4-element bounding box 
            [xmin, ymin, xmax, ymax], a numpy array of coordinates, or a shapely Box.
        source (str, optional): Which ASF source to use. Options are:
            - 'opera' → RTC products
            - 'hyp3'  → GRD_HD products
            Defaults to 'opera'.

    Returns:
        pd.DataFrame: Flattened GeoJSON features from ASF search, including SAR product metadata such as 
        scene name, polarization, granule info, URLs, and acquisition times.

    Raises:
        ValueError: If an unknown source string is provided.
        AssertionError / IndexError: If AOI or dates are invalid (delegated to validate_aoi / validate_dates).
    """
    aoi = validate_aoi(aoi)
    start_date, stop_date = validate_dates(start_date, stop_date)

    platform = asf.PLATFORM.SENTINEL1
    if source == 'opera':
        product_type = asf.PRODUCT_TYPE.RTC
    elif source == 'hyp3':
        product_type = asf.PRODUCT_TYPE.GRD_HD
    else:
        raise ValueError(f"Unknown source: {source}")

    results = asf.geo_search(intersectsWith = aoi.wkt, 
                   start = start_date, 
                   end = stop_date, 
                   processingLevel = product_type, 
                   platform = platform)
    
    results_df = pd.json_normalize(results.geojson(), record_path = ['features'])

    sentinel1_urls = get_urls_from_asf_search(results_df)

    return sentinel1_urls

def get_urls_from_asf_search(asf_results_df):
    """
    Convert dataframe of ASF search results to urls.
    """
    urls = []

    for _, row in asf_results_df.iterrows():
        main = row.get('properties.url')
        if main:
            urls.append(main)

        extras = row.get('properties.additionalUrls', [])
        if extras:
            urls.extend(extras)

    return urls

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
    Optional subset ASF search results with filters and AOI intersection.

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

# forest cover functions
def download_proba_v(out_fp):
    # this is the url from Lievens et al. 2021 paper
    fcf_url = 'https://zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Tree-CoverFraction-layer_EPSG-4326.tif'
    # download just forest cover fraction to out file
    out_fp = download_url(url = fcf_url, filename = out_fp)

    return out_fp