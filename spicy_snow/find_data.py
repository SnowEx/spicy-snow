"""
Functions to search and download Sentinel-1 RTC images for specific geometries and dates
"""
import re
from pathlib import Path
from itertools import chain

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxa
import geopandas as gpd
from pyproj import Transformer

# opera s1
import asf_search as asf
from asf_search import download_url
# viirs snow cover
import earthaccess
from tqdm import tqdm

from shapely.geometry import box, Polygon

from spicy_snow.utils.checks import validate_aoi, validate_dates
from spicy_snow.hyp3_pipeline import hyp3_pipeline

import logging
log = logging.getLogger(__name__)


# MODIS/VIIRS sinusoidal tile grid constants (10 deg per tile at the equator)
_VIIRS_TILE_SIZE = 1111950.519667
_VIIRS_SINU_PROJ = (
    "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 "
    "+ellps=WGS84 +datum=WGS84 +units=m +no_defs"
)
_VIIRS_TILE_RE = re.compile(r"h(\d{2})v(\d{2})")


def _viirs_tiles_for_aoi(aoi_bounds):
    """Return the set of (h, v) MODIS/VIIRS sinusoidal tiles that an AOI
    bbox (xmin, ymin, xmax, ymax in EPSG:4326) actually intersects.

    Densely samples the AOI boundary in lat/lon, projects each point to
    sinusoidal, and reads off the tile indices. This is needed because
    earthaccess's CMR matching uses each granule's WGS84 polygon, which
    at high latitudes spans vastly wider longitudes than the tile's real
    coverage (e.g. h20v01 over Asia matches a Fairbanks bbox).
    """
    transformer = Transformer.from_crs("EPSG:4326", _VIIRS_SINU_PROJ, always_xy=True)
    xmin, ymin, xmax, ymax = aoi_bounds
    n = 30
    lons = np.concatenate([
        np.linspace(xmin, xmax, n), np.linspace(xmin, xmax, n),
        np.full(n, xmin), np.full(n, xmax),
    ])
    lats = np.concatenate([
        np.full(n, ymin), np.full(n, ymax),
        np.linspace(ymin, ymax, n), np.linspace(ymin, ymax, n),
    ])
    sx, sy = transformer.transform(lons, lats)
    tiles = set()
    for x, y in zip(sx, sy):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        h = int(np.floor(x / _VIIRS_TILE_SIZE + 18))
        v = int(np.floor(9 - y / _VIIRS_TILE_SIZE))
        if 0 <= h < 36 and 0 <= v < 18:
            tiles.add((h, v))
    return tiles


def _viirs_tile_from_url(url):
    """Parse 'h##v##' tile ID from a VIIRS granule URL; None if absent."""
    m = _VIIRS_TILE_RE.search(url)
    return (int(m.group(1)), int(m.group(2))) if m else None

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
    # data_links() also returns .h5.xml metadata sidecars; keep only the HDF5
    # granules so downstream h5py.File() doesn't choke on the XML files.
    snowcover_urls = [u for u in snowcover_urls if u.endswith('.h5')]
    # CMR's WGS84 polygon match over-includes tiles at high latitudes;
    # restrict to URLs whose (h, v) sinusoidal tile actually covers the AOI.
    needed_tiles = _viirs_tiles_for_aoi(aoi.bounds)
    snowcover_urls = [u for u in snowcover_urls if _viirs_tile_from_url(u) in needed_tiles]
    if not snowcover_urls:
        raise ValueError(
            f"No VIIRS snowcover tiles intersect AOI {aoi.bounds}; "
            f"earthaccess returned no .h5 granules in tiles {needed_tiles}."
        )

    if date_list is not None:
        date_list = pd.to_datetime(date_list)
        # extract dates from filenames
        url_dates = pd.to_datetime([Path(url).stem.split('.')[1] for url in snowcover_urls], format="A%Y%j")
        snowcover_urls = [url for url, dt in zip(snowcover_urls, url_dates) if dt in date_list]

    return snowcover_urls

def get_sentinel1_urls(start_date, stop_date, aoi, source = 'opera', job_name = None):
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
    
    assert source in ['opera', 'hyp3'], "Source must be either 'opera' or 'hyp3'"

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

    if source == 'opera':
        # short-circuit and return opera urls
        opera_urls = get_opera_urls_from_asf_search(results_df)
        return opera_urls
    
    if job_name is None:
        job_name = f'spicy_snow_hyp3_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}'
        log.debug(f"No job name provided. Using generated name: {job_name}")

    rtc_jobs = hyp3_pipeline(results_df, job_name = job_name, existing_job_name = job_name)
    hyp3_urls = get_hy3p_urls_from_jobs(rtc_jobs)
    return hyp3_urls


def get_hy3p_urls_from_jobs(rtc_jobs):
    """
    Extract download URLs from completed HyP3 RTC jobs.
    """
    urls = []

    for job in rtc_jobs:
        u = job.files[0]['url']
        urls.append(u.replace('.zip', '_VV.tif'))
        urls.append(u.replace('.zip', '_VH.tif'))

    return urls


def get_opera_urls_from_asf_search(asf_results_df):
    """
    Convert dataframe of ASF search results to opera urls.
    """
    urls = []

    for _, row in asf_results_df.iterrows():
        main = row.get('properties.url')
        if main:
            urls.append(main)

        extras = row.get('properties.additionalUrls', [])
        if extras:
            urls.extend(extras)
    
    # only return relevant opera files
    urls = [u for u in urls if u.endswith(('_VV.tif', '_VH.tif', '_mask.tif'))]

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