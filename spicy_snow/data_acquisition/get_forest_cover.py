"""
Functions to download PROBA-V forest-cover-fraction images for specific geometries
"""


import geopandas as gpd
import xarray as xr
import rioxarray as rxa

from shapely.geometry import box

from asf_search import download_url
import pygeohydro as gh

import logging
log = logging.getLogger(__name__)

def download_proba_v(out_fp):
    # this is the url from Lievens et al. 2021 paper
    fcf_url = 'https://zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Tree-CoverFraction-layer_EPSG-4326.tif'
    # download just forest cover fraction to out file
    download_url(url = fcf_url, filename = out_fp)
    # open as dataArray and return
    fcf = xr.open_dataarray(out_fp)[0]

    return fcf

def within_conus(dataset):
    """
    Quick bounding-box check using approximate contiguous US lat/lon limits.
    Assumes ds.rio.bounds() returns (xmin, ymin, xmax, ymax) in EPSG:4326.
    """
    xmin, ymin, xmax, ymax = dataset.rio.bounds()

    # Approximate CONUS bounding envelope
    CONUS_XMIN, CONUS_XMAX = -125, -66
    CONUS_YMIN, CONUS_YMAX = 24, 50

    # Check intersection / overlap
    intersects_lon = not (xmax < CONUS_XMIN or xmin > CONUS_XMAX)
    intersects_lat = not (ymax < CONUS_YMIN or ymin > CONUS_YMAX)

    return intersects_lon and intersects_lat

def download_nlcd(dataset):
    g = gpd.GeoSeries([box(*dataset.rio.bounds())], crs=dataset.rio.crs)
    fcf_da = gh.nlcd_bygeom(geometry = g)[0]['canopy_2021']

    return fcf_da

def download_fcf(dataset: xr.Dataset, fcf_fp: str) -> xr.Dataset:
    """
    Download PROBA-V forest-cover-fraction images.

    Args:
    dataset: large dataset to add IMS data to
    out_fp: filepath to save tiff of results

    Returns:
    dataset: large dataset with 'fcf' added as data variable
    """
    log.debug("Downloading Forest Cover")

    # first check if in us
    if not within_conus(dataset): 
        log.info(f'AOI outside of CONUS. Using Proba-V datasets')
        fcf = download_proba_v(fcf_fp)
    else: 
        log.info(f'AOI inside of CONUS. NLCD 2021 Forest Cover')
        fcf = download_nlcd(dataset)

    # reproject FCF and clip to match dataset
    log.debug(f"Clipping FCF to {dataset['s1'].rio.bounds()}")
    fcf = fcf.rio.reproject_match(dataset['s1'])
    # if max is greater than 1 set to 0-1
    if fcf.max() >= 50:
        log.debug("fcf max > 50 so dividing by 100")
        fcf = fcf / 100
        log.debug(f"New fcf max is {fcf.max()} and min is {fcf.min()}")
    
    assert fcf.max() <= 1, "Forest cover fraction must be bounded 0-1"
    assert fcf.min() >= 0, "Forest cover fraction must be bounded 0-1"

    log.debug(f'FCF min: {fcf.min()}')
    log.debug(f'FCF max: {fcf.max()}')
    log.debug(f'FCF mean: {fcf.mean()}')
    
    # merge FCF and name it 'fcf' as a data variable
    dataset = xr.merge([dataset, fcf.rename('fcf')])

    return dataset

# End of file