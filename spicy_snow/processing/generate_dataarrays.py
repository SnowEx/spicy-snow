from pathlib import Path
from collections import defaultdict
import tempfile

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import box, Polygon
import h5py

# forest cover
import pygeohydro as gh

import sys
sys.path.append('/Users/zmhoppinen/Documents/spicy-snow/spicy_snow/utils')
from raster import tif_to_dataarray, combine_close_images, da_to01
from checks import validate_aoi, within_conus
from download import download_proba_v

import logging
log = logging.getLogger(__name__)

def generate_sentinel1_dataarray(s1_fps, aoi, pol, ref = None):
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
    aoi = validate_aoi(aoi)
    # file_to_row = map_files_to_asf_properties(file_list, search_df)
    pol = pol.lower()
    da_list = []

    for fp in s1_fps:
        if fp.stem.lower().endswith(pol):
            
            time = pd.to_datetime(fp.stem.split('_')[4], format = '%Y%m%dT%H%M%SZ')
            track = int(fp.stem.split('_')[3].split('-')[0][1:])
            satellite_number = fp.stem.split('_')[6]

            mask_fp = fp.parent.joinpath(f'{fp.stem[:-3]}_mask.tif')
            if mask_fp.exists():
                mask = xr.open_dataarray(mask_fp)[0]
                # section 4.3 https://d2pn8kiwq2w21t.cloudfront.net/documents/ProductSpec_RTC-S1.pdf
                # 0 = not affected by layover or shadow
                mask = mask.where(mask == 0)


            # for first s1 setup dataarray by subsetting to aoi
            if len(da_list) == 0 and ref is None: 
                da = tif_to_dataarray(fp, mask = mask, time = time, area = aoi)
                ref = da.isel(time = 0)
            
            # afterwards reproject others to this coordinate grid
            else:
                da = tif_to_dataarray(fp, mask = mask, time = time, ref_da = ref)
            

            # add relative orbit information
            da = da.assign_coords(track = ('time', [track]))

            # add satellite information
            da = da.assign_coords(platform = ('time', [satellite_number]))
            
            da_list.append(da)

    if not da_list:
        return None

    # concatenate along time dimension
    stacked = xr.concat(da_list, dim='time').sortby('time')

    # next stack close times spatially
    stacked = combine_close_images(stacked)
    stacked.name = pol
    return stacked

def generate_snowcover_dataarray(snowcover_fps, ref = None):
    # Organize tiles by date
    tiles_by_date = defaultdict(list)

    for fp in snowcover_fps:
        with h5py.File(fp, 'r') as f:
            # grab datetime from properties
            # based on this documentation. Section 1.2.3
            # https://nsidc.org/sites/default/files/documents/user-guide/multi_vnp10a1f-v002-userguide.pdf
            dt = pd.to_datetime(fp.stem.split('.')[1], format = 'A%Y%j')
            
            # get snow cover data
            data = f['HDFEOS']['GRIDS']['VIIRS_Grid_IMG_2D']['Data Fields']['CGF_NDSI_Snow_Cover'][:]

            # get x and y dimensions
            xdim = f['HDFEOS']['GRIDS']['VIIRS_Grid_IMG_2D']['XDim'][:]
            ydim = f['HDFEOS']['GRIDS']['VIIRS_Grid_IMG_2D']['YDim'][:]
            
            # generate xarray dataarray
            da_tile = xr.DataArray(
                data,
                coords={'y': ydim, 'x': xdim},
                dims=['y', 'x'],
            )
            
            tiles_by_date[dt].append(da_tile)

    # For each date, combine tiles spatially, then stack along time
    daily_arrays = []
    for dt, tile_list in tiles_by_date.items():
        # mosaic tiles for this day
        da_day = xr.combine_by_coords(tile_list, combine_attrs='override')
        # add time dimension
        da_day = da_day.expand_dims(time=[dt])
        daily_arrays.append(da_day)

    # concatenate all days along time
    snowcover_da = xr.concat(daily_arrays, dim='time').sortby('time')
    snowcover_da.name = 'snowcover'

    # convert from NASA's sinusoidal project
    # proj string comes from Table 3 of
    # https://nsidc.org/sites/default/files/documents/user-guide/multi_vnp10a1f-v002-userguide.pdf
    src_crs = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    snowcover_da = snowcover_da.rio.write_crs(src_crs)

    if snowcover_da.max() > 1:
        log.debug("snowcover_da max > 1 so normalizing from 0 to 100 -> 0 to 1")
        snowcover_da = da_to01(snowcover_da, old_min = 0, old_max = 100)
        log.debug(f"New snowcover_da max is {snowcover_da.max()} and min is {snowcover_da.min()}")


    if ref is not None:
        snowcover_da = snowcover_da.rio.reproject_match(ref)
    
    return snowcover_da

def convert_snowcover_dates_to_s1_overpasses(viirs_snowcover, s1_da):
    precise_snowcover = xr.zeros_like(s1_da)
    for s1_time in s1_da.time:
        mask = viirs_snowcover.time.dt.date == s1_time.dt.date
        precise_snowcover.loc[dict(time = s1_time)] = viirs_snowcover.sel(time = mask).isel(time = 0)
    return precise_snowcover

def get_nlcd(aoi):
    g = gpd.GeoSeries([box(*aoi.bounds)], crs='EPSG:4326')
    fcf_da = gh.nlcd_bygeom(geometry = g)[0]['canopy_2021']

    return fcf_da

def generate_forest_fraction_dataarray(aoi, ref = None) -> xr.Dataset:
    """
    Download PROBA-V forest-cover-fraction images.

    Args:
    aoi: 4 element box of AOI

    Returns:
    dataset: Forest cover fraction dataarray over aoi. NLCD if in US otherwise Proba-v
    """
    aoi = validate_aoi(aoi)
    # first check if in us
    if not within_conus(aoi): 
        log.info(f'AOI outside of CONUS. Using Proba-V datasets')
        tmp_dir = tempfile.gettempdir()
        fcf = xr.open_dataarray(download_proba_v(), tmp_dir.joinpath('fcf.tif'))[0]
    else: 
        log.info(f'AOI inside of CONUS. NLCD 2021 Forest Cover')
        fcf = get_nlcd(aoi)

    # reproject FCF and clip to match dataset
    if ref is not None:
        log.debug(f"Clipping FCF to {ref.rio.bounds()}")
        fcf = fcf.rio.reproject_match(ref)

    # if max is greater than 1 () set to 0-1
    if fcf.max() > 1:
        log.debug("fcf max > 1 so normalizing from 0 to 100 -> 0 to 1")
        fcf = da_to01(fcf, old_min = 0, old_max = 100)
        log.debug(f"New fcf max is {fcf.max()} and min is {fcf.min()}")
    
    assert fcf.max() <= 1, "Forest cover fraction must be bounded 0-1"
    assert fcf.min() >= 0, "Forest cover fraction must be bounded 0-1"

    log.debug(f'FCF min: {fcf.min()}')
    log.debug(f'FCF max: {fcf.max()}')
    log.debug(f'FCF mean: {fcf.mean()}')

    fcf.name = 'fcf'

    return fcf

# End of file