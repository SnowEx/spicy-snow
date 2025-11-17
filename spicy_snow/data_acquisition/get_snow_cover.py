"""
Functions to download 500 m VIRRS Snowcover.

https://nsidc.org/data/vj110a1f/versions/2
"""

from pathlib import Path
from collections import defaultdict
from itertools import chain

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray

import earthaccess
import h5py

import sys
sys.path.append('/Users/zmhoppinen/Documents/spicy-snow/spicy_snow/utils')
from checks import validate_dates, validate_aoi

import logging
log = logging.getLogger(__name__)

def find_snowcover_urls(aoi, start_date, stop_date):
    aoi = validate_aoi(aoi)
    start_date, stop_date = validate_dates(start_date, stop_date)
    # https://nsidc.org/data/vj110a1f/versions/2
    results = earthaccess.search_data(
        short_name = "VJ110A1F",
        downloadable = True,
        bounding_box = aoi.bounds,
        temporal = (start_date, stop_date),
    )

    # flatten to 1d list
    snowcover_urls = list(chain.from_iterable([r.data_links() for r in results]))

    return snowcover_urls

def generate_snowcover_dataarray(snowcover_fps):
    # Organize tiles by date
    tiles_by_date = defaultdict(list)

    for fp in snowcover_fps:
        with h5py.File(fp, 'r') as f:
            # grab datetime from properties
            # based on this documentation. Section 1.2.3
            # https://nsidc.org/sites/default/files/documents/user-guide/multi_vnp10a1f-v002-userguide.pdf
            dt = pd.to_datetime(fp.stem.split('.')[1], format = 'A%Y%j')
            
            # get snow cover data
            data = f['HDFEOS']['GRIDS']['VIIRS_Grid_IMG_2D']['Data Fields']['Daily_NDSI_Snow_Cover'][:]

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
    snowcover_da = snowcover_da.rio.write_crs(src_crs).rio.reproject("EPSG:4326")
    
    return snowcover_da

# def earthaccess_property_url_mapping(earthaccess_results, snowcover_fps):
    """
    Unused function to generate properties for each earthaccess image and link to url
    """
#     stem_to_result = {Path(url.data_links()[0]).stem: url for url in earthaccess_results}

#     # Map local files to their original results
#     fp_to_properties_mapping = {fp: stem_to_result[fp.stem] for fp in snowcover_fps}

#     return fp_to_properties_mapping