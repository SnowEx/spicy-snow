"""
Functions to download 500 m VIRRS Snowcover.

https://nsidc.org/data/vj110a1f/versions/2
"""

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr

import earthaccess
import h5py

import logging
log = logging.getLogger(__name__)

# https://nsidc.org/data/vj110a1f/versions/2
results = earthaccess.search_data(
    short_name = "VJ110A1F",
    cloud_hosted = True,
    bounding_box = aoi.bounds,
    temporal = (start_date, stop_date),
)

sc_fps = earthaccess.download(results, local_path = snowcover_dir)

stem_to_result = {Path(url.data_links()[0]).stem: url for url in results}

# Map local files to their original results
mapping = {fp: stem_to_result[fp.stem] for fp in sc_fps}

# Organize tiles by date
tiles_by_date = defaultdict(list)

for fp in sc_fps:
    f = h5py.File(fp, 'r')
    props = mapping[fp].items()
    
    dt = pd.to_datetime(dict(props)['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime'])
    
    data = f['HDFEOS']['GRIDS']['VIIRS_Grid_IMG_2D']['Data Fields']['Daily_NDSI_Snow_Cover'][:]
    xdim = f['HDFEOS']['GRIDS']['VIIRS_Grid_IMG_2D']['XDim'][:]
    ydim = f['HDFEOS']['GRIDS']['VIIRS_Grid_IMG_2D']['YDim'][:]
    
    da_tile = xr.DataArray(
        data,
        coords={'y': ydim, 'x': xdim},
        dims=['y', 'x']
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
da_all = xr.concat(daily_arrays, dim='time').sortby('time')