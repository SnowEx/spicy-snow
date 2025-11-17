"""
Raster processing utilities.
"""

from functools import reduce

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxa

import logging
log = logging.getLogger(__name__)

def to01(array, max = 100, min = 0):

    log.debug("Making {array} to 0-1 range.")
    array[array > max] = np.nan
    array[array < min] = np.nan

    # ignore the Runtime Warning
    with np.errstate(divide='ignore'):
        b = 1. /(max - min)

    if not(np.isfinite(b)):
        b = 0
        
    return np.vectorize(lambda x: b * (x - min))(array)

def tif_to_dataarray(fp, time = None, area = None, ref_da = None):
    """
    Open a single band, reproject to EPSG:4326, clip and pad to AOI, assign time stamp.

    If ref_da given it will try to reproject match those coordinates
    """
    import rioxarray as rxa
    name = fp.stem
    
    img = xr.open_dataarray(fp, masked=True)[0]
    
    if ref_da is None:
        img = img.rio.reproject('EPSG:4326')
        if area is not None:
            img = img.rio.clip_box(*area.bounds)
            img = img.rio.pad_box(*area.bounds)
    else:
        img = img.rio.reproject_match(ref_da)

    if time is not None:
        dt = pd.to_datetime(time)
    return img.expand_dims(time = [dt])

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
