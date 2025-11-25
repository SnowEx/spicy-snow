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

def da_to01(da: xr.DataArray, old_min=0, old_max=100) -> xr.DataArray:
    """
    Normalize an xarray DataArray from [old_min, old_max] to [0, 1].
    Values outside the old range are replaced with NaN.
    """
    da = da.astype(float)  # ensure float for NaNs

    # Mask values outside the old range
    da = da.where((da >= old_min) & (da <= old_max))

    # Normalize
    if old_max == old_min:
        raise ValueError("old_max and old_min cannot be equal")

    return (da - old_min) / (old_max - old_min)

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
    groups = (time_diff >= time_tol).cumsum(dim='time')

    # group images closer than time difference
    return da.groupby(groups).map(mosaic_group)
