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

def tif_to_dataarray(fp, mask = None, time = None, area = None, ref_da = None, spatial_resolution=None, chunks = 'auto'):
    """
    Open a single band, reproject to EPSG:4326, clip and pad to AOI, assign time stamp.

    If ref_da given it will try to reproject match those coordinates
    If mask given we set all nan pixels in mask to nans.
    """
    import rioxarray as rxa    
    img = xr.open_dataarray(fp, masked=True, chunks= chunks)[0]
    
    if mask is not None:
        img = img.where(~mask.isnull())

    if ref_da is not None:
        if spatial_resolution is not None:
            if isinstance(spatial_resolution, (int, float)):
                spatial_resolution = (spatial_resolution, spatial_resolution)
            img = img.rio.reproject(
                dst_crs='EPSG:4326',
                resolution=spatial_resolution
            )
        else:
            img = img.rio.reproject('EPSG:4326')

        img = img.rio.reproject_match(ref_da)
    else:
        img = img.rio.reproject('EPSG:4326')
        if area is not None:
            # clip and pad ensures we either clip or pad to match AOI
            img = img.rio.clip_box(*area.bounds)
            img = img.rio.pad_box(*area.bounds)

    if time is not None:
        dt = pd.to_datetime(time)
        img = img.expand_dims(time = [dt])

    return img

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
