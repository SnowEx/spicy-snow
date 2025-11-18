"""
Functions to calculate delta CR, delta vv, delta gamma, and snow index.
"""

import numpy as np
import pandas as pd
import xarray as xr

from typing import Union

import logging
log = logging.getLogger(__name__)

def calc_delta_vv(dataset: xr.Dataset) -> Union[None, xr.Dataset]:
    """
    Calculate change in vv amplitude between current time step and previous
    from each relative orbit and adds to dataset.
    
    delta-gamma-vv (i, t) = gamma-vv (i, t) - gamma-vv(i, t_previous)

    Returns nans in time slice of deltavv with no previous image 
    (first of each relative orbit)

    Args:
    dataset: Xarray Dataset of sentinel images
    inplace: operate on dataset in place or return copy

    Returns:
    dataset: Xarray Dataset of sentinel images with 'deltavv' added as data var
    """

    # check for amp
    assert dataset.attrs.get('s1_units') == 'dB', 'Sentinel-1 units must be in dB'

    # get all unique relative orbits
    orbits = np.unique(dataset['track'].values)

    for orbit in orbits:
        
        # Calculate change in gamma-vv between previous and current time step from the same relative orbit
        diffvv = dataset['vv'].sel(time = dataset.track == orbit).diff(dim = 'time')
        
        # if adding new 
        if 'deltavv' not in dataset.data_vars:
            # add delta-gamma-vv as variable to dataset
            dataset['deltavv'] = diffvv
        
        else:
            # update deltavv times with this. 
            dataset['deltavv'].loc[dict(time = diffvv.time)] = diffvv
    
    return dataset

def calc_delta_cross_ratio(dataset: xr.Dataset, A: float = 2) -> Union[None, xr.Dataset]:
    """
    Calculate change in cross-polarization ratio for all time steps.
    
    delta-gamma-cr (i, t) = gamma-cr (i, t) - gamma-cr (i, t_previous)

    with:
    gamma-cr = A * vh - vv

    and gamma-vh and gamma-vv in dB. Lieven's et al. 2021 tests A over [1, 2, 3]
    and fit to A = 2.

    Returns nans in time slice of deltaCR with no previous image 
    (first of each relative orbit)

    Args:
    dataset: Xarray Dataset of sentinel images
    A: fitting parameter [default = 2]
    inplace: operate on dataset in place or return copy

    Returns:
    dataset: Xarray Dataset of sentinel images with deltaCR added as data var
    """

    # check for amp
    assert dataset.attrs.get('s1_units') == 'dB', 'Sentinel-1 units must be in dB'

    # calculate cross ratio of vh to vv with fitting parameter A
    # in dB so subtraction is actually a ratio in amplitude
    gamma_cr = (A * dataset['vh']) - dataset['vv']

    # get all unique relative orbits
    orbits = np.unique(dataset['track'].values)
    
    # Identify previous image from the same relative orbit (6, 12, 18, or 24 days ago)
    for orbit in orbits:

        # Calculate change in gamma-cr between previous and current time step
        diffCR = gamma_cr.sel(time = dataset.track == orbit).diff(dim = 'time')
        
        # add delta-gamma-cr as band to dataset
        # if adding new 
        if 'deltaCR' not in dataset.data_vars:
            # add delta-gamma-CR as new variable to dataset
            dataset['deltaCR'] = diffCR
        
        else:
            # update deltaCR times with this. 
            dataset['deltaCR'].loc[dict(time = diffCR.time)] = diffCR
    
    return dataset

def calc_delta_gamma(dataset: xr.Dataset, B: float = 0.5) -> Union[None, xr.Dataset]:
    """
    Calculate change in combined gamma parameter (vv and cross-ratio) between 
    current time step and previous.
    
    delta-gamma(i, t) = (1 - FCF) * delta-gamma-CR (i, t) + (FCF) * B * gamma-vv(i, t)

    with:
    FCF - being the PROBA-V fraction forest cover (ranging from 0 to 1).
    B - fitting parameter tested over 0 to 1 in steps of 0.1 and optimized at 0.5.

    Args:
    dataset: Xarray Dataset of sentinel images with delta-gamma-cr and delta-gamma-vv

    Returns:
    dataset: Xarray Dataset of sentinel images with delta-gamma added as band
    """
    # check to ensure fcf is 0-1 not 0-100
    assert dataset['fcf'].max() <= 1, "Forest cover fraction must be scaled 0-1"
    assert dataset['fcf'].min() >= 0, "Forest cover fraction must be scaled 0-1"

    # Calculate delta gamma from delta-gamma-cr, delta-gamma-vv and FCF
    # add delta-gamma as band to dataset
    dataset['deltaGamma'] = (1 - dataset['fcf']) * dataset['deltaCR'] + \
        (dataset['fcf'] * B * dataset['deltavv'])

    return dataset

def clip_delta_gamma_outlier(dataset: xr.Dataset, thresh: float = 3) -> Union[None, xr.Dataset]: 
    """
    Clip delta gamma to -3 -> 3 dB

    Args:
    dataset: Xarray Dataset of sentinel images with delta-gamma
    thresh: threshold in dB to clip delta Gamma to

    Returns:
    dataset: Xarray Dataset of sentinel images with delta gamma changes clipped
    to -3 -> 3
    """
    # change values above 3 and not nan to 3
    dataset['deltaGamma'] = dataset['deltaGamma'].where((dataset['deltaGamma'] < thresh) | dataset['deltaGamma'].isnull(), thresh)
    
    # change values below -3 and not nan to -3
    dataset['deltaGamma'] = dataset['deltaGamma'].where((dataset['deltaGamma'] > -thresh) | dataset['deltaGamma'].isnull(), -thresh)
    
    return dataset

def find_repeat_interval(dataset: xr.Dataset) -> pd.Timedelta:
    """
    Figures out if datasets repeat interval is 6 days or 12 days. Should raise error
    if not multuple of 6 days.

    Args:
    dataset: dataset of sentinel-1 images

    Returns:
    repeat: pandas timedelta and number of days between images
    """
    # figure out if 6 or 12 days repeat
    orbit_times = dataset.sel(time = dataset.track == dataset['track'][0]).time.diff(dim = 'time').values
    repeat = np.nanmedian([pd.Timedelta(i).round('D') for i in orbit_times]).round('D')

    assert repeat.days % 6 == 0, f"Calculated repeat interval, {repeat}, is not multiple of 6 days."

    return repeat

def calc_prev_snow_index(dataset: xr.Dataset, current_time: np.datetime64, repeat: pd.Timedelta) -> xr.DataArray:
    """
    Calculate previous snow index for +/- 5 days (6 day timestep) or +/- 11 days 
    (12 day time step) from previous time step (6/12 days)'s snow index

    SI (i, t_previous) = sum (t_pri - 5/11 days, t_pri + 5/11 days)(SI * weights) / sum(weights)

    with:
        w_k: as the inverse distance in time from t_previous so for 6-days: 
        wgts=repmat(win+1-abs([-win:win]),dim,1); [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]

    Args:
    dataset: dataset of sentinel-1 images with 'snow-index' data variable
    current_time: the current image date
    repeat: is this region capturing s1 images every 6 or 12 days

    Returns:
    prev_si: the weighted average of previous snow indexes
    """
    # calculate how many days ago we are centering previous snow indexes (6 or 12 days)
    t_prev = current_time - repeat
    # get slice of +- 5 or +- 11 days depending on repeat interval
    t_oldest, t_youngest = pd.to_datetime(t_prev - (repeat - pd.Timedelta('1 day'))) , pd.to_datetime(t_prev + (repeat - pd.Timedelta('1 day')))
    # slice dataset to get all images in previous period
    prev = dataset.sel(time = slice(t_oldest, t_youngest))

    # TODO
    # check this edge case of no previous data found.
    if prev.sizes['time'] == 0:
        prev_si = xr.zeros_like(dataset['snow_index'].isel(time=0))
        return prev_si

    # calculate weights based on days between centered date and image acquistions
    day_weights = repeat.days - np.abs([int((t - t_prev).days) for t in prev.time.values])
    wts = xr.ones_like(prev['snow_index']) * day_weights

    # set wts to nan where snow_index is nan
    wts = wts.where(~prev['snow_index'].isnull())

    # calculate previous snow index weighted by time from last acquistion
    prev_si = (prev['snow_index'] * wts).sum(dim = 'time', skipna = True) / wts.sum(dim = 'time', skipna = True)
    # this is weird but works
    # (prev['snow_index']*wts).sum(dim = 'time') treats nans as 0 so they don't
    # propogate through time series (maybe a problem later for wet snow masking)

    return prev_si

def calc_snow_index(dataset: xr.Dataset, ims_masking: bool = True) -> Union[None, xr.Dataset]:
    """
    Calculate snow index for each time step from previous time steps' snow index
    weights, and current delta-gamma.

    SI (i, t) = SI (i, t_previous) + delta-gamma (i, t)

    with SI (i, t_previous) as:
        SI (i, t_previous) = sum (t_pri - 5/11 days, t_pri + 5/11 days)(SI * weights) / sum(weights)

    Args:
    dataset: Xarray Dataset of sentinel images with delta-gamma
    ims_masking: whether to mask pixels with the IMS data
    inplace: operate on dataset in place or return copy

    Returns:
    dataset: Xarray Dataset of sentinel images with snow-index added as band
    """

    # set all snow index to 0 to start
    dataset['snow_index'] = xr.zeros_like(dataset['deltaGamma'])

    # find repeat interval of dataset
    repeat = find_repeat_interval(dataset)

    # iterate through time steps
    for ct in dataset.time.values:

        # calculate previous snow index
        prev_si = calc_prev_snow_index(dataset, ct, repeat)

        # set prev_si to 0 at nans
        prev_si = prev_si.where(~prev_si.isnull(), 0)
        
        # add deltaGamma to previous snow index
        dataset['snow_index'].loc[dict(time = ct)] = prev_si + dataset['deltaGamma'].sel(time = ct)
        
        # change to 0 when ims snow cover is not 4
        if ims_masking:
            dataset['snow_index'].loc[dict(time = ct)] = dataset['snow_index'].sel(time = ct).where(dataset['snowcover'].sel(time = ct) == True, 0)

        # change to 0 when snow_index is negative
        dataset['snow_index'].loc[dict(time = ct)] = \
            dataset['snow_index'].sel(time = ct).where((dataset['snow_index'].sel(time = ct).isnull()) | (dataset['snow_index'].sel(time = ct) > 0), 0)
    
    return dataset


def calc_snow_index_to_snow_depth(dataset: xr.Dataset, C: float = 0.44) -> Union[None, xr.Dataset]:
    """
    Convert current snow-index to snow depth using the C parameter. Varied 
    from [0->1 by 0.01].

    Args:
    dataset: Xarray Dataset of sentinel images with snow index
    C: fitting parameter
    inplace: operate on dataset in place or return copy

    Returns:
    dataset: Xarray Dataset of sentinel images with retrieved snow depth
    """
    
    dataset['snow_depth'] = dataset['snow_index'] * C

    return dataset