"""
Functions to calculate delta CR, delta VV, delta gamma, and snow index.
"""

import numpy as np
import pandas as pd
import xarray as xr

def calc_delta_VV(dataset: xr.Dataset, inplace: bool = False) -> xr.Dataset:
    """
    Calculate change in VV amplitude between current time step and previous
    from each relative orbit and adds to dataset.
    
    delta-gamma-VV (i, t) = gamma-VV (i, t) - gamma-VV(i, t_previous)

    Returns nans in time slice of deltaVV with no previous image 
    (first of each relative orbit)

    Args:
    dataset: Xarray Dataset of sentinel images
    inplace: operate on dataset in place or return copy

    Returns:
    dataset: Xarray Dataset of sentinel images with 'deltaVV' added as data var
    """
    # check inplace flag
    if not inplace:
        dataset = dataset.copy(deep=True)

    # check for amp
    if 's1_units' in dataset.attrs.keys():
        assert dataset.attrs['s1_units'] == 'dB', 'Sentinel-1 units must be in dB'

    # get all unique relative orbits
    orbits = np.unique(dataset['relative_orbit'].values)

    for orbit in orbits:
        
        # Calculate change in gamma-VV between previous and current time step from the same relative orbit
        diffVV = dataset['s1'].sel(time = dataset.relative_orbit == orbit, band = 'VV').diff(dim = 'time')
        
        # if adding new 
        if 'deltaVV' not in dataset.data_vars:
            # add delta-gamma-VV as variable to dataset
            dataset['deltaVV'] = diffVV
        
        else:
            # update deltaVV times with this. 
            dataset['deltaVV'].loc[dict(time = diffVV.time)] = diffVV
    
    if not inplace:
        return dataset

def calc_delta_cross_ratio(dataset: xr.Dataset, A: float = 2, inplace: bool = False) -> xr.Dataset:
    """
    Calculate change in cross-polarization ratio for all time steps.
    
    delta-gamma-cr (i, t) = gamma-cr (i, t) - gamma-cr (i, t_previous)

    with:
    gamma-cr = A * VH - VV

    and gamma-VH and gamma-VV in dB. Lieven's et al. 2021 tests A over [1, 2, 3]
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

    # check inplace flag
    if not inplace:
        dataset = dataset.copy(deep=True)

    # check for amp
    if 's1_units' in dataset.attrs.keys():
        assert dataset.attrs['s1_units'] == 'dB', 'Sentinel-1 units must be in dB'

    # calculate cross ratio of VH to VV with fitting parameter A
    gamma_cr = (A * dataset['s1'].sel(band='VH')) - dataset['s1'].sel(band='VV')

    # get all unique relative orbits
    orbits = np.unique(dataset['relative_orbit'].values)
    
    # Identify previous image from the same relative orbit (6, 12, 18, or 24 days ago)
    for orbit in orbits:

        # Calculate change in gamma-cr between previous and current time step
        diffCR = gamma_cr.sel(time = dataset.relative_orbit == orbit).diff(dim = 'time')
        
        # add delta-gamma-cr as band to dataset
        # if adding new 
        if 'deltaCR' not in dataset.data_vars:
            # add delta-gamma-CR as new variable to dataset
            dataset['deltaCR'] = diffCR
        
        else:
            # update deltaCR times with this. 
            dataset['deltaCR'].loc[dict(time = diffCR.time)] = diffCR
    
    if not inplace:
        return dataset

def calc_delta_gamma(dataset: xr.Dataset, B: float = 0.5, inplace: bool = False) -> xr.Dataset:
    """
    Calculate change in combined gamma parameter (VV and cross-ratio) between 
    current time step and previous.
    
    delta-gamma(i, t) = (1 - FCF) * delta-gamma-CR (i, t) + (FCF) * B * gamma-VV(i, t)

    with:
    FCF - being the PROBA-V fraction forest cover (ranging from 0 to 1).
    B - fitting parameter tested over 0 to 1 in steps of 0.1 and optimized at 0.5.

    Args:
    dataset: Xarray Dataset of sentinel images with delta-gamma-cr and delta-gamma-VV

    Returns:
    dataset: Xarray Dataset of sentinel images with delta-gamma added as band
    """
    # check inplace flag
    if not inplace:
        dataset = dataset.copy(deep=True)

    # Calculate delta gamma from delta-gamma-cr, delta-gamma-VV and FCF
    # add delta-gamma as band to dataset
    dataset['deltaGamma'] = (1 - dataset['fcf']) * dataset['deltaCR'] + \
        (dataset['fcf'] * B * dataset['deltaVV'])

    if not inplace:
        return dataset

def clip_delta_gamma_outlier(dataset: xr.Dataset, thresh: float = 3, inplace: bool = False) -> xr.Dataset: 
    """
    Clip delta gamma to -3 -> 3 dB

    Args:
    dataset: Xarray Dataset of sentinel images with delta-gamma

    Returns:
    dataset: Xarray Dataset of sentinel images with delta gamma changes clipped
    to -3 -> 3
    """
    # check inplace flag
    if not inplace:
        dataset = dataset.copy(deep=True)

    # change values above 3 and not nan to 3
    dataset['deltaGamma'] = dataset['deltaGamma'].where((dataset['deltaGamma'] < thresh) | dataset['deltaGamma'].isnull(), thresh)
    
    # change values below -3 and not nan to -3
    dataset['deltaGamma'] = dataset['deltaGamma'].where((dataset['deltaGamma'] > -thresh) | dataset['deltaGamma'].isnull(), -thresh)
    
    if not inplace:
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
    orbit_times = dataset.sel(time = dataset.relative_orbit == dataset['relative_orbit'][0]).time.diff(dim = 'time').values
    repeat = np.nanmedian([pd.Timedelta(i).round('D') for i in orbit_times]).round('D')

    assert repeat.days % 6 == 0, "Calculated repeat interval is not multiple of 6 days."

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
    # calculate weights based on days between centered date and image acquistions
    wts = repeat.days - np.abs([int((t - t_prev).days) for t in prev.time.values])
    # calculate previous snow index weighted by time from last acquistion
    prev_si = (prev['snow_index']*wts).sum(dim = 'time')/np.sum(wts)
    # this is weird but works
    # (prev['snow_index']*wts).sum(dim = 'time') treats nans as 0 so they don't
    # propogate through time series (maybe a problem later for wet snow masking)

    return prev_si

def calc_snow_index(dataset: xr.Dataset, inplace: bool = False) -> xr.Dataset:
    """
    Calculate snow index for each time step from previous time steps' snow index
    weights, and current delta-gamma.

    SI (i, t) = SI (i, t_previous) + delta-gamma (i, t)

    with SI (i, t_previous) as:
        SI (i, t_previous) = sum (t_pri - 5/11 days, t_pri + 5/11 days)(SI * weights) / sum(weights)

    Args:
    dataset: Xarray Dataset of sentinel images with delta-gamma
    inplace: operate on dataset in place or return copy

    Returns:
    dataset: Xarray Dataset of sentinel images with snow-index added as band
    """
    # check inplace flag
    if not inplace:
        dataset = dataset.copy(deep=True)

    # set all snow index to 0 to start
    dataset['snow_index'] = xr.zeros_like(dataset['deltaGamma'])

    # find repeat interval of dataset
    repeat = find_repeat_interval(dataset)

    # iterate through time steps
    for ct in dataset.time.values:
        # calculate previous snow index
        prev_si = calc_prev_snow_index(dataset, ct, repeat)
        # add deltaGamma to previous snow inded
        dataset['snow_index'].loc[dict(time = ct)] = prev_si + dataset['deltaGamma'].sel(time = ct)
    
    if not inplace:
        return dataset


def calc_snow_index_to_snow_depth(dataset: xr.Dataset, C: float = 0.44, inplace: bool = False) -> xr.Dataset:
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
    # check inplace flag
    if not inplace:
        dataset = dataset.copy(deep=True)
    
    dataset['snow_depth'] = dataset['snow_index'] * C

    if not inplace:
        return dataset