"""
Functions to calculate delta CR, delta VV, delta gamma, and snow index.
"""

import xarray as xr
import numpy as np

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
            
    dataset['deltaVV'] = dataset.deltaVV.assign_attrs(
                        {'description':
                         'change in VV gamma-nought along time dimension, calculated according to eqn 2, Lievens et al., (2022)'})
    
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
    
    dataset['deltaCR'] = dataset.deltaCR.assign_attrs(
                        {'description':
                         'change in weighted combination of gamma-nought VH and gamma-nought VV along time dimension, calculated according to eqn 3, Lievens et al., (2022)'})
    
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
    
    dataset['deltaGamma'] = dataset.deltaGamma.assign_attrs(
                        {'description':
                         'change in S1 backscatter along time dimension, calculated according to eqn 4, Lievens et al., (2022)'})
    
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

def calc_snow_index(dataset: xr.Dataset, previous_snow_index: xr.DataArray) -> xr.Dataset:
    """
    Calculate snow index for each time step from previous time steps' snow index
    weights, and current delta-gamma.
    SI (i, t) = SI (i, t_previous) + delta-gamma (i, t)
    with SI (i, t_previous) as:
        SI (i, t_previous) = 1 / (sum())
    Args:
    dataset: Xarray Dataset of sentinel images with delta-gamma
    Returns:
    dataset: Xarray Dataset of sentinel images with snow-index added as band
    """

    # if August 1st - set snow index to 0 across image and continue

    # Identify previous image from the same relative orbit (6, 12, 18, or 24 days ago)

    # Calculate delta gamma from weighted average of +/- five images' SI 
    # and current delta-gamma

    # set snow-index to 0 for negative SIs and for IMS == 2 (land w/o snow-cover)

    # add snow-index as band to dataset

def calc_prev_snow_index(dataset: xr.Dataset, weights: np.array) -> xr.DataArray:
    """
    Calculate previous snow index for +/- 5 days (6 day timestep) or +/- 11 days 
    (12 day time step) from previous time step (6/12 days)'s snow index
    SI_prev (i, t_prev) = 1 / (sum (k = t - 5/11 days -> t + 5/11 days) weight (k)) \
        * sum (k = t - 5/11 days -> t + 5/11 days) weight(k) SI (i, k)
    
    with:
        w_k: as the inverse distance in time from t_previous so for 6-days: 
        wgts=repmat(win+1-abs([-win:win]),dim,1); [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]
        
    % Calculate average 
    si_pri(:,cnt)=nansum(wgts.*si_tmp,2)./nansum(wgts,2);
    Args:
    dataset: Xarray Dataset of sentinel images with delta-gamma
    Returns:
    DataArray: Xarray DataArray of previous snow-indexes weighted by time
    """

    # if August 1st - set snow index to 0 across image and continue

    # Identify previous image from the same relative orbit (6, 12, 18, or 24 days ago)

    # Calculate delta gamma from weighted average of +/- five images' SI 
    # and current delta-gamma

    # set snow-index to 0 for negative SIs and for IMS == 2 (land w/o snow-cover)

    # add snow-index as band to dataset

def snow_index_to_snow_depth(dataset: xr.Dataset, C: float = 0.44) -> xr.Dataset:
    """
    Convert current snow-index to snow depth using the C parameter. Varied 
    from [0->1 by 0.01].
    Args:
    dataset: Xarray Dataset of sentinel images with snow index
    C: fitting parameter
    Returns:
    dataset: Xarray Dataset of sentinel images with retrieved snow depth
    """

    pass