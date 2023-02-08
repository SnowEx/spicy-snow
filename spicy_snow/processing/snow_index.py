"""
Functions to calculate delta CR, delta VV, delta gamma, and snow index.
"""

import xarray

def calc_delta_cross_ratio(dataset: xr.Dataset) -> xr.Dataset:
    """
    Calculate change in cross-polarization ratio between current time step and previous.
    
    delta-gamma-cr (i, t) = gamma-cr (i, t) - gamma-cr (i, t_previous)

    with:
    gamma-cr = A * gamma-VH - gamma-VV

    and gamma-VH and gamma-VV in dB. Lieven's et al. 2021 tests A over [1, 2, 3]
    and fit to A = 2.

    Args:
    dataset: Xarray Dataset of sentinel images

    Returns:
    dataset: Xarray Dataset of sentinel images with delta-gamma-cr added as band
    """

    # Identify previous image from the same relative orbit (6, 12, 18, or 24 days ago)

    # subtract VH and VV for current and previous time step with A parameter

    # Calculate change in gamma-cr between previous and current time step

    # add delta-gamma-cr as band to dataset

def calc_delta_VV(dataset: xr.Dataset) -> xr.Dataset:
    """
    Calculate change in VV amplitude between current time step and previous.
    
    delta-gamma-VV (i, t) = gamma-VV (i, t) - gamma-VV(i, t_previous)

    Args:
    dataset: Xarray Dataset of sentinel images

    Returns:
    dataset: Xarray Dataset of sentinel images with delta-gamma-VV added as band
    """

    # Identify previous image from the same relative orbit (6, 12, 18, or 24 days ago)

    # Calculate change in gamma-VV between previous and current time step

    # add delta-gamma-VV as band to dataset

def calc_delta_gamma(dataset: xr.Dataset) -> xr.Dataset:
    """
    Calculate change in combined gamma parameter (VV and cross-ratio) between 
    current time step and previous.
    
    delta-gamma (i, t) = (1 - FCF) * delta-gamma-CR (i, t) + (FCF) * B * gamma-VV(i, t)

    with:
    FCF - being the PROBA-V fraction forest cover (ranging from 0 to 1).
    B - fitting parameter tested over 0 to 1 in steps of 0.1 and optimized at 0.5.

    Args:
    dataset: Xarray Dataset of sentinel images with delta-gamma-cr and delta-gamma-VV

    Returns:
    dataset: Xarray Dataset of sentinel images with delta-gamma added as band
    """

    # Calculate delta gamma from delta-gamma-cr, delta-gamma-VV and FCF

    # add delta-gamma as band to dataset

def calc_snow_index(dataset: xr.Dataset) -> xr.Dataset:
    """
    Calculate snow index for each time step from previous time step's snow index
    and current delta-gamma.

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

def calc_prev_snow_index(dataset: xr.Dataset) -> xr.DataArray:
    """
    Calculate previous snow index for +/- 5 days (6 day timestep) or +/- 11 days 
    (12 day time step) from previous time step (6/12 days)'s snow index

    SI_prev (i, t_prev) = 1 / (sum (t - 5/11 days, t + 5/11 days) weight (k)) \
        * sum (t - 5/11 days, t + 5/11 days) weight(k) SI (i, k)
    
    with:
        w_k: as the inverse distance in time from t_previous so 

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