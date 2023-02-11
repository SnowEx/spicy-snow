"""
Functions to calculate delta CR, delta VV, delta gamma, and snow index.
"""

import xarray as xr
import numpy as np

def calc_delta_cross_ratio(dataset: xr.Dataset, a: float = 2) -> xr.Dataset:
    """
    Calculate change in cross-polarization ratio between current time step and previous.
    
    delta-gamma-cr (i, t) = gamma-cr (i, t) - gamma-cr (i, t_previous)

    with:
    gamma-cr = A * gamma-VH - gamma-VV

    and gamma-VH and gamma-VV in dB. Lieven's et al. 2021 tests A over [1, 2, 3]
    and fit to A = 2.

    Args:
    dataset: Xarray Dataset of sentinel images
    a: fitting parameter [default = 2]

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

def calc_delta_gamma(dataset: xr.Dataset, b: float = 0.5) -> xr.Dataset:
    """
    Calculate change in combined gamma parameter (VV and cross-ratio) between 
    current time step and previous.
    
    delta-gamma(i, t) = (1 - FCF) * delta-gamma-CR (i, t) + (FCF) * B * gamma-VV(i, t)

    with:
    FCF - being the PROBA-V fraction forest cover (ranging from 0 to 1).
    b - fitting parameter tested over 0 to 1 in steps of 0.1 and optimized at 0.5.

    Args:
    dataset: Xarray Dataset of sentinel images with delta-gamma-cr and delta-gamma-VV

    Returns:
    dataset: Xarray Dataset of sentinel images with delta-gamma added as band
    """

    # Calculate delta gamma from delta-gamma-cr, delta-gamma-VV and FCF

    # add delta-gamma as band to dataset

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

def calc_timesteps_weights(dataset: xr.Dataset) -> np.array:
    """
    Calculate weight for snow index effects of surrounding snow indexes

    Matlab code: 
    # make initial weights
    wgts=repmat(win+1-abs([-win:win]),dim,1); [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]
    # set no weights to steps steps without obs
    wgts(isnan(si_tmp))=NaN;
    # Reduce weight (e.g. by 3) for time steps with wet snow
    wgts(wet_tmp==1)=wgts(wet_tmp==1)./3;

    Args:
    dataset: Xarray Dataset of sentinel images with snow-index

    Returns:
    weights: weights for 5, 11 days
    """
    pass

def calc_prev_snow_index(dataset: xr.Dataset, weights: np.array) -> xr.DataArray:
    """
    Calculate previous snow index for +/- 5 days (6 day timestep) or +/- 11 days 
    (12 day time step) from previous time step (6/12 days)'s snow index

    SI_prev (i, t_prev) = 1 / (sum (k = t - 5/11 days -> t + 5/11 days) weight (k)) \
        * sum (k = t - 5/11 days -> t + 5/11 days) weight(k) SI (i, k)
    
    with:
        w_k: as the inverse distance in time from t_previous so for 6-days: 
        
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

def reduce_snow_index_snowfree(dataset: xr.Dataset) -> xr.Dataset: 
    """
    Reduce increment if prior date was still snowfree (to avoid jumps in shallow snow)
    
    # change in cross-pol and VV is reduced by 0.5 when past IMS snow cover is snow-free
    dra_u(sc_pri==0)=0.5.*dra_u(sc_pri==0);
    dvv_u(sc_pri==0)=0.5.*dvv_u(sc_pri==0);
    
    Args:
    dataset: Xarray Dataset of sentinel images with delta-gamma

    Returns:
    dataset: Xarray Dataset of sentinel images with 2.5 +  reduced
    for snow-free times
    """
    pass

def reduce_snow_index_big_decreases(dataset: xr.Dataset) -> xr.Dataset: 
    """
    Reduce impact of strong decreases in backscatter 
    (e.g. caused by rain on snow, or wet snow)
    
    # change in cross-pol and VV when change is decreasing more than 2.5
    # if d_vv or d_cross < -2.5 then d_XX = -2.5 + 0.5 * (2.5 + d_XX)
    dra_u(dra_u<-2.5)=-2.5+0.5.*(2.5+dra_u(dra_u<-2.5));  
    dvv_u(dvv_u<-2.5)=-2.5+0.5.*(2.5+dvv_u(dvv_u<-2.5));  

    Args:
    dataset: Xarray Dataset of sentinel images with delta-gamma

    Returns:
    dataset: Xarray Dataset of sentinel images with big decreases mitigated
    """

    pass

def reduce_snow_index_big_increases(dataset: xr.Dataset) -> xr.Dataset: 
    """
    Reduce impact of strong increases in backscatter 
    (e.g. caused by (re)frost, or wet snow roughness)
    
    # change in cross-pol and VV when change is decreasing more than 2.5
    # if d_vv or d_cross > 2.5 then d_XX = 2.5 + 0.5 * ( d_XX - 2.5)
    dra_u(dra_u>2.5)=2.5+0.5.*(dra_u(dra_u>2.5)-2.5);
    dvv_u(dvv_u>2.5)=2.5+0.5.*(dvv_u(dvv_u>2.5)-2.5);

    Args:
    dataset: Xarray Dataset of sentinel images with delta-gamma

    Returns:
    dataset: Xarray Dataset of sentinel images with big increases mitigated
    """

    pass

def sentinel_1_clip_outliers(dataset: xr.Dataset) -> xr.Dataset: 
    """
    Clip changes to -3 -> 3 dB

    dra_u(dra_u>3)=3; dra_u(dra_u<-3)=-3; 
    dvv_u(dvv_u>3)=3; dvv_u(dvv_u<-3)=-3;

    Args:
    dataset: Xarray Dataset of sentinel images with delta-gamma

    Returns:
    dataset: Xarray Dataset of sentinel images with backscatter changes clipped
    to -3 -> 3
    """

    pass

def snow_index_to_snow_depth(dataset: xr.Dataset, c: float = 0.44) -> xr.Dataset:
    """
    Convert current snow-index to snow depth using the C parameter. Varied 
    from [0->1 by 0.01].

    Args:
    dataset: Xarray Dataset of sentinel images with snow index
    c: fitting parameter

    Returns:
    dataset: Xarray Dataset of sentinel images with retrieved snow depth
    """

    pass