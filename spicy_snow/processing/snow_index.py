"""
Functions to calculate delta CR, delta VV, delta gamma, and snow index.
"""

import xarray as xr
import numpy as np

def add_band(dataset: xr.Dataset, dataArray: xr.DataArray, band_name: str, inplace: bool = False) -> xr.Dataset:
    """
    Add band to sentinel 1 datavariable in Dataset. Can have multiple time steps
    and already exists.

    Args:
    dataset: Dataset containing sentinel-1 images
    dataArray: dataArray to add to dataset
    band_name: name for band, can already be in dataset
    inplace: operate on dataset in place or return copy

    Returns:
    dataset: dataset with dataArray add to it at the dataArray's timeslice with
    that band name.
    """
    # check inplace flag
    if not inplace:
        dataset = dataset.copy(deep=True)

    # remove DataArrays coordinates so they don't concatenated only band dimension
    dataArray = dataArray.reset_coords(names = ['flight_dir', 'platform', 'relative_orbit'], drop = True)
    
    # rename dataArrays band
    dataArray['band'] = band_name

    # if this is the first time adding this band
    if band_name not in dataset.band:
        # combine the 's1' dataArray and your new data on band
        com_da = xr.concat([dataset['s1'], dataArray], dim = 'band')

        # remove s1 from data variables to remerge
        vars = list(dataset.data_vars)
        vars.remove('s1')

        # merge dataset sans sentinel1 with sentinel dataArray with new band
        dataset = xr.merge([dataset[vars], com_da])
    else:
        # if band already in dataset just slice into data array time slices and add data
        dataset['s1'].loc[dict(band = band_name, time = dataArray.time)] = dataArray

    if not inplace:
        return dataset

def calc_delta_VV(dataset: xr.Dataset, inplace: bool = False) -> xr.Dataset:
    """
    Calculate change in VV amplitude between current time step and previous
    from each relative orbit.
    
    delta-gamma-VV (i, t) = gamma-VV (i, t) - gamma-VV(i, t_previous)

    Returns nans in band deltaVV for time slices with no previous image 
    (first of each relative orbit)

    Args:
    dataset: Xarray Dataset of sentinel images
    inplace: operate on dataset in place or return copy

    Returns:
    dataset: Xarray Dataset of sentinel images with 'deltaVV' added as band
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
        diff = dataset['s1'].sel(time = dataset.relative_orbit == orbit, band = 'VV').diff(dim = 'time')
        
        # add delta-gamma-VV as band to dataset
        dataset = add_band(dataset, diff, 'deltaVV')
    
    if not inplace:
        return dataset

def calc_delta_cross_ratio(dataset: xr.Dataset, A: float = 2) -> xr.Dataset:
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