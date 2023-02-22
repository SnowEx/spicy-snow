"""
Functions to identify, mask, and create weights for wet-snow.
"""

import numpy as np
import xarray as xr
from typing import Union

def id_newly_wet_snow(dataset: xr.Dataset, wet_thresh: int = -2, inplace: bool = False) -> Union[None, xr.Dataset]:
    """
    Identifies time steps with newly wet snow. Identifies time slices where
    deltaVV decreases by 2dB in pixels with FCF > 0.5 and deltaCR decreases by
    2dB in pixels with FCF < 0.5.

    Args:
    dataset: xarray dataset with deltaVV and deltaCR as data vars
    wet_thresh: decrease in dB of deltaVV or deltaCR to identify melting
    inplace: return copy of dataset or operate on dataset inplace?

    Returns:
    dataset: xarray data with melting data var
    """
    # check inplace flag
    if not inplace:
        dataset = dataset.copy(deep=True)
    
    # check we have the neccessary variables
    necessary_vars = set(['fcf', 'deltaCR', 'deltaVV'])
    assert necessary_vars.issubset(set(dataset.data_vars)),\
          f"Missing variables {necessary_vars.difference(set(dataset.data_vars))}"
    
    # add wet_flag to dataset if not already present    
    if 'wet_flag' not in dataset.data_vars:
        dataset['wet_flag'] = xr.zeros_like(dataset['deltaVV'])

    # identify possible newly wet snow in regions FCF < 0.5 with deltaCR
    dataset['wet_flag'] = dataset['wet_flag'].where(((dataset['fcf'] > 0.5) | (dataset['deltaCR'] > wet_thresh)), 1)
    # identify possible newly wet snow in regions FCF > 0.5 with deltaVV
    dataset['wet_flag'] = dataset['wet_flag'].where(((dataset['fcf'] < 0.5) | (dataset['deltaVV'] > wet_thresh)), 1)

    if not inplace:
        return dataset

def id_newly_frozen_snow(dataset: xr.Dataset, freeze_thresh: int = 2, inplace: bool = False) -> Union[None, xr.Dataset]:
    """
    Identifies time steps with probable re-frozen snow. Identifies time slices where
    deltaGammaNaught increases by 2 dB

    Args:
    dataset: xarray dataset with deltaGammaNaught as data var
    freeze_thresh: increase in dB of deltaGammaNaught to identify refreeze
    inplace: return copy of dataset or operate on dataset inplace?

    Returns:
    dataset: xarray data with freezing data var
    """
    # check inplace flag
    if not inplace:
        dataset = dataset.copy(deep=True)

    # check we have the neccessary variables
    necessary_vars = set(['deltaGamma'])
    assert necessary_vars.issubset(set(dataset.data_vars)),\
          f"Missing variables {necessary_vars.difference(set(dataset.data_vars))}"

    # add wet_flag to dataset if not already present    
    if 'freeze_flag' not in dataset.data_vars:
        dataset['freeze_flag'] = xr.zeros_like(dataset['deltaVV'])

    # identify possible re-freezing by increases of deltaGammaNaught of 2dB
    dataset['freeze_flag'] = dataset['freeze_flag'].where(dataset['deltaGamma'] < freeze_thresh, 1)

    if not inplace:
        return dataset

def id_wet_negative_si(dataset: xr.Dataset, inplace: bool = False) -> Union[None, xr.Dataset]:
    """
    Additional wet snow criteria if sd retrieval (snow-index since they are linear)
    becomes negative with snow cover is present we set pixel to wet.

    Args:
    dataset: xarray dataset with snow_index as data vars
    inplace: return copy of dataset or operate on dataset inplace?

    Returns:
    dataset: xarray data with wet_snow data var
    """
    # check inplace flag
    if not inplace:
        dataset = dataset.copy(deep=True)

    # check we have the neccessary variables
    necessary_vars = set(['snow_index', 'ims'])
    assert necessary_vars.issubset(set(dataset.data_vars)),\
          f"Missing variables {necessary_vars.difference(set(dataset.data_vars))}"

    # add alt_wet_flag to dataset if not already present    
    if 'alt_wet_flag' not in dataset.data_vars:
        dataset['alt_wet_flag'] = xr.zeros_like(dataset['deltaVV'])

    # identify wetting of snow by negative snow index with snow present
    dataset['alt_wet_flag'] = dataset['alt_wet_flag'].where(((dataset['ims'] != 4) | (dataset['snow_index'] > 0)), 1)

    if not inplace:
        return dataset

def flag_wet_snow(dataset: xr.Dataset, inplace: bool = False) -> Union[None, xr.Dataset]:
    """
    Identifies time steps with wet snow. Sets all time slices, for a relative orbit,
    as dry until the first melting then sets all time steps, for that relative orbit,
    as wet until either end of time series or next re-freezing.

    If re-freezing revert to set all as dry until next melt. Repeat to end of time series.
    
    If after Feburary 1st until August 1st if last two of four relative orbits
    were classified as wet we set remained to wet to stop retrievals

    Args:
    dataset: xarray dataset with melting, freezing as data vars
    inplace: return copy of dataset or operate on dataset inplace?

    Returns:
    dataset: xarray data with wet_snow data var
    """
    # check inplace flag
    if not inplace:
        dataset = dataset.copy(deep=True)

    # check we have the neccessary variables
    necessary_vars = set(['wet_flag', 'alt_wet_flag', 'freeze_flag'])
    assert necessary_vars.issubset(set(dataset.data_vars)),\
          f"Missing variables {necessary_vars.difference(set(dataset.data_vars))}"
    
    # create new data variable wet_snow and set to 0
    dataset['wet_snow'] = xr.zeros_like(dataset['wet_flag'])
    dataset['perma_wet'] = xr.zeros_like(dataset['wet_flag'])

    # get all unique relative orbits
    orbits = np.unique(dataset['relative_orbit'].values)

    # Identify previous image from the same relative orbit (6, 12, 18, or 24 days ago)
    for orbit in orbits:
        # select just that orbit to do wet snow propogation
        orbit_dataset = dataset.sel(time = dataset.relative_orbit == orbit)

        prev_time = None

        # loop through time steps in this orbit
        for ts in orbit_dataset.time:

            # if not first round then pull in previous time step's values for wet-snow
            if prev_time:
                dataset['wet_snow'].loc[dict(time = ts)] = dataset.sel(time = prev_time)['wet_snow']

            # add newly wet snow flags to old wet snow and then bound at 1
            dataset['wet_snow'].loc[dict(time = ts)]= dataset.sel(time = ts)['wet_snow'] + dataset.sel(time = ts)['wet_flag']
            dataset['wet_snow'].loc[dict(time = ts)] = dataset.sel(time = ts)['wet_snow'] + dataset.sel(time = ts)['alt_wet_flag']
            dataset['wet_snow'].loc[dict(time = ts)] = dataset.sel(time = ts)['wet_snow'].where(dataset.sel(time = ts)['wet_snow'] < 1, 1)
            # add newly frozen snow flags to old wet snow and then bound at 0 to avoid negatives
            dataset['wet_snow'].loc[dict(time = ts)] = dataset.sel(time = ts)['wet_snow'] - dataset.sel(time = ts)['freeze_flag']
            dataset['wet_snow'].loc[dict(time = ts)] = dataset.sel(time = ts)['wet_snow'].where(dataset.sel(time = ts)['wet_snow'] > 0, 0)

            prev_time = ts

        # if >50% wet of last 4 cycles after feb 1 then set remainer till
        # august 1st to perma-wet(2?)
    
        melt_season = (dataset['time.month'] > 1) & (dataset['time.month'] < 8)
        melt_orbit = (melt_season & (dataset.relative_orbit == orbit))

        # check if there are at least 4 time slices in melt season for this orbit
        
        if len(dataset['perma_wet'].loc[dict(time = melt_orbit)]) > 4:
            dataset['perma_wet'].loc[dict(time = melt_orbit)] = \
                dataset['wet_flag'].loc[dict(time = melt_orbit)]
            
            dataset['perma_wet'].loc[dict(time = melt_orbit)] = dataset['perma_wet'].loc[dict(time = melt_orbit)] + \
                dataset['alt_wet_flag'].loc[dict(time = melt_orbit)]
            
            dataset['perma_wet'].loc[dict(time = melt_orbit)] = \
                dataset['perma_wet'].loc[dict(time = melt_orbit)].where(dataset['perma_wet'].loc[dict(time = melt_orbit)] <= 1 , 1)
            dataset['perma_wet'].loc[dict(time = melt_orbit)] = \
                dataset['perma_wet'].loc[dict(time = melt_orbit)].rolling(time = 4).mean()
            dataset['perma_wet'].loc[dict(time = melt_orbit)] = \
                dataset['perma_wet'].loc[dict(time = melt_orbit)].rolling(time = len(orbit_dataset.time), min_periods = 1).max()
        
    dataset['perma_wet'] = dataset['perma_wet'].where(~dataset['perma_wet'].isnull(), 0)
    dataset['wet_snow'] = dataset['wet_snow'].where(dataset['perma_wet'] < 0.5, 1)

    return dataset