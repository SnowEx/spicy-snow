"""
Functions to identify, mask, and create weights for wet-snow.
"""
import sys
import numpy as np
import xarray as xr
from typing import Union

import logging
log = logging.getLogger(__name__)

def id_newly_wet_snow(dataset: xr.Dataset, wet_thresh: int = -2) -> Union[None, xr.Dataset]:
    """
    Identifies time steps with newly wet snow. Identifies time slices where
    deltavv decreases by 2dB in pixels with FCF > 0.5 and deltaCR decreases by
    2dB in pixels with FCF < 0.5.

    Args:
    dataset: xarray dataset with deltavv and deltaCR as data vars
    wet_thresh: decrease in dB of deltavv or deltaCR to identify melting
    inplace: return copy of dataset or operate on dataset inplace?

    Returns:
    dataset: xarray data with melting data var
    """
    
    # check we have the neccessary variables
    necessary_vars = set(['fcf', 'deltaCR', 'deltavv'])
    assert necessary_vars.issubset(set(dataset.data_vars)),\
          f"Missing variables {necessary_vars.difference(set(dataset.data_vars))}"
    
    # add wet_flag to dataset if not already present    
    if 'wet_flag' not in dataset.data_vars:
        dataset['wet_flag'] = xr.zeros_like(dataset['deltavv'])

    # identify possible newly wet snow in regions FCF < 0.5 with deltaCR
    dataset['wet_flag'] = dataset['wet_flag'].where(((dataset['fcf'] > 0.5) | (dataset['deltaCR'] > wet_thresh) | (dataset['deltaCR'].isnull())), 1)
    # identify possible newly wet snow in regions FCF > 0.5 with deltavv
    dataset['wet_flag'] = dataset['wet_flag'].where(((dataset['fcf'] < 0.5) | (dataset['deltavv'] > wet_thresh) | (dataset['deltavv'].isnull())), 1)

    # mask nans from Sentinel-1 data
    dataset['wet_flag'] = dataset['wet_flag'].where(~dataset['vv'].isnull(), np.nan)
    
    return dataset

def id_newly_frozen_snow(dataset: xr.Dataset, freeze_thresh: int = 2) -> Union[None, xr.Dataset]:
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

    # check we have the neccessary variables
    necessary_vars = set(['deltaGamma'])
    assert necessary_vars.issubset(set(dataset.data_vars)),\
          f"Missing variables {necessary_vars.difference(set(dataset.data_vars))}"

    # add wet_flag to dataset if not already present    
    if 'freeze_flag' not in dataset.data_vars:
        dataset['freeze_flag'] = xr.zeros_like(dataset['deltavv'])

    # identify possible re-freezing by increases of deltaGammaNaught of 2dB
    dataset['freeze_flag'] = dataset['freeze_flag'].where((dataset['deltaGamma'] < freeze_thresh) | (dataset['deltaGamma'].isnull()), 1)

    # mask nans from Sentinel-1 data
    dataset['freeze_flag'] = dataset['freeze_flag'].where(~dataset['snow_index'].isnull())

    return dataset

def id_wet_negative_si(dataset: xr.Dataset, wet_SI_thresh = 0) -> Union[None, xr.Dataset]:
    """
    Additional wet snow criteria if sd retrieval (snow-index since they are linear)
    becomes negative with snow cover is present we set pixel to wet.

    Args:
    dataset: xarray dataset with snow_index as data vars
    wet_SI_thresh: threshold to use for negative SI [default: 0]
    inplace: return copy of dataset or operate on dataset inplace?

    Returns:
    dataset: xarray data with wet_snow data var
    """

    # check we have the neccessary variables
    necessary_vars = set(['snow_index', 'snowcover'])
    assert necessary_vars.issubset(set(dataset.data_vars)),\
          f"Missing variables {necessary_vars.difference(set(dataset.data_vars))}"

    # add alt_wet_flag to dataset if not already present    
    if 'alt_wet_flag' not in dataset.data_vars:
        dataset['alt_wet_flag'] = xr.zeros_like(dataset['deltavv'])

    # identify wetting of snow by negative snow index with snow present
    dataset['alt_wet_flag'] = dataset['alt_wet_flag'].where(((dataset['snowcover'] == False) | (dataset['snow_index'] > wet_SI_thresh) | (dataset['snow_index'].isnull())), 1)

    # mask nans from Sentinel-1 data
    dataset['alt_wet_flag'] = dataset['alt_wet_flag'].where(~dataset['snow_index'].isnull())

    return dataset

def flag_wet_snow(dataset: xr.Dataset) -> Union[None, xr.Dataset]:
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

    # check we have the neccessary variables
    necessary_vars = set(['wet_flag', 'alt_wet_flag', 'freeze_flag'])
    assert necessary_vars.issubset(set(dataset.data_vars)),\
          f"Missing variables {necessary_vars.difference(set(dataset.data_vars))}"
    
    # create new data variable wet_snow and set to 0
    dataset['wet_snow'] = xr.zeros_like(dataset['wet_flag'])
    dataset['perma_wet'] = xr.zeros_like(dataset['wet_flag'])

    # get all unique relative orbits
    orbits = np.unique(dataset['track'].values)

    # Identify previous image from the same relative orbit (6, 12, 18, or 24 days ago)
    for orbit in orbits:
        # select just that orbit to do wet snow propogation
        orbit_dataset = dataset.sel(time = dataset.track == orbit)

        prev_time = None

        # loop through time steps in this orbit
        for ts in orbit_dataset.time:

            # if not first round then pull in previous time step's values for wet-snow
            if prev_time:
                dataset['wet_snow'].loc[dict(time = ts)] = dataset.sel(time = prev_time)['wet_snow']

            # add newly wet snow flags to old wet snow and then bound at 1

            dataset['wet_snow'].loc[dict(time = ts)] = xr.where(~dataset.sel(time = ts)['wet_flag'].isnull(),dataset.sel(time = ts)['wet_snow'] + dataset.sel(time = ts)['wet_flag'],np.nan)
            dataset['wet_snow'].loc[dict(time = ts)] = xr.where(~dataset.sel(time = ts)['alt_wet_flag'].isnull(),dataset.sel(time = ts)['wet_snow'] + dataset.sel(time = ts)['alt_wet_flag'],np.nan)
            dataset['wet_snow'].loc[dict(time = ts)] = dataset.sel(time = ts)['wet_snow'].where((dataset.sel(time = ts)['wet_snow'] < 1) | (dataset.sel(time = ts)['wet_snow'].isnull()), 1)

            # add newly frozen snow flags to old wet snow and then bound at 0 to avoid negatives
            dataset['wet_snow'].loc[dict(time = ts)] = xr.where(~dataset.sel(time = ts)['freeze_flag'].isnull(),dataset.sel(time = ts)['wet_snow'] - dataset.sel(time = ts)['freeze_flag'],np.nan)
            dataset['wet_snow'].loc[dict(time = ts)] = dataset.sel(time = ts)['wet_snow'].where((dataset.sel(time = ts)['wet_snow'] > 0) | (dataset.sel(time = ts)['wet_snow'].isnull()), 0)

            # set non snow (IMS != 4) to not wet (0)
            dataset['wet_snow'].loc[dict(time = ts)] = dataset.sel(time = ts)['wet_snow'].where(dataset.sel(time = ts)['snowcover'] == True, 0)

            # make nans at areas without S1 data
            dataset['wet_snow'].loc[dict(time = ts)] = dataset.sel(time = ts)['wet_snow'].where(~dataset['vv'].sel(time = ts).isnull(), np.nan)
                    
            prev_time = ts

        # if >50% wet of last 4 cycles after feb 1 then set remainer till
        # august 1st to perma-wet(2?)
    
        melt_season = (dataset['time.month'] > 1) & (dataset['time.month'] < 8)

        # get images of this relative orbit and in the melt season
        melt_season = (dataset['time.month'] > 1) & (dataset['time.month'] < 8)

        # get images of this relative orbit and in the melt season
        melt_orbit = (melt_season & (dataset.track == orbit))

        # check if there are at least 4 time slices in melt season for this orbit
        if len(dataset['perma_wet'].loc[dict(time = melt_orbit)]) < 4:
            continue

        # first set all perma wet to match the flagged wet times from dB drop
        dataset['perma_wet'].loc[dict(time = melt_orbit)] = \
            dataset['wet_flag'].loc[dict(time = melt_orbit)]
        
        # then set times that were made wet by negative snow index to wet as well
        dataset['perma_wet'].loc[dict(time = melt_orbit)] = dataset['perma_wet'].loc[dict(time = melt_orbit)] + \
            dataset['alt_wet_flag'].loc[dict(time = melt_orbit)]
        
        # now if we are over 1 (ie it was flagged wet by dB and negative SI flags) we should floor those back to 1
        dataset['perma_wet'].loc[dict(time = melt_orbit)] = \
            dataset['perma_wet'].loc[dict(time = melt_orbit)].where((dataset['perma_wet'].loc[dict(time = melt_orbit)] <= 1) | (dataset['perma_wet'].loc[dict(time = melt_orbit)].isnull()) , 1)
        
        # now calculate the rolling mean of the perma wet so we have a % 0-1 of days out of 4 that were flagged
        dataset['perma_wet'].loc[dict(time = melt_orbit)] = \
            dataset['perma_wet'].loc[dict(time = melt_orbit)].rolling(time = 4).mean()
        
        # then propogate forward so that if we get to > 50% in a 4 image window we mask the remained of the melt season
        # this will fail if bottleneck is installed due to it lacking the min_periods keyword
        # see: https://github.com/pydata/xarray/issues/4922


        if 'bottleneck' not in sys.modules:
            dataset['perma_wet'].loc[dict(time = melt_orbit)] = \
                dataset['perma_wet'].loc[dict(time = melt_orbit)].rolling(time = len(orbit_dataset.time), min_periods = 1).max()
        else:
            log.info("bottleneck installed. Consider pip uninstalling and re-running if this fails.")
            dataset['perma_wet'].loc[dict(time = melt_orbit)] = \
                dataset['perma_wet'].loc[dict(time = melt_orbit)].rolling(time = len(orbit_dataset.time)).max()

        # set perma wet to nans if no S1 data
        dataset['perma_wet'].loc[dict(time = melt_orbit)] = dataset.sel(dict(time = melt_orbit))['perma_wet'].where(~dataset['vv'].sel(dict(time = melt_orbit)).isnull(), np.nan)

        # set perma wet to 0 if no snow
        dataset['perma_wet'].loc[dict(time = melt_orbit)] = dataset.sel(dict(time = melt_orbit))['perma_wet'].where(dataset['snowcover'].sel(dict(time = melt_orbit)) == True, 0)

    # if we have no data just set it to not be flagged perma_wet
    dataset['perma_wet'] = dataset['perma_wet'].where(~dataset['perma_wet'].isnull(), 0)

    # if less than 50% are wet then keep the save value for wet_snow otherwise set to 1
    dataset['wet_snow'] = dataset['wet_snow'].where(dataset['perma_wet'] < 0.5, 1)

    dataset['wet_snow'].loc[dict(time = ts)] = dataset.sel(time = ts)['wet_snow'].where(dataset.sel(time = ts)['snowcover'] == True, 0)

    return dataset
