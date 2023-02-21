"""
Functions to identify, mask, and create weights for wet-snow.
"""

import xarray as xr

def id_new_wet_snow(dataset: xr.Dataset, wet_thresh: int = -2, inplace: bool = False) -> xr.Dataset:
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

    # identify possible newly wet snow in regions FCF < 0.5 with deltaCR

    # identify possible newly wet snow in regions FCF > 0.5 with deltaVV

    # add melting as data variable to dataset

    pass

def id_newly_frozen_snow(dataset: xr.Dataset, freeze_thresh: int = 2, inplace: bool = False) -> xr.Dataset:
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

    # identify possible re-freezing by increases of deltaGammaNaught of 2dB

    # add freezing as data variable to dataset

    pass

def id_wet_snow(dataset: xr.Dataset, inplace: bool = False) -> xr.Dataset:
    """
    Identifies time steps with wet snow. Sets all time slices, for a relative orbit,
    as dry until the first melting then sets all time steps, for that relative orbit,
    as wet until either end of time series or next re-freezing.

    If re-freezing revert to set all as dry until next melt. Repeat to end of time series.

    Additional wet snow criteria if sd retrieval (snow-index since they are linear)
    becomes negative with snow cover is present we set pixel to wet. 
    
    If after Feburary 1st until August 1st if last two of four relative orbits
    were classified as wet we set remained to wet to stop retrievals

    Args:
    dataset: xarray dataset with melting, freezing, snow_index as data vars
    inplace: return copy of dataset or operate on dataset inplace?

    Returns:
    dataset: xarray data with wet_snow data var
    """
    # create new data variable wet_snow and set to 0

    # identify all relative orbits and loop

    # for each time step 

        # set to wet if previous was wet

        # set to wet if melting == True

        # set to wet if snow_index < negative and snow_cover = True

        # set to dry if freezing == True and perma-wet = False

        # set to dry if snow cover = False and perma-wet = False

        # if >50% wet of last 4 cycles after feb 1 then set remainer till
        # august 1st to perma-wet(2?)