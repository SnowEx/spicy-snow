"""
Functions to preprocess s1-images using methods of 2021 Lievens et al.

https://tc.copernicus.org/articles/16/159/2022/#section2
"""

from typing import Dict, List
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from rioxarray.merge import merge_arrays
from itertools import product

# import sys
# sys.path.append('/Users/zmhoppinen/Documents/spicy-snow/spicy_snow')
from spicy_snow.spicy_constants import s1_dual_pols

import logging
log = logging.getLogger(__name__)

def amplitude_to_dB(dataset: xr.Dataset):
    """
    Convert s1 images from amplitude to dB

    Args:
    dataset: Xarray Dataset of sentinel images in amplitude

    Returns:
    dataset: Xarray dataset of sentinel image in dB
    """
    # check for dB
    if dataset.attrs.get('s1_units') == 'dB':
        log.info("Sentinel 1 units already in dB.")
        return dataset
    
    for pol in s1_dual_pols:
        # mask all values 0 or negative
        dataset[pol] = dataset[pol].where(dataset[pol] > 0)
        # convert all s1 images from amplitude to dB
        dataset[pol] = 10 * np.log10(dataset[pol])

    dataset.attrs['s1_units'] = 'dB'
    return dataset
    
def dB_to_amplitude(dataset: xr.Dataset):
    """
    Convert s1 images from dB to amp

    Args:
    dataset: Xarray Dataset of sentinel images in dB

    Returns:
    dataset: Xarray Dataset of sentinel images in amplitude
    """
    # check for amp
    if dataset.attrs.get('s1_units') == 'amp':
        log.info("Sentinel 1 units already in dB.")
        return dataset
        
    # convert all s1 images from amplitude to dB
    for pol in s1_dual_pols:
        dataset[pol] = 10 ** (dataset[pol] / 10)
    
    dataset.attrs['s1_units'] = 'amp'
    return dataset

def s1_orbit_averaging(dataset: xr.Dataset) -> xr.Dataset:
    """
    Normalize s1 images by rescaling each image so its orbit's mean matches the
    overall time series mean. To allow for different orbits to be compared

    Args:
    dataset: Xarray Dataset of sentinel images to normalize by orbit 

    Returns:
    dataset: Xarray Dataset of sentinel images with all s1 images normalized to total mean
    """

    # check for dB
    # TODO does this make sense?
    if dataset.attrs.get('s1_units') == 'amp':
        dataset = amplitude_to_dB(dataset)

    # get all unique relative orbits
    orbits = np.unique(dataset['track'].values)

    # loop through bands
    for pol in s1_dual_pols:
        
        # calculate the overall (all orbits) mean
        overall_mean  = dataset[pol].mean(dim = ['x','y','time'])
        log.debug(f"dataset's mean: {overall_mean}")

        for orbit in orbits:
            
            # calculate each orbit's mean value
            orbit_mean = dataset[pol].sel(time = dataset.track == orbit).mean(dim = ['x','y','time'])
            log.debug(f"Orbit's {orbit} pre-mean: {overall_mean}")

            # rescale each image by the mean correction (orbit mean -> overall mean)
            dataset[pol].loc[dict(time = dataset.track == orbit)] = \
                dataset[pol].loc[dict(time = dataset.track == orbit)] - (orbit_mean - overall_mean)
    
    return dataset

def s1_clip_outliers(dataset: xr.Dataset) -> xr.Dataset:
    """
    Remove s1 image outliers by masking pixels 3 dB above 90th percentile or
    3 dB before the 10th percentile. (-35 -> 15 dB for VV) and (-40 -> 10 for VH
    in HL's code - line # 291)

    Args:
    dataset: Xarray Dataset of sentinel images to clip outliers

    Returns:
    dataset: Xarray Dataset of sentinel images with masked outliers
    """
    # check for dB
    if 's1_units' in dataset.attrs.keys():
        assert dataset.attrs['s1_units'] == 'dB', "Sentinel 1 units must be dB not amplitude."

    # Calculate time series 10th and 90th percentile 
    # Threshold vals 3 dB above/below percentiles
    for pol in s1_dual_pols:
        data = dataset[pol]

        thresh_lo, thresh_hi = data.quantile([0.1, 0.9], skipna = True)
        thresh_lo -= 3
        thresh_hi += 3
        # Mask using percentile thresholds
        data_masked = data.where((data > thresh_lo) & (data < thresh_hi))


        log.debug(f'Clipping: {pol}')
        log.debug(f'Thresh min: {thresh_lo.values}. Thresh max: {thresh_hi.values}')
        pre_min, pre_max = data.min().values, data.max().values
        log.debug(f'Data min: {pre_min}. Data max: {pre_max}')
        min, max = data_masked.min().values, data_masked.max().values
        log.debug(f'Masked data min: {min}. Data max: {max}')

        dataset[pol] = data_masked

    return dataset

def ims_water_mask(dataset: xr.Dataset) -> xr.Dataset:
    """
    Mask s1 pixels over water (not masked in watermask)

    Args:
    dataset: Xarray Dataset of sentinel images to clip outliers

    Returns:
    dataset: Xarray Dataset of sentinel images with masked water and sea ice
    """

    assert 'watermask' in dataset.data_vars

    dataset = dataset.where(dataset['watermask'] != 1)

    return dataset

def merge_s1_subsets(dataset_dictionary: Dict[str, xr.Dataset]) -> xr.Dataset:
    """
    Merge dictionarys containing Datasets within into one dataset

    Args:
    {'1A-asc': dataset, ...}: dictionary of up to 4 s1 datasets for permutations of 
    ascending, descending, and 1A and 1B with keys {satellite}-{direction}.

    Returns:
    dataset: Xarray Dataset of all preprocessed sentinel images
    """
    # merge subsets of orbit/satellite into one Dataset
    dataset = xr.merge(dataset_dictionary.values(), compat='override')

    return dataset