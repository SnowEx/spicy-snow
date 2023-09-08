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

import logging
log = logging.getLogger(__name__)

def s1_power_to_dB(dataset: xr.Dataset, inplace: bool = False):
    """
    Convert s1 images from amplitude to dB

    Args:
    dataset: Xarray Dataset of sentinel images in amplitude
    inplace: boolean flag to modify original Dataset or return a new Dataset

    Returns:
    dataset: Xarray dataset of sentinel image in dB
    """
    if not inplace:
        dataset = dataset.copy(deep=True)

    # check for dB
    if 's1_units' in dataset.attrs.keys():
        if dataset.attrs['s1_units'] == 'dB' and not inplace:
            log.info("Sentinel 1 units already in dB.")
            return dataset
        if dataset.attrs['s1_units'] == 'dB':
            return
    
    # mask all values 0 or negative
    dataset['s1'] = dataset['s1'].where(dataset['s1'] > 0)
    # convert all s1 images from amplitude to dB
    dataset['s1'].loc[dict(band = ['VV','VH'])] = 10 * np.log10(dataset['s1'].sel(band = ['VV','VH']))

    dataset.attrs['s1_units'] = 'dB'
    
    if not inplace:
        return dataset

def s1_dB_to_power(dataset: xr.Dataset, inplace: bool = False):
    """
    Convert s1 images from dB to amp

    Args:
    dataset: Xarray Dataset of sentinel images in dB
    inplace: boolean flag to modify original Dataset or return a new Dataset

    Returns:
    dataset: Xarray Dataset of sentinel images in amplitude
    """
    if not inplace:
        dataset = dataset.copy(deep=True)
    
    # check for amp
    if 's1_units' in dataset.attrs.keys():
        if dataset.attrs['s1_units'] == 'amp' and not inplace:
            print("Sentinel 1 units already in dB.")
            return dataset
        if dataset.attrs['s1_units'] == 'amp':
            return
        
    # convert all s1 images from amplitude to dB
    dataset['s1'].loc[dict(band = ['VV','VH'])] = 10 ** (dataset['s1'].sel(band = ['VV','VH']) / 10)
    dataset.attrs['s1_units'] = 'amp'
    if not inplace:
        return dataset


def merge_partial_s1_images(dataset, inplace: bool = False) -> xr.Dataset:
    """
    Merges s1 images that have been split by hyp3 into a single image with the 
    first time stamp of that relative orbit pass as the time index.

    Args:
    dataset: Xarray Dataset with Sentinel 1 images that have been arbitrarily
    split by hyp3 into an arbitrary number of subswaths

    Return:
    dataset: Xarray Dataset with Sentinel 1 images combined into single images and
    only the first subswath's time step
    """
    if not inplace:
        dataset = dataset.copy(deep=True)
    
    # split out each relative orbit and make its own dataset
    for orbit_num, orbit_ds in dataset.groupby('relative_orbit'):

        # split out each absolute orbit and combine images
        for absolute_num, abs_ds in orbit_ds.groupby('absolute_orbit'):

            # if only 1 image in this absolute orbit go to the next
            if len(abs_ds.time) == 1:
                continue

            # combine all images into the first image time step
            combo_imgs = abs_ds.mean(dim = 'time')

            # set first image in dataset to this new combined image
            dataset.loc[{'time' : abs_ds.isel(time = 0).time}] = combo_imgs

            # drop images that have been combined
            dataset = dataset.drop_sel(time = abs_ds.isel(time = slice(1, len(abs_ds.time))).time)

    # can leave some outliers in the dataset along the edges so remove unreasonable values
    if 's1' in dataset.data_vars:
        dataset['s1'] = dataset['s1'].where(dataset['s1'] < 100)
        dataset['s1'] = dataset['s1'].where(dataset['s1'] > -1e30)

    if not inplace:
        return dataset

def subset_s1_images(dataset: xr.Dataset) -> Dict[str, xr.Dataset]:
    """
    Subset s1 dataset into 4 subsets: ascending 1A, descending 1A, ascending 1B,
    and descending 1B

    Args:
    dataset: Xarray Dataset of sentinel images

    Returns:
    {'1A-asc': dataset, ...}: dictionary of up to 4 s1 datasets for permutations of 
    ascending, descending, and 1A and 1B with keys {satellite}-{direction}.
    """

    # options for platform and direction
    platforms, directions = ['S1A','S1B'], ['descending', 'ascending']

    # make subset dictionary
    subset_ds = {}

    # loop through possible combinations
    for platform, direction in product(platforms,directions):

        # subset on platform and direction
        subset = dataset.sel(time = dataset.platform == platform)
        subset = subset.sel(time = subset.flight_dir == direction)

        # save subset to dictionary
        if len(subset['s1']) > 0:
            subset_ds[f'{platform}-{direction}'] = subset
            log.debug(f"{platform}-{direction}: length = {len(subset_ds[f'{platform}-{direction}'])}")
    
    return subset_ds

def s1_orbit_averaging(dataset: xr.Dataset, inplace: bool = False) -> xr.Dataset:
    """
    Normalize s1 images by rescaling each image so its orbit's mean matches the
    overall time series mean. To allow for different orbits to be compared

    Args:
    dataset: Xarray Dataset of sentinel images to normalize by orbit 
    inplace: boolean flag to modify original Dataset or return a new Dataset

    Returns:
    dataset: Xarray Dataset of sentinel images with all s1 images normalized to total mean
    """

    # check inplace flag
    if not inplace:
        dataset = dataset.copy(deep=True)
    
    # check for dB
    if 's1_units' in dataset.attrs.keys():
        assert dataset.attrs['s1_units'] == 'dB', "Sentinel 1 units must be dB not amplitude."

    # get all unique relative orbits
    orbits = np.unique(dataset['relative_orbit'].values)

    # loop through bands
    for band in ['VV', 'VH']:
        
        # calculate the overall (all orbits) mean
        overall_mean  = dataset['s1'].mean(dim = ['x','y','time']).sel(band = band)
        log.debug(f"dataset's mean: {overall_mean}")

        for orbit in orbits:
            
            # calculate each orbit's mean value
            orbit_mean = dataset['s1'].sel(time = dataset.relative_orbit == orbit, band = band).mean(dim = ['x','y','time'])
            log.debug(f"Orbit's {orbit} pre-mean: {overall_mean}")

            # rescale each image by the mean correction (orbit mean -> overall mean)
            dataset['s1'].loc[dict(time = dataset.relative_orbit == orbit, band = band)] = \
                dataset['s1'].loc[dict(time = dataset.relative_orbit == orbit, band = band)] - (orbit_mean - overall_mean)

    if not inplace:
        return dataset

def s1_clip_outliers(dataset: xr.Dataset, inplace: bool = False) -> xr.Dataset:
    """
    Remove s1 image outliers by masking pixels 3 dB above 90th percentile or
    3 dB before the 10th percentile. (-35 -> 15 dB for VV) and (-40 -> 10 for VH
    in HL's code - line # 291)

    Args:
    dataset: Xarray Dataset of sentinel images to clip outliers
    inplace: boolean flag to modify original Dataset or return a new Dataset

    Returns:
    dataset: Xarray Dataset of sentinel images with masked outliers
    """
    # Check inplace flag
    if not inplace:
        dataset = dataset.copy(deep=True)

    # check for dB
    if 's1_units' in dataset.attrs.keys():
        assert dataset.attrs['s1_units'] == 'dB', "Sentinel 1 units must be dB not amplitude."

    # Calculate time series 10th and 90th percentile 
    # Threshold vals 3 dB above/below percentiles
    for band in ['VV','VH']:
        data = dataset['s1'].sel(band=band)

        thresh_lo, thresh_hi = data.quantile([0.1, 0.9], skipna = True)
        thresh_lo -= 3
        thresh_hi += 3
        # Mask using percentile thresholds
        data_masked = data.where((data > thresh_lo) & (data < thresh_hi))


        log.debug(f'Clipping band: {band}')
        log.debug(f'Thresh min: {thresh_lo.values}. Thresh max: {thresh_hi.values}')
        pre_min, pre_max = data.min().values, data.max().values
        log.debug(f'Data min: {pre_min}. Data max: {pre_max}')
        min, max = data_masked.min().values, data_masked.max().values
        log.debug(f'Masked data min: {min}. Data max: {max}')

        dataset['s1'].loc[dict(band = band)] = data_masked

    if not inplace:
        return dataset

def ims_water_mask(dataset: xr.Dataset) -> xr.Dataset:
    """
    Mask s1 pixels over water (1) or sea ice (3)

    Args:
    dataset: Xarray Dataset of sentinel images to clip outliers

    Returns:
    dataset: Xarray Dataset of sentinel images with masked water and sea ice
    """

    # mask all pixels in dataset where ims == 1 or 3.

def s1_incidence_angle_masking(dataset: xr.Dataset, inplace: bool = False) -> xr.Dataset:
    """
    Remove s1 image outliers by masking pixels with incidence angles > 70 degrees
    
    Args:
    dataset: Xarray Dataset of sentinel images to mask

    Returns:
    dataset: Xarray Dataset of sentinel images with incidence angles > 70 degrees
    masked
    """
    
    # Check inplace flag
    if not inplace:
            dataset = dataset.copy(deep=True)

    # Mask pixels with incidence angle > 70 degrees
    dataset['s1'] = dataset['s1'].where(dataset['s1'].sel(band = 'inc') < np.deg2rad(70))

    if not inplace:
            return dataset

def merge_s1_subsets(dataset_dictionary: Dict[str, xr.Dataset]) -> xr.Dataset:
    """
    Remove s1 image outliers by masking pixels with incidence angles > 70 degrees

    Args:
    {'1A-asc': dataset, ...}: dictionary of up to 4 s1 datasets for permutations of 
    ascending, descending, and 1A and 1B with keys {satellite}-{direction}.

    Returns:
    dataset: Xarray Dataset of all preprocessed sentinel images
    """
    # merge subsets of orbit/satellite into one Dataset
    dataset = xr.merge(dataset_dictionary.values())

    return dataset

def add_confidence_angle(dataset: xr.Dataset, inplace: bool = False):
    """
    Function to add confidence angle to dataset.

    Confidence Angle = angle[ abs(dVH/dt / mean(dVH/dt)), abs(dVV/dt / mean(dVV/dt)) ]

    Args:
    dataset: Xarray Dataset of sentinel images to add confidence angle to
    inplace: boolean flag to modify original Dataset or return a new Dataset

    Returns:
    dataset: Xarray dataset of sentinel image with confidence interval in 
    """
    ds_amp = s1_dB_to_power(dataset).copy()
    ds_amp['deltaVH_amp'] = ds_amp['s1'].sel(band = 'VH').diff(1)
    ds_amp['deltaVV_amp'] = ds_amp['s1'].sel(band = 'VV').diff(1)

    ds_amp['deltaVH_norm'] = np.abs(ds_amp['deltaVH_amp'] / ds_amp['deltaVH_amp'].mean())
    ds_amp['deltaVV_norm'] = np.abs(ds_amp['deltaVV_amp'] / ds_amp['deltaVV_amp'].mean())

    ds_amp['confidence'] = (ds_amp['deltaVH_norm'].dims, np.angle(ds_amp['deltaVV_norm'].values + ds_amp['deltaVH_norm'].values * 1j))
    dataset['confidence'] = ds_amp['confidence'].mean('time')

    if not inplace:
        return dataset
