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
            print("Sentinel 1 units already in dB.")
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

def merge_s1_times(dataset: xr.Dataset, times: List[np.datetime64], verbose: bool = False) -> xr.Dataset:
    """
    Merge sentinel-1 times into a single timestamp using first time of list times

    Args:
    dataset: Xarray Dataset with Sentinel 1 images
    times: list of times to combine
    verbose: print out times to be combined?

    Return:
    Xarray dataset: Xarray Dataset with sentinel-1 images with times combined into one
    """

    if verbose:
        print('Merging:', times)

    # make list of DataArrays at each time step in times
    das = [dataset.sel(time = ts)['s1'] for ts in times]

    # check all images are same relative_orbit? 
    assert len(das) == len([d for d in das if d['relative_orbit'] == das[0]['relative_orbit']])
    if das[0].where(das[0] == 0).notnull().sum() > 0:
        nodata_value = 0
    else:
        nodata_value = np.nan
    merged = merge_arrays(das, crs = 'EPSG:4326', nodata= nodata_value)
    dataset = dataset.drop_sel(time = times[1:])
    dataset['s1'].loc[dict(time = times[0])] = merged.values
    times = []

    return dataset

def merge_partial_s1_images(dataset: xr.Dataset) -> xr.Dataset:
    """
    Merges s1 images that have been split by hyp3 into a single image with the 
    first time stamp of that relative orbit pass as the time index.

    Args:
    dataset: Xarray Dataset with Sentinel 1 images that have been arbitrarily
    split by hyp3

    Return:
    dataset: Xarray Dataset with Sentinel 1 images combined into single images and
    few time steps
    """
    times = []
    for ts in dataset.time.values:
        
        if not times:
            times.append(ts)
            continue

        if abs(ts - times[0]) > pd.Timedelta('1 minute'):
            if len(times) == 1:
                times = [ts]
                continue
            else:
                dataset = merge_s1_times(dataset, times)
                times = []
        else:
            times.append(ts)
        
        if ts == dataset.time.values[-1] and len(times) > 1:
            dataset = merge_s1_times(dataset, times)
            times = []
        
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
        subset_ds[f'{platform}-{direction}'] = subset
    
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
        for orbit in orbits:
            
            # calculate each orbit's mean value
            orbit_mean = dataset['s1'].sel(time = dataset.relative_orbit == orbit, band = band).mean(dim = ['x','y','time'])

            # rescale each image by the mean correction (orbit mean -> overall mean)
            dataset['s1'].loc[dict(time = dataset.relative_orbit == orbit, band = band)] = \
                dataset['s1'].loc[dict(time = dataset.relative_orbit == orbit, band = band)] - (orbit_mean - overall_mean)

    if not inplace:
        return dataset

def s1_clip_outliers(dataset: xr.Dataset, inplace: bool = False, verbose: bool = False) -> xr.Dataset:
    """
    Remove s1 image outliers by masking pixels 3 dB above 90th percentile or
    3 dB before the 10th percentile. (-35 -> 15 dB for VV) and (-40 -> 10 for VH
    in HL's code - line # 291)

    Args:
    dataset: Xarray Dataset of sentinel images to clip outliers
    inplace: boolean flag to modify original Dataset or return a new Dataset
    verbose: flag to print out masking stats

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

        if verbose:
            print(f'Clipping band: {band}')
            print(f'Thresh min: {thresh_lo.values}. Thresh max: {thresh_hi.values}')
            pre_min, pre_max = data.min().values, data.max().values
            print(f'Data min: {pre_min}. Data max: {pre_max}')
            min, max = data_masked.min().values, data_masked.max().values
            print(f'Masked data min: {min}. Data max: {max}')

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

def s1_incidence_angle_masking(dataset: xr.Dataset) -> xr.Dataset:
    """
    Remove s1 image outliers by masking pixels with incidence angles > 70 degrees

    Args:
    dataset: Xarray Dataset of sentinel images to mask incidence angle outliers

    Returns:
    dataset: Xarray Dataset of sentinel images with incidence angles > 70 degrees
    masked
    """

    # mask pixels with incidence angle > 70 degrees

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

