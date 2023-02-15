"""
Functions to preprocess s1-images using methods of 2021 Lievens et al.

https://tc.copernicus.org/articles/16/159/2022/#section2
"""

from typing import Dict
import numpy as np
import xarray as xr

def s1_amp_to_dB(dataset: xr.Dataset):
    """
    Convert s1 images from amplitude to dB

    Args:
    dataset: Xarray Dataset of sentinel images in amplitude

    Returns:
    None: modifies Dataset in place
    """
    # mask all values 0 or negative
    dataset['s1'].loc[dict(band = ['VV','VH'])] = dataset['s1'].loc[dict(band = ['VV','VH'])].where(dataset['s1'].loc[dict(band = ['VV','VH'])] > 0)
    # convert all s1 images from amplitude to dB
    dataset['s1'].loc[dict(band = ['VV','VH'])] = 10 * np.log10(dataset['s1'].sel(band = ['VV','VH']))

def s1_dB_to_amp(dataset: xr.Dataset):
    """
    Convert s1 images from dB to amp

    Args:
    dataset: Xarray Dataset of sentinel images in dB

    Returns:
    None: modifies Dataset in place
    """

    # convert all s1 images from amplitude to dB
    dataset['s1'].loc[dict(band = ['VV','VH'])] = 10 ** (dataset['s1'].sel(band = ['VV','VH']) / 10)

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

    # Calculate unique orbits, ascending, satellites

    # Add orbit, ascending vs. descending, satellite #

    # split Dataset by satellite and 

def s1_orbit_averaging(dataset: xr.Dataset) -> xr.Dataset:
    """
    Normalize s1 images by rescaling each image so its orbit's mean matches the
    overall time series mean. To allow for different orbits to be compared

    Args:
    dataset: Xarray Dataset of sentinel images to normalize by orbit 

    Returns:
    dataset: Xarray Dataset of sentinel images with all s1 images normalized to total mean
    """
    # Calculate the overall (all orbits) mean

    # Calculate each orbit's mean value

    # rescale each image by the mean correction (orbit mean -> overall mean)

    pass

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

    # Calculate time series 10th and 90th percentile 

    # mask pixels 3dB below or 3dB above percentiles

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

def merge_s1_subsets(dataset: Dict[str, xr.Dataset]) -> xr.Dataset:
    """
    Remove s1 image outliers by masking pixels with incidence angles > 70 degrees

    Args:
    {'1A-asc': dataset, ...}: dictionary of up to 4 s1 datasets for permutations of 
    ascending, descending, and 1A and 1B with keys {satellite}-{direction}.

    Returns:
    dataset: Xarray Dataset of all preprocessed sentinel images
    """

    # merge subsets of orbit/satellite into one Dataset

