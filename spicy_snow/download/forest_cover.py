"""
Functions to download PROBA-V forest-cover-fraction images for specific geometries
"""

import sys
import os
from os.path import basename, exists, expanduser, join
import shutil
import shapely
import rioxarray as rxa
import xarray as xr

sys.path.append(expanduser('~/Documents/spicy-snow'))
from spicy_snow.utils.download import url_download

def download_fcf(out_fp: str) -> xr.DataArray:
    """
    Download PROBA-V forest-cover-fraction images.

    Args:
    out_fp: filepath to save tiff of results

    Returns:
    None
    """
    # this is the url from Lievens et al. 2021 paper
    fcf_url = 'https://zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Tree-CoverFraction-layer_EPSG-4326.tif'
    # download just forest cover fraction to out file
    url_download(fcf_url, out_fp)
    # open as dataArray and return
    fcf = rxa.open_rasterio(out_fp)
    return fcf

def add_fcf(dataset: xr.Dataset, fcf: xr.DataArray) -> xr.Dataset:
    """
    Add xarray dataArray of forest cover data to a larger xarray Dataset.

    Args:
    dataset: large dataset to add IMS data to
    fcf: fcf dataArray for all data

    Returns:
    dataset: large dataset with 'fcf' added as data variable
    """
    # clip FCF to dataset boundaries (from user defined geometry)
    fcf = fcf.rio.clip_box(*dataset['s1'].rio.bounds())
    # reproject FCF to match dataset
    fcf = fcf.rio.reproject_match(dataset['s1'])
    # remove band dimension as it only has one band
    fcf = fcf.squeeze('band')
    # merge FCF and name it 'fcf' as a data variable
    dataset = xr.merge([dataset, fcf.rename('fcf')])

    return dataset

# End of file