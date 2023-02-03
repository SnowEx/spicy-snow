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
    fcf_url = 'https://zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Tree-CoverFraction-layer_EPSG-4326.tif'
    url_download(fcf_url, out_fp)
    fcf = rxa.open_rasterio(out_fp)
    return fcf

def add_fcf(dataset: xr.Dataset, fcf: xr.DataArray) -> xr.Dataset:
    fcf = fcf.rio.clip_box(*dataset['s1'].rio.bounds())
    fcf = fcf.rio.reproject_match(dataset)
    fcf = fcf.squeeze('band')
    dataset = xr.merge([dataset, fcf.rename('fcf')])

    return dataset

if __name__ == '__main__':
    fcf = download_fcf('/Users/zachkeskinen/Documents/spicy-snow/data/fcf.tif')
    import pickle
    with open('/Users/zachkeskinen/Documents/spicy-snow/data/ims_v1.pkl', 'rb') as f:
        ds = pickle.load(f)
    ds = add_fcf(ds, fcf)
    with open('/Users/zachkeskinen/Documents/spicy-snow/data/fcf_ims_v1.pkl', 'wb') as f:
        pickle.dump(ds, f)