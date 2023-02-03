"""
Functions to download IMS snow-coverage images.

https://usicecenter.gov/Products/ImsHome
https://nsidc.org/data/user-resources/help-center/how-access-data-using-ftp-client-command-line-wget-or-python
"""

import sys
import os
from os.path import basename, exists, expanduser, join
import shutil
from datetime import datetime
import urllib.request
from urllib.error import URLError
from tqdm import tqdm
from pyproj import Transformer
import shapely
import pandas as pd
import xarray as xr
import rioxarray as rxa


sys.path.append(expanduser('~/Documents/spicy-snow'))
from spicy_snow.utils.download import url_download, decompress

def get_ims_day_data(year: str, doy: str, tmp_dir: str) -> xr.DataArray:
    """
    Download and decompress one days worth of IMS data.

    Args:
    year: Year of the data you want.
    doy: Calendar day of year you want in 'DDD' format. Range is 001 - 366.
    """
    os.makedirs(tmp_dir, exist_ok = True)
    os.chdir(tmp_dir)
    local_fp, _ = urllib.request.urlretrieve(f'ftp://sidads.colorado.edu/pub/DATASETS/NOAA/G02156/netcdf/1km/{year}/ims{year}{doy}_1km_v1.3.nc.gz', f'ims{year}{doy}_1km_v1.3.nc.gz')
    out_file = decompress(local_fp, local_fp.replace('.gz',''))
    ims = rxa.open_rasterio(out_file, decode_times = False)

    return ims

def download_snow_cover(dataset: xr.Dataset, tmp_dir: str = './tmp', clean: bool = True) -> xr.Dataset:
    """
    Download IMS snow-cover images.

    Args:
    dataset: Full dataset to add IMS data to
    tmp_dir: filepath to save temporary downloads to [default: './tmp']
    clean: Remove temporary directory after download?

    Returns:
    None
    """
    days = [pd.to_datetime(d) for d in dataset.time.values]
    for day in tqdm(days):
        ims = get_ims_day_data(day.year, f'{day.day:03}', tmp_dir = tmp_dir) #revert to clean = True at somepoint
        dataset = add_ims_data(dataset, ims, day)
    
    if clean == True:
        shutil.rmtree(tmp_dir)
    
    return dataset

def add_ims_data(dataset: xr.Dataset, ims: xr.DataArray, date: pd.Timestamp) -> xr.Dataset:
    """
    Add xarray dataArray of IMS data to a larger xarray Dataset.

    Args:
    dataset: large dataset to add IMS data to
    ims: IMS dataArray for one days worth of data
    date: Date of IMS retrieval
    """
    transformer = Transformer.from_crs(4326, 9001, always_xy=True)
    polar_bounds = transformer.transform(*dataset['s1'].rio.bounds())
    ims = ims.rio.clip_box(*polar_bounds)
    ims = ims.rio.reproject_match(dataset['s1'])
    ims = ims.assign_coords(time = [date])
    dataset = xr.merge([dataset, ims.rename('ims')])

    return dataset

if __name__ == '__main__':
    import pickle
    with open('/Users/zachkeskinen/Documents/spicy-snow/tests/test_data/s1_da.pkl', 'rb') as f:
        da = pickle.load(f)
    ds = da.to_dataset(name = 's1', promote_attrs = True)
    ds = download_snow_cover(ds, tmp_dir= '/Users/zachkeskinen/Documents/spicy-snow/data/tmp')
    with open('/Users/zachkeskinen/Documents/spicy-snow/data/ims_v1.pkl', 'wb') as f:
        pickle.dump(ds, f)