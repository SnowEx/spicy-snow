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

import logging
log = logging.getLogger(__name__)

sys.path.append(expanduser('~/Documents/spicy-snow'))
from spicy_snow.utils.download import url_download, decompress

def get_ims_day_data(year: str, doy: str, tmp_dir: str) -> xr.DataArray:
    """
    Download and decompress one days worth of IMS data.

    Args:
    year: Year of the data you want.
    doy: Calendar day of year you want in 'DDD' format. Range is 001 - 366.
    """

    # make temporary directory
    cd = os.getcwd()
    os.makedirs(tmp_dir, exist_ok = True)
    os.chdir(tmp_dir)

    # if we haven't found a file that works add one more to day and try again
    local_fp = None
    while not local_fp:
        try:
            # download IMS data for this day and year
            local_fp, _ = urllib.request.urlretrieve(f'ftp://sidads.colorado.edu/pub/DATASETS/NOAA/G02156/netcdf/1km/{year}/ims{year}{doy}_1km_v1.3.nc.gz', f'ims{year}{doy}_1km_v1.3.nc.gz')
        except:
            doy = f'{int(doy) + 1:03}'
    # decompress .gz compression
    out_file = decompress(local_fp, local_fp.replace('.gz',''))
    # open as xarray dataArray
    ims = rxa.open_rasterio(out_file, decode_times = False)
    os.chdir(cd)
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
    # get list of days that we have Sentinel-1 data for
    days = [pd.to_datetime(d) for d in dataset.time.values]

    all_ims = []
    for day in tqdm(days, desc = 'Downloading IMS snow-cover'):
        # download IMS data for this day and open with xarray
        ims = get_ims_day_data(day.year, f'{day.dayofyear:03}', tmp_dir = tmp_dir) #revert to clean = True at somepoint
        # add timestamp info
        ims = ims.assign_coords(time = [day])
        # reproject and clip to match dataset
        ims = ims.rio.reproject_match(dataset['s1'])
        # add day to list of ims days
        all_ims.append(ims)
    # make dataArray of all IMS images
    full_ims = xr.concat(all_ims, dim = 'time')

    # add dataArray of IMS to the sentinel-1 dataset with timestamp information
    dataset = xr.merge([dataset, full_ims.rename('ims')])
    
    # remove data directory of downloaded IMS tifs
    if clean == True:
        shutil.rmtree(tmp_dir)
    
    return dataset

# End of file