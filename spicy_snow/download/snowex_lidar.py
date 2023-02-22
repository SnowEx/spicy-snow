import sys
import os
from os.path import basename, exists, expanduser, join
import shutil
import asf_search as asf
import pandas as pd
import xarray as xr
import rioxarray as rxa
from rioxarray.merge import merge_arrays
import shapely.geometry
from datetime import date
from tqdm import tqdm


import sys
from os.path import expanduser
sys.path.append(expanduser('~/Documents/spicy-snow'))
from spicy_snow.utils.nsidc import cmr_download, cmr_search

import logging
log = logging.getLogger(__name__)

def download_dem(out_dir = './tmp'):
    """
    Function to download snow-free DEMs to outdir
    https://nsidc.org/data/snex20_qsi_dem/versions/1

    Args:
    out_dir: directory to save lidar veg height tifs to

    Returns:
    None
    """
    os.makedirs(out_dir, exist_ok = True)
    old_cd = os.getcwd()
    os.chdir(out_dir)

    short_name = 'SNEX20_QSI_DEM'
    version = '1'
    time_start = '2021-09-16T00:00:00Z'
    time_end = '2021-09-21T23:59:59Z'
    bounding_box = ''
    polygon = ''
    filename_filter = ''
    url_list = []
    
    quiet = False
    force = False

    try:
        if not url_list:
            url_list = cmr_search(short_name, version, time_start, time_end,
                                    bounding_box=bounding_box, polygon=polygon,
                                    filename_filter=filename_filter, quiet=quiet)

        cmr_download(url_list, force=force, quiet=quiet)
    except KeyboardInterrupt:
        quit()

    os.chdir(old_cd)

def download_snow_depth(out_dir = './tmp'):
    """
    Function to download snow depth files to outdir.
    https://nsidc.org/data/snex20_qsi_sd/versions/1

    Args:
    out_dir: directory to save lidar veg height tifs to

    Returns:
    None
    """

    os.makedirs(out_dir, exist_ok = True)
    old_cd = os.getcwd()
    os.chdir(out_dir)

    short_name = 'SNEX20_QSI_SD'
    version = '1'
    time_start = '2020-02-09T00:00:00Z'
    time_end = '2021-03-20T23:59:59Z'
    bounding_box = ''
    polygon = ''
    filename_filter = ''
    url_list = []
    
    quiet = False
    force = False

    try:
        if not url_list:
            url_list = cmr_search(short_name, version, time_start, time_end,
                                    bounding_box=bounding_box, polygon=polygon,
                                    filename_filter=filename_filter, quiet=quiet)

        cmr_download(url_list, force=force, quiet=quiet)
    except KeyboardInterrupt:
        quit()
    
    os.chdir(old_cd)

def download_veg_height(out_dir = './tmp'):
    """
    Function to download vegation height files to outdir
    https://nsidc.org/data/snex20_qsi_vh/versions/1

    Args:
    out_dir: directory to save lidar veg height tifs to

    Returns:
    None
    """

    os.makedirs(out_dir, exist_ok = True)
    old_cd = os.getcwd()
    os.chdir(out_dir)

    short_name = 'SNEX20_QSI_VH'
    version = '1'
    time_start = '2020-02-01T00:00:00Z'
    time_end = '2021-03-20T23:59:59Z'
    bounding_box = '-116.81,39.69,-105.64,44.38'
    polygon = ''
    filename_filter = ''
    url_list = []
    
    quiet = False
    force = False

    try:
        if not url_list:
            url_list = cmr_search(short_name, version, time_start, time_end,
                                    bounding_box=bounding_box, polygon=polygon,
                                    filename_filter=filename_filter, quiet=quiet)

        cmr_download(url_list, force=force, quiet=quiet)
    except KeyboardInterrupt:
        quit()
    
    os.chdir(old_cd)