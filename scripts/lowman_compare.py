import pickle
import shapely
from glob import glob
import os
from os.path import join, exists
import pandas as pd
import xarray as xr
import rioxarray as rxa

from pathlib import Path

# Add main repo to path
import sys
from os.path import expanduser
sys.path.append(expanduser('../'))

from spicy_snow.retrieval import retrieve_snow_depth
from spicy_snow.IO.user_dates import get_input_dates

area = shapely.geometry.box(-117, 43, -113, 46)

os.makedirs(Path('~/scratch/spicy-lowman/').expanduser(), exist_ok = True)
out_nc = Path('~/scratch/spicy-lowman/spicy-lowman.nc').expanduser()

dates = get_input_dates('2021-04-01') #2021-04-01

spicy_ds = retrieve_snow_depth(area = area, dates = dates, 
                               work_dir = Path('~/scratch/spicy-lowman/data/').expanduser(), 
                               job_name = f'spicy-lowman-v1', # v1
                               existing_job_name = 'spicy-lowman-v1', # v1
                               debug=False,
                               outfp=out_nc)
        
spicy_ds.to_netcdf(out_nc)
