import pickle
import shapely
from glob import glob
import os
from os.path import join, exists
import pandas as pd
import xarray as xr
import rioxarray as rxa

# Add main repo to path
import sys
from os.path import expanduser
sys.path.append(expanduser('../'))

from spicy_snow.retrieval import retrieve_snow_depth
from spicy_snow.IO.user_dates import get_input_dates

area = shapely.geometry.box(-117, 43, -113, 46)

os.makedirs('/home/zacharykeskinen/scratch/spicy-lowman/', exist_ok = True)
out_nc = f'/home/zacharykeskinen/scratch/spicy-lowman/spicy-lowman.nc'

dates = get_input_dates('2021-04-01')

spicy_ds = retrieve_snow_depth(area = area, dates = dates, 
                               work_dir = '/home/zacharykeskinen/scratch/data/', 
                               job_name = f'spicy-lowman-v1', 
                               existing_job_name = 'spicy-lowman-v1',
                               debug=False)
        
spicy_ds.to_netcdf(out_nc)
