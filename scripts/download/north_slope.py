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

area = shapely.geometry.box(-150, 68.5, -149, 69.5)

os.makedirs('/home/zacharykeskinen/scratch/north-slope/', exist_ok = True)
out_nc = f'/home/zacharykeskinen/scratch/north-slope/spicy-imnavait.nc'

dates = get_input_dates('2023-03-10')

spicy_ds = retrieve_snow_depth(area = area, dates = dates, 
                               work_dir = '/home/zacharykeskinen/scratch/data/', 
                               job_name = f'spicy-imnavait-v3', 
                               existing_job_name = 'spicy-imnavait-v3',
                               debug=False)

#ds.attrs['site'] = 'imnavait-creek'
        
spicy_ds.to_netcdf(out_nc)
