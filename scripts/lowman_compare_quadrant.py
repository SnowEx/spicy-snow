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

# area = shapely.geometry.box(-117, 43, -113, 46)

os.makedirs(Path('~/scratch/spicy-lowman-quadrant/').expanduser(), exist_ok = True)

dates = get_input_dates('2021-04-01') #2021-04-01

from shapely import geometry
from itertools import product
for lon_min, lat_min in product(range(-117, -113), range(43, 46)):
    area = shapely.geometry.box(lon_min, lat_min, lon_min + 1, lat_min + 1)
    out_nc = Path(f'~/scratch/spicy-lowman-quadrant/spicy-lowman_{lon_min}-{lon_min + 1}_{lat_min}-{lat_min + 1}.nc').expanduser()
    spicy_ds = retrieve_snow_depth(area = area, dates = dates, 
                                work_dir = Path('~/scratch/spicy-lowman-quadrant/data/').expanduser(), 
                                job_name = f'spicy-lowman-{lon_min}-{lon_min + 1}_{lat_min}-{lat_min + 1}', # v1
                                existing_job_name = f'spicy-lowman-{lon_min}-{lon_min + 1}_{lat_min}-{lat_min + 1}', # v1
                                debug=False,
                                outfp=out_nc)
            
    spicy_ds.to_netcdf(out_nc)
