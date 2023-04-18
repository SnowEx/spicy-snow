# FOR GENERATING THE PARAMETER DATASETS! #

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from shapely import wkt
from shapely.geometry import box
from pathlib import Path
from datetime import datetime
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import sys
sys.path.append('../..')

from spicy_snow.processing.snow_index import calc_delta_cross_ratio, calc_delta_gamma, \
    clip_delta_gamma_outlier, calc_snow_index, calc_snow_index_to_snow_depth
from spicy_snow.processing.wet_snow import id_newly_wet_snow, id_wet_negative_si, \
    id_newly_frozen_snow, flag_wet_snow

# Create parameter space
A = np.round(np.arange(1, 3.1, 0.5), 2)
B = np.round(np.arange(0, 1.01, 0.1), 2)
C = np.round(np.arange(0, 1.001, 0.01), 2)

files = Path('../../Lidar_s1_stacks/').glob('*.nc')
param_dir = Path('~/scratch/params').expanduser()
for f in files:

    # get dataset
    ds_name = f.name.split('stacks/')[-1].split('.')[0]
    print(datetime.now(), f' -- starting {ds_name}')
    ds_ = xr.open_dataset(f).load()
    dataset = ds_[['s1','deltaVV','ims','fcf','lidar-sd']]

    # find closest timestep to lidar
    td = abs(pd.to_datetime(dataset.time) - pd.to_datetime(dataset.attrs['lidar-flight-time']))
    closest_ts = dataset.time[np.argmin(td)]

    param_dir.joinpath(f'{ds_name}').mkdir()

    # Brute-force processing loop
    for a in tqdm(A):
        # print(f'A: {a}')
        ds = calc_delta_cross_ratio(dataset, A = a)
        for b in B:
            # print(f'    B: {b}')
            ds = calc_delta_gamma(ds, B = b, inplace=False)
            ds = clip_delta_gamma_outlier(ds)
            ds = calc_snow_index(ds)
            ds = id_newly_wet_snow(ds)
            ds = id_wet_negative_si(ds)
            ds = id_newly_frozen_snow(ds)
            ds = flag_wet_snow(ds)
            for c in C:
                # print(f'        c: {c}')
                # print(f'A={a}; B={b}; C={c}')

                ds = calc_snow_index_to_snow_depth(ds, C = c)

                sub = ds.sel(time = closest_ts)[['snow_depth', 'wet_snow', 'lidar-sd']]
                sub.to_netcdf(param_dir.joinpath(f'{ds_name}/{a}_{b}_{c}.nc'))

# for getting more efficent numpy files

# from itertools import product
from tqdm.contrib.itertools import product
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import time

def get_numpy(x, y):
    # ravel to numpy arrays
    x = x.values.ravel()
    y = y.values.ravel()

    # remove nans
    x_buff = x[(~np.isnan(x)) & (~np.isnan(y))]
    y = y[(~np.isnan(x)) & (~np.isnan(y))]
    x = x_buff

    return x, y

param_dir = Path('~/scratch/params').expanduser()
# find length of lidar, spicy arrays to pre-allocate
data_length = 0
for loc_dir in param_dir.glob('*'):
    res = xr.open_dataset(param_dir.joinpath(f'{loc_dir.name}/{1.0}_{0.0}_{0.0}.nc'))
    sd_actual, sd_pred = get_bootstrap(res['lidar-sd'], res['snow_depth'])
    data_length += len(sd_actual)

param_dir = Path('~/scratch/params').expanduser()
new_param_dir = Path('~/scratch/params_np').expanduser()

# Create parameter space
A = np.round(np.arange(1, 3.1, 0.5), 2)
B = np.round(np.arange(0, 1.01, 0.1), 2)
C = np.round(np.arange(0, 1.001, 0.01), 2)
ABC = [A, B, C]
locs = list(param_dir.glob('*'))
DSs = {}
for a, b, c in product(*ABC):
    lidar = np.empty(data_length, dtype = float)
    spicy = np.empty(data_length, dtype = float) 
    s_idx = 0
    for loc_dir in locs:
        ds = xr.open_dataset(param_dir.joinpath(f'{loc_dir.name}/{a}_{b}_{c}.nc'))
        lidar_sd, spicy_sd = get_numpy(ds['lidar-sd'], ds['snow_depth'])
        e_idx = s_idx + len(lidar_sd)
        lidar[s_idx: e_idx] = lidar_sd
        spicy[s_idx: e_idx] = spicy_sd
        s_idx = e_idx
    param_np = np.vstack([lidar, spicy])
    np.save(new_param_dir.joinpath(f'{a}_{b}_{c}.npy'), param_np)
