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
    sd_actual, sd_pred = get_numpy(res['lidar-sd'], res['snow_depth'])
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
    if new_param_dir.joinpath(f'{a}_{b}_{c}.npy').exists():
        continue
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