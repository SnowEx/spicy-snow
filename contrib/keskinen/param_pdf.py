# from itertools import product
from tqdm.contrib.itertools import product
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import time

# multiprocessing
# https://superfastpython.com/multiprocessing-for-loop/
from multiprocessing import Process

def get_bootstrap(x, y):
    # ravel to numpy arrays
    x = x.values.ravel()
    y = y.values.ravel()

    # remove nans
    x_buff = x[(~np.isnan(x)) & (~np.isnan(y))]
    y = y[(~np.isnan(x)) & (~np.isnan(y))]
    x = x_buff

    # bootstrap
    np.random.seed(int(time.time()))
    x_bs = np.random.choice(x, size = len(x))
    y_bs = np.random.choice(y, size = len(y))

    return x_bs, y_bs

def calc_rmse(y_actual, y_pred):
    rms = mean_squared_error(y_actual, y_pred, squared = False)
    return rms

param_dir = Path('~/scratch/params').expanduser()
# find length of lidar, spicy arrays to pre-allocate
data_length = 0
for loc_dir in param_dir.glob('*'):
    res = xr.open_dataset(param_dir.joinpath(f'{loc_dir.name}/{1.0}_{0.0}_{0.0}.nc'))
    sd_actual, sd_pred = get_bootstrap(res['lidar-sd'], res['snow_depth'])
    data_length += len(sd_actual)

# Create parameter space
A = np.round(np.arange(1, 3.1, 0.5), 2)
B = np.round(np.arange(0, 1.01, 0.1), 2)
C = np.round(np.arange(0, 1.001, 0.01), 2)
ABC = [A, B, C]

df = pd.DataFrame(np.empty((3, 4), dtype = float), columns = ['a', 'b', 'c', 'rmse'])

for i in range(3):
    rmse_no_flag = xr.DataArray(np.empty((len(A), len(B), len(C)))*np.nan,
                            coords=(A, B, C), dims=('A','B','C'))

    lidar = np.empty(data_length, dtype = float)
    spicy = np.empty(data_length, dtype = float)

    for a, b, c in product(*ABC):
        s_idx = 0
        for loc_dir in param_dir.glob('*'):
            res = xr.open_dataset(param_dir.joinpath(f'{loc_dir.name}/{a}_{b}_{c}.nc'))
            sd_actual, sd_pred = get_bootstrap(res['lidar-sd'], res['snow_depth'])
            e_idx = s_idx + len(sd_actual)
            lidar[s_idx: e_idx] = sd_actual
            spicy[s_idx: e_idx] = sd_pred
            s_idx = e_idx
            
        rmse_no_flag.loc[a, b, c] = calc_rmse(lidar, spicy)

    best = rmse_no_flag.where(rmse_no_flag==rmse_no_flag.min(), drop=True).squeeze()
    a, b, c = best.coords.values()
    df.loc[i, 'a'] = a
    df.loc[i, 'b'] = b
    df.loc[i, 'c'] = c
    df.loc[i, 'rmse'] = best.values

v_num = 1
df.to_csv(f'/bsuhome/zacharykeskinen/spicy-snow/scripts/optimize/param_pdfs_v2/pdf_v{v_num}.csv')