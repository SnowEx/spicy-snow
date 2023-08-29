import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error

from itertools import product
# from tqdm.contrib.itertools import product

def get_stats(a, b):
    r, p = pearsonr(a, b)
    error = mae(a, b)
    rmse = mean_squared_error(a, b, squared=False)
    return r, error, rmse

res_fp = Path('/bsuhome/zacharykeskinen/spicy-snow/data/res_ds_iter_large.nc')

if res_fp.exists():
    print('Already exists...')
    res_ds = xr.load_dataset(res_fp)

else:
    param_fp = Path('/bsuhome/zacharykeskinen/scratch/param_regional')

    locs = list(param_fp.glob('*'))
    locs = [l.stem for l in locs]
    # Create parameter space
    A = np.round(np.arange(1, 3.1, 0.5), 2)
    B = np.round(np.arange(0, 1.01, 0.1), 2)
    C = np.round(np.arange(0.01, 1.001, 0.01), 2)
    iterations = np.arange(500)
    res = np.zeros((len(locs), len(A), len(B), len(C), len(iterations)))


    da = xr.DataArray(res, coords = [locs, A, B, C, iterations], dims = ['location', 'A', 'B','C', 'iteration'], name = 'pearsonr')
    res_ds = xr.merge([da, da.copy().rename('mae'), da.copy().rename('rmse')])

    for loc_fp in param_fp.glob('*'):
        print(loc_fp)
        lidar_orig = np.load(loc_fp.joinpath('lidar.npy'))
        for a, b, c in product(A, B, C):
            sds_orig = np.load(loc_fp.joinpath(f'{a}_{b}_{c}.npy'))
            combo = np.vstack([lidar_orig, sds_orig])
            for iter in iterations:
                idx = np.random.choice(combo.shape[1], combo.shape[1], replace = True)
                sds, lidar = combo.T[idx].T
                r, mean_error, rmse = get_stats(lidar, sds)
                res_ds['pearsonr'].loc[dict(location = loc_fp.stem, A = a, B = b, C = c, iteration = iter)] = r
                res_ds['mae'].loc[dict(location = loc_fp.stem, A = a, B = b, C = c, iteration = iter)] = mean_error
                res_ds['rmse'].loc[dict(location = loc_fp.stem, A = a, B = b, C = c, iteration = iter)] = rmse
    res_ds.to_netcdf(res_fp)