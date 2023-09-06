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
sys.path.append(Path('../..'))

from spicy_snow.processing.snow_index import calc_delta_cross_ratio, calc_delta_gamma, \
    clip_delta_gamma_outlier, calc_snow_index, calc_snow_index_to_snow_depth
from spicy_snow.processing.wet_snow import id_newly_wet_snow, id_wet_negative_si, \
    id_newly_frozen_snow, flag_wet_snow

def get_idx(x, y, z, q):
    # ravel to numpy arrays
    x = x.values.ravel()
    y = y.values.ravel()
    z = z.values.ravel()
    q = q.values.ravel()

    # find non nans
    id = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(z)) & (~np.isnan(q))
    return id

# Create parameter space
A = np.round(np.arange(1, 3.1, 0.5), 2)
B = np.round(np.arange(0, 1.01, 0.1), 2)
C = np.round(np.arange(0, 1.001, 0.01), 2)

files = Path('/bsuhome/zacharykeskinen/spicy-snow/Lidar_s1_stacks/').glob('*fix.nc')

param_dir = Path('~/scratch/param_regional_v2').expanduser()
param_dir.mkdir(exist_ok = True)
for f in files:
    # get dataset
    ds_name = f.name.split('stacks/')[-1].split('.')[0]
    print(datetime.now(), f' -- starting {ds_name}')
    ds_ = xr.open_dataset(f).load()
    dataset = ds_[['s1','deltaVV','ims','fcf', 'lidar-sd', 'dem']]

    # find closest timestep to lidar
    td = abs(pd.to_datetime(dataset.time) - pd.to_datetime(dataset.attrs['lidar-flight-time']))
    closest_ts = dataset.time[np.argmin(td)]

    if 'Frasier_2020-02-11.nc' in f.name:
        closest_ts = '2020-02-16T13:09:43'

    param_dir.joinpath(ds_name).mkdir(exist_ok = True)

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

                sub = ds.sel(time = closest_ts)[['snow_depth', 'lidar-sd', 'fcf', 'dem']]
                id = get_idx(sub['snow_depth'], sub['lidar-sd'], sub['fcf'], sub['dem'])
                spicy_sd = sub['snow_depth'].values.ravel()[id]
                # param_np = np.vstack([spicy_sd, spicy_wet])
                np.save(param_dir.joinpath(ds_name, f'{a}_{b}_{c}.npy'), spicy_sd)

                if not param_dir.joinpath(ds_name, 'lidar.npy').exists():
                    lidar_sd = sub['lidar-sd'].values.ravel()[id]
                    np.save(param_dir.joinpath(ds_name, f'lidar.npy'), lidar_sd)
                    fcf = sub['fcf'].values.ravel()[id]
                    np.save(param_dir.joinpath(ds_name, f'fcf.npy'), fcf)
                    dem = sub['dem'].values.ravel()[id]
                    np.save(param_dir.joinpath(ds_name, f'dem.npy'), dem)
