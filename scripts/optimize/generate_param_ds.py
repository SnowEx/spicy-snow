import numpy as np
import pandas as pd
import geopandas as gpd
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

def get_stats(df):
    r, p = pearsonr(df.depth, df.spicy)
    rmse = mean_squared_error(df.depth, df.spicy, squared=False)
    return r, rmse, len(df)

# Create parameter space
A = np.arange(1, 3.1, 0.5)
B = np.arange(0, 1.01, 0.1)
C = np.arange(0, 1.001, 0.01)

files = Path('../../Lidar_s1_stacks/').glob('*.nc')
for f in files:

    # get dataset
    ds_name = f.name.split('stacks/')[-1].split('.')[0]
    print(datetime.now(), f' -- starting {ds_name}')
    ds_ = xr.open_dataset(f).load()
    dataset = ds_[['s1','deltaVV','ims','fcf','lidar-sd']]

    # find closest timestep to lidar
    td = abs(pd.to_datetime(dataset.time) - pd.to_datetime(dataset.attrs['lidar-flight-time']))
    closest_ts = dataset.time[np.argmin(td)]

    Path(f'./param_sds/{ds_name}/').mkdir(exist_ok = True, parents=True)

    # Brute-force processing loop
    for a in tqdm(A):
        # print(f'A: {a}')
        a = np.round(a, 2)
        ds = calc_delta_cross_ratio(dataset, A = a)
        for b in B:
            b = np.round(b, 2)
            # print(f'    B: {b}')
            ds = calc_delta_gamma(ds, B = b, inplace=False)
            ds = clip_delta_gamma_outlier(ds)
            ds = calc_snow_index(ds)
            ds = id_newly_wet_snow(ds)
            ds = id_wet_negative_si(ds)
            ds = id_newly_frozen_snow(ds)
            ds = flag_wet_snow(ds)
            for c in C:
                c = np.round(c, 2)
                # print(f'        c: {c}')
                # print(f'A={a}; B={b}; C={c}')

                ds = calc_snow_index_to_snow_depth(ds, C = c)

                sub = ds.sel(time = closest_ts)[['snow_depth', 'wet_snow', 'lidar-sd']]
                sub.to_netcdf(f'./param_sds/{ds_name}/{a}_{b}_{c}.nc')