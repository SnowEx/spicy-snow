# from itertools import product
from tqdm.contrib.itertools import product
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

def get_bootstrap(x, y):
    # ravel to numpy arrays
    x = x.values.ravel()
    y = y.values.ravel()

    # remove nans
    x_buff = x[(~np.isnan(x)) & (~np.isnan(y))]
    y = y[(~np.isnan(x)) & (~np.isnan(y))]
    x = x_buff

    # bootstrap
    x_bs = np.random.choice(x, size = len(x))
    y_bs = np.random.choice(y, size = len(y))

    return x_bs, y_bs

def calc_rmse(y_actual, y_pred):
    rms = mean_squared_error(y_actual, y_pred, squared = False)
    return rms

# Create parameter space
A = np.round(np.arange(1, 3.1, 0.5), 2)
B = np.round(np.arange(0, 1.01, 0.1), 2)
C = np.round(np.arange(0, 1.001, 0.01), 2)
ABC = [A, B, C]

param_dir = Path('~/scratch/params').expanduser()

df = pd.DataFrame(np.empty((10, 4), dtype = float), columns = ['a', 'b', 'c', 'rmse'])

for i in range(10):
    print(i)
    np.random.seed(i)

    rmse_no_flag = xr.DataArray(np.empty((len(A), len(B), len(C)))*np.nan,
                            coords=(A, B, C), dims=('A','B','C'))

    for a, b, c in product(*ABC):
        lidar = []
        spicy = []
        for loc_dir in param_dir.glob('*'):
            res = xr.open_dataset(param_dir.joinpath(f'{loc_dir.name}/{a}_{b}_{c}.nc'))
            sd_actual, sd_pred = get_bootstrap(res['lidar-sd'], res['snow_depth'])
            lidar.extend(sd_actual)
            spicy.extend(sd_pred)
            
        rmse_no_flag.loc[a, b, c] = calc_rmse(lidar, spicy)

    best = rmse_no_flag.where(rmse_no_flag==rmse_no_flag.min(), drop=True).squeeze()
    a, b, c = best.coords.values()
    df.loc[i, 'a'] = a
    df.loc[i, 'b'] = b
    df.loc[i, 'c'] = c
    df.loc[i, 'rmse'] = best

df.to_csv('/bsuhome/zacharykeskinen/spicy-snow/scripts/optimize/param_pdf/pdf_v1.csv')

# FOR GENERATING THE PARAMETER DATASETS! #

# import numpy as np
# import pandas as pd
# import xarray as xr
# import matplotlib.pyplot as plt
# from shapely import wkt
# from shapely.geometry import box
# from pathlib import Path
# from datetime import datetime
# from scipy.stats import pearsonr
# from sklearn.metrics import mean_squared_error
# from tqdm import tqdm

# import sys
# sys.path.append('../..')

# from spicy_snow.processing.snow_index import calc_delta_cross_ratio, calc_delta_gamma, \
#     clip_delta_gamma_outlier, calc_snow_index, calc_snow_index_to_snow_depth
# from spicy_snow.processing.wet_snow import id_newly_wet_snow, id_wet_negative_si, \
#     id_newly_frozen_snow, flag_wet_snow

# # Create parameter space
# A = np.round(np.arange(1, 3.1, 0.5), 2)
# B = np.round(np.arange(0, 1.01, 0.1), 2)
# C = np.round(np.arange(0, 1.001, 0.01), 2)

# files = Path('../../Lidar_s1_stacks/').glob('*.nc')
# param_dir = Path('~/scratch/params').expanduser()
# for f in files:

#     # get dataset
#     ds_name = f.name.split('stacks/')[-1].split('.')[0]
#     print(datetime.now(), f' -- starting {ds_name}')
#     ds_ = xr.open_dataset(f).load()
#     dataset = ds_[['s1','deltaVV','ims','fcf','lidar-sd']]

#     # find closest timestep to lidar
#     td = abs(pd.to_datetime(dataset.time) - pd.to_datetime(dataset.attrs['lidar-flight-time']))
#     closest_ts = dataset.time[np.argmin(td)]

#     param_dir.joinpath(f'{ds_name}').mkdir()

#     # Brute-force processing loop
#     for a in tqdm(A):
#         # print(f'A: {a}')
#         ds = calc_delta_cross_ratio(dataset, A = a)
#         for b in B:
#             # print(f'    B: {b}')
#             ds = calc_delta_gamma(ds, B = b, inplace=False)
#             ds = clip_delta_gamma_outlier(ds)
#             ds = calc_snow_index(ds)
#             ds = id_newly_wet_snow(ds)
#             ds = id_wet_negative_si(ds)
#             ds = id_newly_frozen_snow(ds)
#             ds = flag_wet_snow(ds)
#             for c in C:
#                 # print(f'        c: {c}')
#                 # print(f'A={a}; B={b}; C={c}')

#                 ds = calc_snow_index_to_snow_depth(ds, C = c)

#                 sub = ds.sel(time = closest_ts)[['snow_depth', 'wet_snow', 'lidar-sd']]
#                 sub.to_netcdf(param_dir.joinpath(f'{ds_name}/{a}_{b}_{c}.nc'))