from itertools import product
# from tqdm.contrib.itertools import product
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

if __name__ == '__main__':
    pass

# def get_params(v_num):

#     param_dir = Path('~/scratch/params').expanduser()
#     # Create parameter space
#     A = np.round(np.arange(1, 3.1, 0.5), 2)
#     B = np.round(np.arange(0, 1.01, 0.1), 2)
#     C = np.round(np.arange(0, 1.001, 0.01), 2)
#     ABC = [A, B, C]

#     df = pd.DataFrame(np.empty((10, 4), dtype = float), columns = ['a', 'b', 'c', 'rmse'])

#     for i in range(10):

#         rmse_no_flag = xr.DataArray(np.empty((len(A), len(B), len(C)))*np.nan,
#                                 coords=(A, B, C), dims=('A','B','C'))

#         for a, b, c in product(*ABC):
#             lidar = []
#             spicy = []
#             for loc_dir in param_dir.glob('*'):
#                 res = xr.open_dataset(param_dir.joinpath(f'{loc_dir.name}/{a}_{b}_{c}.nc'))
#                 sd_actual, sd_pred = get_bootstrap(res['lidar-sd'], res['snow_depth'])
#                 lidar.extend(sd_actual)
#                 spicy.extend(sd_pred)
                
#             rmse_no_flag.loc[a, b, c] = calc_rmse(lidar, spicy)

#         best = rmse_no_flag.where(rmse_no_flag==rmse_no_flag.min(), drop=True).squeeze()
#         a, b, c = best.coords.values()
#         df.loc[i, 'a'] = a
#         df.loc[i, 'b'] = b
#         df.loc[i, 'c'] = c
#         df.loc[i, 'rmse'] = best

#     df.to_csv(f'/bsuhome/zacharykeskinen/spicy-snow/scripts/optimize/param_pdfs_v2/pdf_v{v_num}.csv')

# if __name__ == '__main__':
#     # create all tasks
#     processes = [Process(target=get_params, args=(i,)) for i in range(5)]
#     # start all processes
#     for process in processes:
#         process.start()
#     # wait for all processes to complete
#     for process in processes:
#         process.join()
#     # report that all tasks are completed
#     print('Done', flush=True)