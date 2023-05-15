import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error

out_dir = Path('/bsuhome/zacharykeskinen/spicy-snow/results/site_param_rmses/')

# get best for each site
res = pd.DataFrame()
regions = Path('~/scratch/param_regional').expanduser()
for region in regions.glob('*'):
    print(region.stem)
    lidar = np.load(next(region.glob('lidar.npy')))
    lowest_rmse = np.inf
    best = {'a': -1, 'b': -1, 'c': -1}
    for param_set in region.glob('*_*.npy'):
        a, b, c = param_set.stem.split('_')
        rmse = mean_squared_error(lidar, np.load(param_set), squared = False)
        if rmse < lowest_rmse:
            lowest_rmse = rmse
            best['a'], best['b'], best['c'] = a, b, c
    for k, v in best.items():
        res.loc[region.stem, k] = v
    res.loc[region.stem, 'rmse'] = lowest_rmse
res.to_csv(out_dir.joinpath('varying_all.csv'))

# get best holding A and B constant

res = pd.DataFrame()
for region in regions.glob('*'):
    print(region.stem)
    lidar = np.load(next(region.glob('lidar.npy')))
    lowest_rmse = np.inf
    best = {'a': -1, 'b': -1, 'c': -1}
    for param_set in region.glob('*_*.npy'):
        a, b, c = param_set.stem.split('_')
        if a != '1.0' or b != '1.0':
            continue
        rmse = mean_squared_error(lidar, np.load(param_set), squared = False)
        if rmse < lowest_rmse:
            lowest_rmse = rmse
            best['a'], best['b'], best['c'] = a, b, c
    for k, v in best.items():
        res.loc[region.stem, k] = v
    res.loc[region.stem, 'rmse'] = lowest_rmse
res.to_csv(out_dir.joinpath('varying_c.csv'))

# get best holding A constant

res = pd.DataFrame()
for region in regions.glob('*'):
    print(region.stem)
    lidar = np.load(next(region.glob('lidar.npy')))
    lowest_rmse = np.inf
    best = {'a': -1, 'b': -1, 'c': -1}
    for param_set in region.glob('*_*.npy'):
        a, b, c = param_set.stem.split('_')
        if a != '1.0':
            continue
        rmse = mean_squared_error(lidar, np.load(param_set), squared = False)
        if rmse < lowest_rmse:
            lowest_rmse = rmse
            best['a'], best['b'], best['c'] = a, b, c
    for k, v in best.items():
        res.loc[region.stem, k] = v
    res.loc[region.stem, 'rmse'] = lowest_rmse
res.to_csv(out_dir.joinpath('varying_b_c.csv'))

# get best holding B constant

res = pd.DataFrame()
for region in regions.glob('*'):
    print(region.stem)
    lidar = np.load(next(region.glob('lidar.npy')))
    lowest_rmse = np.inf
    best = {'a': -1, 'b': -1, 'c': -1}
    for param_set in region.glob('*_*.npy'):
        a, b, c = param_set.stem.split('_')
        if b != '1.0':
            continue
        rmse = mean_squared_error(lidar, np.load(param_set), squared = False)
        if rmse < lowest_rmse:
            lowest_rmse = rmse
            best['a'], best['b'], best['c'] = a, b, c
    for k, v in best.items():
        res.loc[region.stem, k] = v
    res.loc[region.stem, 'rmse'] = lowest_rmse
res.to_csv(out_dir.joinpath('varying_a_c.csv'))

# get rmse for leivens 2021 params

res = pd.DataFrame()
for region in regions.glob('*'):
    print(region.stem)
    lidar = np.load(next(region.glob('lidar.npy')))
    lowest_rmse = np.inf
    best = {'a': -1, 'b': -1, 'c': -1}
    for param_set in region.glob('*_*.npy'):
        a, b, c = param_set.stem.split('_')
        if a != '2.0' or b != '0.5' or c != '0.44':
            continue
        rmse = mean_squared_error(lidar, np.load(param_set), squared = False)
        if rmse < lowest_rmse:
            lowest_rmse = rmse
            best['a'], best['b'], best['c'] = a, b, c
    for k, v in best.items():
        res.loc[region.stem, k] = v
    res.loc[region.stem, 'rmse'] = lowest_rmse
res.to_csv(out_dir.joinpath('leivens_params.csv'))

# get rmse for when global best (all site optimation is used)

combo_nps = Path('~/scratch/params_np').expanduser()
global_best = {'a': -1, 'b': -1, 'c': -1}
lowest_rmse = np.inf
for param_set in combo_nps.glob('*'):
    a, b, c = param_set.stem.split('_')
    data = np.load(param_set)
    rmse = mean_squared_error(data[0, :], data[1, :] , squared = False)
    if rmse < lowest_rmse:
        lowest_rmse = rmse
        global_best['a'], global_best['b'], global_best['c'] = a, b, c

res = pd.DataFrame()
for region in regions.glob('*'):
    lidar = np.load(next(region.glob('lidar.npy')))
    lowest_rmse = np.inf
    best = {'a': -1, 'b': -1, 'c': -1}
    for param_set in region.glob('*_*.npy'):
        a, b, c = param_set.stem.split('_')
        if a != global_best['a'] or b != global_best['b'] or c != global_best['c']:
            continue
        rmse = mean_squared_error(lidar, np.load(param_set), squared = False)
        if rmse < lowest_rmse:
            lowest_rmse = rmse
            best['a'], best['b'], best['c'] = a, b, c
    for k, v in best.items():
        res.loc[region.stem, k] = v
    res.loc[region.stem, 'rmse'] = lowest_rmse
res.to_csv(out_dir.joinpath('all_sites.csv'))