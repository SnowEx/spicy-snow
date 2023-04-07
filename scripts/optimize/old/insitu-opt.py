# import numpy as np
# import pandas as pd
# import geopandas as gpd
# import xarray as xr
# import matplotlib.pyplot as plt
# from shapely import wkt
# from shapely.geometry import box
# from pathlib import Path
# from datetime import datetime
# from scipy.stats import pearsonr
# from sklearn.metrics import mean_squared_error

# import sys
# sys.path.append('../..')

# from spicy_snow.processing.snow_index import calc_delta_cross_ratio, calc_delta_gamma, \
#     clip_delta_gamma_outlier, calc_snow_index, calc_snow_index_to_snow_depth
# from spicy_snow.processing.wet_snow import id_newly_wet_snow, id_wet_negative_si, \
#     id_newly_frozen_snow, flag_wet_snow

# def get_stats(df):
#     r, p = pearsonr(df.depth, df.spicy)
#     rmse = mean_squared_error(df.depth, df.spicy, squared=False)
#     return r, rmse, len(df)

# # Create parameter space
# A = np.arange(1, 3.1, 0.5)
# B = np.arange(0, 1.01, 0.1)
# C = np.arange(0, 1.001, 0.01)

# files = Path('../../Lidar_s1_stacks/').glob('*.nc')
# for f in files:

#     # get dataset
#     ds_name = f.name.split('stacks/')[-1].split('.')[0]
#     print(datetime.now(), f' -- starting {ds_name}')
#     ds_ = xr.open_dataset(f).load()
#     dataset = ds_[['s1','deltaVV','ims','fcf','lidar-sd']]

#     # find closest timestep to lidar
#     td = abs(pd.to_datetime(dataset.time) - pd.to_datetime(dataset.attrs['lidar-flight-time']))
#     closest_ts = dataset.time[np.argmin(td)]
    
#     if Path(f'rmse_insitu/{ds_name}_wet_flag.nc').exists():
#         continue

#     # Brute-force processing loop
#     for a in A:
#         print(f'A: {a}')
#         ds = calc_delta_cross_ratio(dataset, A = a)
#         for b in B:
#             print(f'    B: {b}')
#             ds = calc_delta_gamma(ds, B = b, inplace=False)
#             ds = clip_delta_gamma_outlier(ds)
#             ds = calc_snow_index(ds)
#             ds = id_newly_wet_snow(ds)
#             ds = id_wet_negative_si(ds)
#             ds = id_newly_frozen_snow(ds)
#             ds = flag_wet_snow(ds)
#             for c in C:
#                 print(f'        c: {c}')
#                 # print(f'A={a}; B={b}; C={c}')

#                 ds = calc_snow_index_to_snow_depth(ds, C = c)

#                 # Compare insitu snow depths - mask wet snow
#                 # tolerance around each site 100 m\n",
#                 tol = 0.00090009

#                 insitu_loc['spicy'] = np.nan
#                 insitu_loc['wet'] = np.nan
#                 for date in insitu_loc.date.unique():
#                     insitu_ts = insitu_loc.loc[insitu_loc.date == date]

#                     date = pd.to_datetime(date)
#                     ds_ts = ds.sel(time = slice(date - pd.Timedelta('2 days'), date + pd.Timedelta('2 days')))

#                     if len(ds_ts.time) > 0:
#                         for i, r in insitu_ts.iterrows():
#                             insitu_loc.loc[i, 'spicy'] = ds_ts.mean(dim = 'time')['snow_depth'].sel(x = slice(r.geometry.x - tol, r.geometry.x + tol), y = slice(r.geometry.y + tol, r.geometry.y - tol)).mean() * 100
#                             insitu_loc.loc[i, 'wet'] = ds_ts.isel(time = 0)['wet_snow'].sel(x = r.geometry.x, y = r.geometry.y, method = 'nearest')

#                 insitu_loc = insitu_loc.dropna(subset = ['depth', 'spicy'])
                
#                 if len(insitu_loc) > 2:
#                     r, rmse, n = get_stats(insitu_loc)
#                 else:
#                     rmse = np.nan
#                 rmse_no_flag.loc[a, b, c] = rmse

#                 wet_insitu = insitu_loc[insitu_loc.wet == 0]
#                 if len(wet_insitu) > 1:
#                     r_wet, rmse_wet, n_wet = get_stats(wet_insitu)
#                 else:
#                     rmse_wet = np.nan
#                 rmse_wet_flag.loc[a, b, c] = rmse_wet

#                 print(f'        rmse: {rmse}')
#                 print(f'        dry rmse: {rmse_wet}')
    
#     # After loop, save RMSE results per file
#     rmse_wet_flag.to_netcdf(f'rmse_insitu/{ds_name}_wet_flag.nc')
#     rmse_no_flag.to_netcdf(f'rmse_insitu/{ds_name}_no_flag.nc')