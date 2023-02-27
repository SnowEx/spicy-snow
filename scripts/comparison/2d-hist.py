import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

fp = '../../SnowEx-Data'
lidar_dir = Path(fp)
fps = list(lidar_dir.glob('*.nc'))
fps = [f for f in fps if '.sub.nc' not in str(f)]
fps = [f for f in fps if '.old.nc' not in str(f)]

xs = []
ys = []

for fp in fps:
    ds = xr.open_dataset(fp)
    try:
        ft = pd.to_datetime(ds.attrs['lidar-flight-time'])
    except KeyError:

        ft = pd.to_datetime(fp.stem.split('_')[1].strip('.old'))

    dt = pd.Timedelta('3 days')
    loc = fp.stem.split('_')[0]
    if loc == 'Little':
        loc = 'Little Cottonwood'

    lidar_sd = ds['lidar-sd']
    
    xs.append(ds['lidar-sd'].values.ravel())
    ys.append(ds['snow_depth'].sel(time = slice(ft - dt, ft + dt)).where(~lidar_sd.isnull()).mean(dim = 'time').values.ravel())

# Frasier now
ds = xr.open_dataset('../../SnowEx-Data/Frasier_2021-03-19.old.nc')
xs.append(ds['lidar-sd'].isel(time= 56).values.ravel())
ys.append(ds['snow_depth'].isel(time = 59).where(~ds['lidar-sd'].isel(time= 56).isnull()).values.ravel())

ds = xr.open_dataset('/Users/zachkeskinen/Documents/spicy-snow/SnowEx-Data/Frasier_2020-02-11.old.nc')
xs.append(ds['lidar-sd'].values.ravel())
ys.append(ds['snow_depth'].isel(time = 48).where(~ds['lidar-sd'].isnull()).values.ravel())

# stack arrays
xs = np.hstack(xs)
ys = np.hstack(ys)

xs_tmp = xs[~np.isnan(xs) & ~np.isnan(ys)]
ys = ys[~np.isnan(xs) & ~np.isnan(ys)]
xs = xs_tmp

plt.figure(figsize = (6, 6))
plt.hist2d(xs, ys, bins = 150)
plt.xlim(0, 4)
plt.ylim(0,4)
plt.xlabel('Lidar Snow Depth')
plt.ylabel('Spicy Snow Depth')
plt.show()

from sklearn.metrics import mean_squared_error
rms = mean_squared_error(xs, ys, squared=False)

from scipy.stats import pearsonr
r, p = pearsonr(xs, ys)

print(f'RMSE: {rms}, pearson r: {r}')