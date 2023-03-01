import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

fp = '../../SnowEx-Data'
lidar_dir = Path(fp)
fps = list(lidar_dir.glob('*.nc'))
fps = [f for f in fps if '.sub.nc' not in str(f)]
fps = [f for f in fps if '.old.nc' not in str(f)]

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
    f, axes = plt.subplots(1, 2, figsize = (12, 8))

    lidar_sd = ds['lidar-sd']
    
    lidar_sd.plot(ax = axes[0], vmin = 0, vmax = 4)
    axes[0].set_title(f'{loc} Lidar Snow Depth')
    if '.old.nc' not in str(fp):
        ds['snow_depth'].sel(time = slice(ft - dt, ft + dt)).where(~lidar_sd.isnull()).mean(dim = 'time').plot(ax = axes[1], vmin = 0, vmax = 4)
    axes[1].set_title(f'{loc} Spicy Snow Depth')
    plt.savefig(f"../../images/lidar-spicy-plots/{loc}_{ft.strftime('%Y-%m-%d')}")

# do frasiers

ds = xr.open_dataset('../../SnowEx-Data/Frasier_2021-03-19.old.nc')
fig, axes = plt.subplots(1, 2, figsize = (12,8))
ds['lidar-sd'].isel(time= 56).plot(ax = axes[0], vmin = 0, vmax = 4)
axes[0].set_title('Frasier Lidar Snow Depth')
ds['snow_depth'].isel(time = 59).where(~ds['lidar-sd'].isel(time= 56).isnull()).plot(ax = axes[1], vmin = 0, vmax = 4)
axes[1].set_title('Frasier Spicy Snow Depth')
loc = 'Frasier'
ft = pd.to_datetime('2021-03-19')
plt.savefig(f"../../images/lidar-spicy-plots/lidar-spicy-plots/{loc}_{ft.strftime('%Y-%m-%d')}")

ds = xr.open_dataset('/Users/zachkeskinen/Documents/spicy-snow/SnowEx-Data/Frasier_2020-02-11.old.nc')
fig, axes = plt.subplots(1, 2, figsize = (12,8))
ds['lidar-sd'].plot(ax = axes[0], vmin = 0, vmax = 4)
axes[0].set_title('Frasier Lidar Snow Depth')
ds['snow_depth'].isel(time = 48).where(~ds['lidar-sd'].isnull()).plot(ax = axes[1], vmin = 0, vmax = 4)
axes[1].set_title(f'{loc} Spicy Snow Depth')
ft = pd.to_datetime('2020-02-11')
plt.savefig(f"../../images/lidar-spicy-plots/lidar-spicy-plots/{loc}_{ft.strftime('%Y-%m-%d')}")