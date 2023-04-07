import pickle
import shapely
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxa
import matplotlib.pyplot as plt

for fp in Path('./rmse_insitu').glob('*no_flag.nc'):
    fp_lidar = next(Path('./rmse_out').glob(fp.name))
    da = xr.open_dataarray(fp)
    da2 = xr.open_dataarray(fp_lidar) * 100
    insitu_best = da.where(da==da.min(), drop=True).squeeze()
    lidar_best = da2.where(da2==da2.min(), drop=True).squeeze()

    fig, axes = plt.subplots(3, 2, figsize = (12, 8))
    plt.suptitle(f'{fp.stem} RMSE Comparison')

    for i, param in enumerate(['A', 'B', 'C']):
        vmin, vmax = da.reduce(np.min, dim = [param]).quantile([.2, .8])
        vmin2, vmax2 = da2.reduce(np.min, dim = [param]).quantile([.2, .8])
        vmin = np.min([vmin, vmin2])
        vmax = np.max([vmax, vmax2])
        da.reduce(np.min, dim = [param]).plot(ax = axes[i, 0]) # vmin = vmin, vmax = vmax
        da2.reduce(np.min, dim = [param]).plot(ax = axes[i, 1]) # vmin = vmin, vmax = vmax
        axes[i, 0].set_title(f'In-Situ {param} @ {np.round(insitu_best[param].data, 2)}', fontweight='bold')
        axes[i, 1].set_title(f'Lidar {param} @ {np.round(lidar_best[param].data, 2)}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'../../images/param-compare/{fp.stem}.png')