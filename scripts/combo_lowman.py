from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxa

quad_dir = Path('~/scratch/spicy-lowman-quadrant').expanduser()
quad_dir.exists()

DAs = []
for fp in quad_dir.glob('*.nc'):
    if '.un.nc' not in fp.name:
        ds = xr.open_dataset(fp)
        ds = ds['snow_depth']
        ds = ds.sel(time = slice('2020-11-01', '2021-03-01'))
        ds = ds.reindex(lat=list(reversed(ds.y)))
        DAs.append(ds)

res = xr.combine_by_coords(DAs, join = 'override')
res.to_netcdf(quad_dir.joinpath('lowman_combo.nc'))