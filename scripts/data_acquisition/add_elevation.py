import xarray as xr
from pathlib import Path
import py3dep
import rioxarray as rxa
files = Path('/bsuhome/zacharykeskinen/spicy-snow/SnowEx-Data').glob('*.nc')
for fp in files:
    ds = xr.load_dataset(fp)
    dem = py3dep.get_dem(ds.rio.bounds(), 30)
    ds['dem'] = dem.drop(['band', 'spatial_ref']).interp_like(ds)
    ds.to_netcdf(fp.with_suffix('.fix.nc'))