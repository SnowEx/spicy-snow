import pickle
import shapely
from glob import glob
from os.path import join
import pandas as pd
import xarray as xr
import rioxarray as rxa

# Add main repo to path
import sys
from os.path import expanduser
sys.path.append(expanduser('../'))

from spicy_snow.retrieval import retrieve_snow_depth

from spicy_snow.download.snowex_lidar import download_dem, download_snow_depth,\
      download_veg_height, make_site_ds

lidar_dir = '../data/lidar'
download_snow_depth(lidar_dir)
download_veg_height(lidar_dir)
download_dem(lidar_dir)

sites = {'USCOCP': 'Cameron', 'USCOFR': 'Frasier', 'USIDBS': 'Banner', 'USIDDC': 'Dry_Creek',
         'USIDMC': 'Mores', 'USUTLC': 'Little_Cottonwood'}

for site, site_name in sites.items():

    lidar_ds = make_site_ds(site, lidar_dir = lidar_dir)

    lidar_ds = lidar_ds.where(lidar_ds < 1000).where(lidar_ds > -1000)

    area = shapely.geometry.box(*lidar_ds.rio.bounds())

    for date in lidar_ds.time:

        if date.dt.month > 5:
            continue

        if date.dt.month < 8:
            date1 = pd.to_datetime(f'{int(date.dt.year - 1)}-08-01')
        else:
            date1 = pd.to_datetime(f'{int(date.dt.year)}-08-01')

        dates = (date1.strftime('%Y-%m-%d'), pd.to_datetime((date + pd.Timedelta('14 day')).values).strftime('%Y-%m-%d'))

        spicy_ds = retrieve_snow_depth(area = area, dates = dates, work_dir = '../data/', job_name = f'spicy-{site}', existing_job_name = f'spicy-{site}')

        lidar_ds = lidar_ds.rio.reproject_match(spicy_ds)

        ds = xr.merge([spicy_ds, lidar_ds], combine_attrs = 'drop_conflicts')

        ds = ds[['lidar-sd', 'lidar-vh', 'lidar-dem', 'snow_depth', 's1', 'wet_snow']]

        with open(f'../SnowEx-Data/{site_name}_{(date).dt.strftime("%y-%m-%d").values}.pkl', 'wb') as f:
            pickle.dump(ds, f)