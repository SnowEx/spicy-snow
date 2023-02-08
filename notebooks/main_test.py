import pickle

import shapely

import sys
from os.path import expanduser
sys.path.append(expanduser('~/Documents/spicy-snow'))

from spicy_snow.download.sentinel1 import s1_img_search, download_s1_imgs
from spicy_snow.download.forest_cover import download_fcf, add_fcf
from spicy_snow.download.snow_cover import download_snow_cover

dates = ('2019-08-01', '2020-03-30')
area = shapely.geometry.box(-115, 43, -114, 44)


skip = True

if not skip:
    search_results = s1_img_search(area, dates)

    print(f'Found {len(search_results)} results')

    ds = download_s1_imgs(search_results, area, tmp_dir = '/Users/zachkeskinen/Documents/spicy-snow/data/tmp', job_name = '2019-2020', existing_job_name = False)

    # with open('/Users/zachkeskinen/Documents/spicy-snow/data/s1_dt.pkl', 'wb') as f:
    #     pickle.dump(ds, f)

    ds = download_snow_cover(ds, tmp_dir = '/Users/zachkeskinen/Documents/spicy-snow/data/tmp/', clean = False)

    # with open('/Users/zachkeskinen/Documents/spicy-snow/data/s1_ds.pkl', 'wb') as f:
    #     pickle.dump(ds, f)

    fcf = download_fcf('/Users/zachkeskinen/Documents/spicy-snow/data/fcf.tif')
    ds = add_fcf(ds, fcf)

with open('/Users/zachkeskinen/Documents/spicy-snow/data/all_2019_2020.pkl', 'rb') as f:
    ds = pickle.load(f)

print(ds)