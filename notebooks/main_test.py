import pickle

import shapely

import sys
from os.path import expanduser
sys.path.append(expanduser('~/Documents/spicy-snow'))

from spicy_snow.download.sentinel1 import s1_img_search, download_s1_imgs
from spicy_snow.download.forest_cover import download_fcf, add_fcf
from spicy_snow.download.snow_cover import download_snow_cover

dates = ('2019-10-01', '2020-04-01')
area = shapely.geometry.box(-114.4, 43, -114.3, 43.1)


skip = False

if not skip:
    search_results = s1_img_search(area, dates)
    print(f'Found {len(search_results)} results')

    ds = download_s1_imgs(search_results, area, tmp_dir = '/Users/zachkeskinen/Documents/spicy-snow/data/tmp', job_name = 'idaho_snow_test')

    
download_snow_cover(ds, tmp_dir = '/Users/zachkeskinen/Documents/spicy-snow/data/tmp/', clean = False)


with open('/Users/zachkeskinen/Documents/spicy-snow/data/s1_ds.pkl', 'wb') as f:
    pickle.dump(ds, f)

fcf = download_fcf('/Users/zachkeskinen/Documents/spicy-snow/data/fcf.tif')
ds = add_fcf(ds, fcf)

with open('/Users/zachkeskinen/Documents/spicy-snow/data/all_ds.pkl', 'wb') as f:
    pickle.dump(ds, f)
    
