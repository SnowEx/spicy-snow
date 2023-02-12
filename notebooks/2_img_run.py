import pickle

import shapely

# Add main repo to path
import sys
import os
from os.path import expanduser
sys.path.append(expanduser('../'))

# import modules
from spicy_snow.download.sentinel1 import s1_img_search, download_s1_imgs
from spicy_snow.download.forest_cover import download_fcf, add_fcf
from spicy_snow.download.snow_cover import download_snow_cover

# sets dates and areas
dates = ('2019-02-01', '2019-02-25')
area = shapely.geometry.box(-115, 43, -114, 44)

# Set skip = True to skip indented steps
skip = False

if not skip:
    os.makedirs('../data' , exist_ok = True)
    # get asf_search search results
    search_results = s1_img_search(area, dates)
    print(f'Found {len(search_results)} results')

    # download s1 images into dataset ['s1'] keyword
    ds = download_s1_imgs(search_results, area, tmp_dir = '../data/tmp3', job_name = 'small-img-test', existing_job_name = 'small-img-test')

    # download IMS snow cover and add to dataset ['ims'] keyword
    ds = download_snow_cover(ds, tmp_dir = '../data/tmp/', clean = False)
    # download fcf and add to dataset ['fcf'] keyword
    fcf = download_fcf('../data/fcf.tif')
    ds = add_fcf(ds, fcf)

    # dump completed dataset to data directory

    with open('../data/2_img_test.pkl', 'wb') as f:
        ds = pickle.dump(ds, f)

# If you want to load dataset for testing use this code:
# with open('../data/main_test.pkl', 'rb') as f:
    #     ds = pickle.dump(f)
