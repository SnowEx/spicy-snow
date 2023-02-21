import pickle

import shapely

# Add main repo to path
import sys
import os
from os.path import expanduser
sys.path.append(expanduser('../'))

# import modules
from spicy_snow.download.sentinel1 import s1_img_search, hyp3_pipeline, download_hyp3, combine_s1_images
from spicy_snow.download.forest_cover import download_fcf
from spicy_snow.download.snow_cover import download_snow_cover

# sets dates and areas
dates = ('2019-08-01', '2019-12-30')
area = shapely.geometry.box(-115, 43, -114, 44)

# Set skip = True to skip indented steps
skip = False

if not skip:
    os.makedirs('../data' , exist_ok = True)

    # get asf_search search results
    search_results = s1_img_search(area, dates)
    print(f'Found {len(search_results)} results')

    # download s1 images into dataset ['s1'] keyword
    jobs = hyp3_pipeline(search_results, job_name = '2019-2020')
    imgs = download_hyp3(jobs, area, outdir = '../data/tmp', clean = False)
    ds = combine_s1_images(imgs)

    #slice first 50
    # ds = ds.sel(time = slice(50))
    # download IMS snow cover and add to dataset ['ims'] keyword
    ds = download_snow_cover(ds, tmp_dir = '../data/tmp/', clean = False)

    # download fcf and add to dataset ['fcf'] keyword
    fcf = download_fcf(ds, '../data/fcf.tif')

    # dump completed dataset to data directory

    with open('../data/main_test.pkl', 'wb') as f:
        pickle.dump(ds, f)

    from spicy_snow.processing.s1_preprocessing import merge_partial_s1_images, s1_orbit_averaging,\
    s1_clip_outliers

    ds = merge_partial_s1_images(ds)

    ds = s1_orbit_averaging(ds)

    ds = s1_clip_outliers(ds)

# import the function to test
from spicy_snow.processing.snow_index import calc_delta_VV, calc_delta_cross_ratio, \
    calc_delta_gamma, clip_delta_gamma_outlier, calc_snow_index, calc_snow_index_to_snow_depth

print('calculating CR')
calc_delta_cross_ratio(ds, inplace = True)
print('calculating delta VV')
calc_delta_VV(ds, inplace = True)
print('calculating delta gamma')
calc_delta_gamma(ds, inplace = True)
clip_delta_gamma_outlier(ds, inplace = True)

print('saving processed dataset')
# If you want to load dataset for testing use this code:
with open('../data/main_test_proc.pkl', 'wb') as f:
        pickle.dump(ds, f)

ds = calc_snow_index(ds)
ds = calc_snow_index_to_snow_depth(ds)

print('saving processed dataset')
# If you want to load dataset for testing use this code:
with open('../data/main_test_proc.pkl', 'wb') as f:
        pickle.dump(ds, f)