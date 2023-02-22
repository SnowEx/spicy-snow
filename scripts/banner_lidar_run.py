import pickle

import shapely

# Add main repo to path
import sys
import os
from os.path import expanduser
sys.path.append(expanduser('../'))

from spicy_snow.retrieval import retrieve_snow_depth

# sets dates and areas
dates = ('2020-08-01', '2021-03-31')
area = shapely.geometry.box(-115.3, 44.0, -115.0, 44.5)

ds = retrieve_snow_depth(area, dates, work_dir = '../data', job_name = 'banner lidar', existing_job_name = 'banner lidar')

# dump completed dataset to data directory
with open('../data/banner_lidar.pkl', 'wb') as f:
    pickle.dump(ds, f)