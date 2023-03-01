import pickle

import shapely

# Add main repo to path
import sys
import os
from os.path import expanduser
sys.path.append(expanduser('../'))

from spicy_snow.retrieval import retrieve_snow_depth

# sets dates and areas
dates = ('2019-08-01', '2019-12-30')
area = shapely.geometry.box(-115, 43, -114, 44)

ds = retrieve_snow_depth(area, dates, work_dir = '../data', existing_job_name = 'spicy-snow-run')

# dump completed dataset to data directory
with open('../data/bundle_test.pkl', 'wb') as f:
    pickle.dump(ds, f)