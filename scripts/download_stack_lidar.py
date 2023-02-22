import pickle
import shapely

# Add main repo to path
import sys
from os.path import expanduser
sys.path.append(expanduser('../'))

from spicy_snow.retrieval import retrieve_snow_depth

from spicy_snow.download.snowex_lidar import download_dem, download_snow_depth, download_veg_height

lidar_dir = '../data/lidar'
download_snow_depth(lidar_dir)
download_veg_height(lidar_dir)
download_dem(lidar_dir)

# sets dates and areas
# dates = ('2020-08-01', '2021-03-31')

# ds = retrieve_snow_depth(area, dates, work_dir = '../data', job_name = 'banner lidar', existing_job_name = 'banner lidar')

# dump completed dataset to data directory
# with open('../data/banner_lidar.pkl', 'wb') as f:
#     pickle.dump(ds, f)