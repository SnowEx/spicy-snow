import numpy as np
import time
from pathlib import Path

import sys
project_root = Path(__file__).resolve().parent.parent.parent  # ../../.. from script
sys.path.insert(0, str(project_root))

from spicy_snow.find_data import get_sentinel1_urls
from spicy_snow.utils.download import download_urls_parallel
from spicy_snow.legacy.hyp3_rtc_download import s1_hyp3_search, hyp3_pipeline, download_hyp3, combine_s1_images

test_dates = ('2024-10-01', '2025-11-01')
test_aoi = [-114.5, 43, -114, 44]

# -- Run opera -- #
start_time = time.perf_counter()
work_dir = Path('/Users/zmhoppinen/Documents/spicy-snow/local')
s1_urls = get_sentinel1_urls(start_date = test_dates[0], stop_date = test_dates[1], aoi = test_aoi)
s1_urls = [u for u in s1_urls if u.endswith(('_VV.tif', '_VH.tif', '_mask.tif'))]
s1_fps = download_urls_parallel(s1_urls, work_dir.joinpath('opera'))

# End timer
end_time = time.perf_counter()
# Compute elapsed time
elapsed = end_time - start_time
minutes = int(elapsed // 60)
seconds = elapsed % 60
print(f"Elapsed time for opera to do 1 month: {minutes} min {seconds:.2f} sec")

# -- Run hyp3 -- #
start_time = time.perf_counter()
job_name = 'speed_test'
# get asf_search search results
search_results = s1_hyp3_search(test_aoi, test_dates)

# download s1 images into dataset ['s1'] variable name
jobs = hyp3_pipeline(search_results, job_name = job_name, existing_job_name = job_name)
imgs = download_hyp3(jobs, test_aoi, outdir = work_dir.joinpath('hyp3'), clean = False)
ds = combine_s1_images(imgs)

# End timer
end_time = time.perf_counter()
# Compute elapsed time
elapsed = end_time - start_time
minutes = int(elapsed // 60)
seconds = elapsed % 60
print(f"Elapsed time for hyp3 to do 1 month: {minutes} min {seconds:.2f} sec")

