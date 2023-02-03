import shapely
import os
os.environ['PROJ_LIB'] = '/Users/zachkeskinen/miniconda3/pkgs/proj-9.1.0-hf909084_1/share/proj'

import pyproj
pyproj.datadir.set_data_dir('/Users/zachkeskinen/miniconda3/pkgs/proj-9.1.0-hf909084_1/share/proj')
pyproj._pyproj_global_context_initialize()

import rioxarray as rio
img = rio.open_rasterio('/Users/zachkeskinen/Documents/spicy-snow/data/fcf.tif')
print(img)

# area = shapely.geometry.box(-114.4, 43, -114.3, 43.1)
# clip = img.rio.clip([area])