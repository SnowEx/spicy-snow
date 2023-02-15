import unittest

from shapely.geometry import box
import pandas as pd
import xarray
import pickle
import pandas.testing as pd_testing

import sys
from os.path import expanduser
sys.path.append(expanduser('.'))
from spicy_snow.download.sentinel1 import s1_img_search, download_s1_imgs
from spicy_snow.download.forest_cover import download_fcf, add_fcf

class TestSentinel1PreProcessing(unittest.TestCase):
    """
    Test functionality of preprocessing functions for Sentinel-1
    """

    
    
if __name__ == '__main__':
    unittest.main()