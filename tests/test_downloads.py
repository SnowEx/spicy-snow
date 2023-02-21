import unittest
from unittest.mock import MagicMock
from numpy.testing import assert_allclose
import pandas.testing as pd_testing

import numpy as np
import xarray as xr
import pandas as pd
import pickle

from shapely.geometry import box

import sys
from os.path import expanduser
sys.path.append(expanduser('.'))
from spicy_snow.download.sentinel1 import s1_img_search, combine_s1_images

class TestSentinel1Search(unittest.TestCase):
    """
    Test functionality of searching sentinel-1 granule names from dates and geometry.
    """

    area = box(-114.4, 43, -114.3, 43.1)
    dates = ('2019-12-28', '2020-02-02')
    search_result = s1_img_search(area, dates)

    def test_search_return_type(self, search_result = search_result):
        """
        Test return type of Sentinel-1 search
        """
        self.assertEqual(type(search_result), pd.DataFrame, "Seach results should return DataFrame")
    
    def test_search_return_number(self, search_result = search_result):
        """
        Test if sentinel-1 search returns expected number of results
        """
        # Expected number of Sentinel-1 images
        self.assertEqual(len(search_result), 14, "Seach results should contain XXX images")
    
    def test_search_return_values(self, search_result = search_result):
        """
        Test if sentinel-1 search returns correct images in dataframe
        """
        # Add in expected results
        with open('./tests/test_data/search_result', 'rb') as f:
            expected_result = pickle.load(f)
        try:
            pd_testing.assert_frame_equal(search_result, expected_result)
        except AssertionError as e:
            raise self.failureException("Search results should be this expected result.") from e
    
    def test_search_exceptions(self, area = area, dates = dates):
        """
        Test thrown exceptions/errors in sentinel 1 image searches.
        """
        # Incorrect number/type of inputs
        self.assertRaises(TypeError, s1_img_search, area, '2019-12-30')
        self.assertRaises(TypeError, s1_img_search, area, 1)
        self.assertRaises(TypeError, s1_img_search, area, [])
        self.assertRaises(TypeError, s1_img_search, [-115, 46, -115.5, 46.5], dates)
        self.assertRaises(TypeError, s1_img_search, dates)
        self.assertRaises(TypeError, s1_img_search, [], dates)
        self.assertRaises(TypeError, s1_img_search, ['-115', '46', '-115.5', '46.5'], dates)

        # Logical errors in inputs
        # dates are reverse
        self.assertRaises(ValueError, s1_img_search, area, ('2021-01-02','2020-01-02'))
        # dates are in future or too far in past
        self.assertRaises(IndexError, s1_img_search, area, ('2030-01-02','2032-01-02'))
        self.assertRaises(IndexError, s1_img_search, area, ('1999-01-02','2000-01-02'))
        # geometry outside of acceptable bounds or in non-geographic coordinates
        self.assertRaises(IndexError, s1_img_search, box(-115, -46, -115.5, -46.5), dates)
        self.assertRaises(IndexError, s1_img_search, box(300000, 50000, 300000, 55000), dates)

    das = {}
    granules = ['S1B_IW_GRDH_1SDV_20200201T013528_20200201T013553_020069_025FB3_6D5E',
    'S1B_IW_GRDH_1SDV_20200130T134920_20200130T134948_020047_025EF4_9218',
    'S1B_IW_GRDH_1SDV_20200127T012726_20200127T012751_019996_025D37_D0F0',
    'S1A_IW_GRDH_1SDV_20200126T013605_20200126T013634_030965_038E37_760E',
    'S1A_IW_GRDH_1SDV_20200124T135002_20200124T135027_030943_038D6F_FED5']
    for i in range(5):
        backscatter = np.random.randn(10, 10, 3)
        x = np.linspace(0, 9, 10)
        y = np.linspace(10, 19, 10)
        lon, lat = np.meshgrid(x, y)

        da = xr.DataArray(data = backscatter,
            dims = ["x","y","band"],
            coords = dict(
                lon=(["x", "y"], lon),
                lat=(["x", "y"], lat)))
        das[granules[i]] = da

    def test_combine_s1_imgs(self, das = das):
        ds = combine_s1_images(das)

        self.assertEqual(type(ds), xr.Dataset)

        self.assertEqual(len(ds.time), 5)
        self.assertEqual(len(ds.flight_dir), 5)
        self.assertEqual(len(ds.platform), 5)
        self.assertEqual(len(ds.relative_orbit), 5)

        self.assertEqual(len(ds.band), 3)

        self.assertEqual(ds.attrs['s1_units'], 'dB')
        self.assertEqual(ds.attrs['resolution'], '30')

        assert_allclose(das['S1B_IW_GRDH_1SDV_20200127T012726_20200127T012751_019996_025D37_D0F0'], ds.isel(time = 2)['s1'])

if __name__ == '__main__':
    unittest.main()