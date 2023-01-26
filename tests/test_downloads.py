import unittest

from shapely.geometry import box
import pandas as pd
import xarray
import pickle
import pandas.testing as pd_testing

import sys
from os.path import expanduser
sys.path.append(expanduser('~/Documents/spicy-snow'))
from spicy_snow.download.sentinel1 import s1_img_search, download_s1_imgs

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
        with open('./tests/test_data/search_result.pkl', 'rb') as f:
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

# class TestSentinel1Download(unittest.TestCase):
#     """
#     Test functionality of downloading sentinel-1 images from url search results.
#     """

#     # Create expected result dataframe (same as above)
#     with open('./tests/test_data/search_result.pkl', 'rb') as f:
#         search_results = pickle.load(f)
#     s1_dataset = download_s1_imgs(search_results.iloc[:2])

#     def test_download_return_type(self, dataset = s1_dataset):
#         """
#         Test return type of Sentinel-1 download
#         """
#         self.assertEqual(type(dataset), xarray.DataArray, "Download results should return xarray Dataset")
    
#     def test_download_return_number(self, dataset = s1_dataset):
#         """
#         Test if sentinel-1 download returns expected number of results
#         """
#         # Add in expected number of Sentinel-1 images
#         self.assertEqual(len(dataset), 2, "Download dataset should contain 2 images")
    
#     def test_download_return_values(self, dataset = s1_dataset):
#         """
#         Test if sentinel-1 download returns correct images in dataframe
#         """
#         # Add in expected results
#         expected_result = xarray.Dataset()
#         self.assertEqual(dataset, expected_result, "Download result should match.")

if __name__ == '__main__':
    unittest.main()